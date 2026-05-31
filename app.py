import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from components.generation import generate_final_response
from components.preprocessing import EMBEDDING_DIM, process_file_pipeline
from components.retrieval import retrieve_context
from components.router import route_query
from config import settings
from database import repository as repo
from database.db_manager import PineconeClient
from database.session import build_engine, build_sessionmaker
from dependencies import (
    get_db_session,
    get_db_sessionmaker,
    get_embedding_client,
    get_pinecone_client,
    get_s3_client,
    get_web_search_client,
)
from exceptions import AppException, app_exception_handler
from integrations.duckduckgo.client import DuckDuckGoClient
from integrations.huggingface.client import HuggingFaceClient
from integrations.s3.client import S3Client
from logging_config import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.pinecone = PineconeClient.from_settings(settings)
    await app.state.pinecone.ensure_index()
    app.state.s3 = S3Client.from_settings(settings)
    app.state.embedder = HuggingFaceClient.from_settings(settings)
    app.state.web = DuckDuckGoClient()
    app.state.db_engine = build_engine(settings)
    app.state.db_sessionmaker = build_sessionmaker(app.state.db_engine)
    logger.info("clients_initialized", environment=settings.ENVIRONMENT)
    yield
    await app.state.db_engine.dispose()


app = FastAPI(
    title="Dynamic Knowledge RAG Engine",
    version="1.0.0",
    description="Multi-agent RAG with Pinecone, S3, and Gemini",
    lifespan=lifespan,
)

app.add_exception_handler(AppException, app_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ========= MODELS =========


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    web_search_allowed: bool = True


class CleanupRequest(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    status: str
    message: str
    session_id: str
    s3_key: str


# ========= HELPERS FOR COMBINED ROUTING =========


RAG_THRESHOLD = 0.4


async def get_query_embedding(text: str, embedder: HuggingFaceClient) -> list[float]:
    """Get a single 384-dim embedding using the same model as ingestion."""
    embs = await embedder.embed_batch([text], batch_size=1)
    return embs[0] if embs else [0.0] * EMBEDDING_DIM


async def check_docs_relevant(
    query: str,
    session_id: str,
    pinecone: PineconeClient,
    embedder: HuggingFaceClient,
) -> tuple[bool, bool]:
    """Returns (has_documents, docs_relevant)."""
    try:
        q_emb = await get_query_embedding(query, embedder)
        results = await pinecone.search_vectors(q_emb, top_k=3, session_id=session_id)
        if not results:
            return False, False
        top_score = results[0]["score"]
        docs_relevant = top_score >= RAG_THRESHOLD
        logger.info(
            "pinecone_relevance_check",
            top_score=round(top_score, 3),
            docs_relevant=docs_relevant,
        )
        return True, docs_relevant
    except Exception:
        logger.error("doc_relevance_check_failed", exc_info=True)
        return False, False


def decide_combined_route(
    base_route: str,
    has_documents: bool,
    docs_relevant: bool,
    web_allowed: bool,
) -> str:
    """Combine base route (RAG/WEB/DIRECT) with doc relevance into a final route label."""
    base = base_route.upper()

    if has_documents and docs_relevant:
        if base == "WEB" and web_allowed:
            return "WEB+RAG"
        if base == "DIRECT":
            return "DIRECT+RAG"
        return "RAG"

    if web_allowed:
        return "DIRECT+WEB"

    return "DIRECT"


# ========= UPLOAD + INGEST =========


@app.post("/api/upload", response_model=UploadResponse)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    s3: S3Client = Depends(get_s3_client),
    embedder: HuggingFaceClient = Depends(get_embedding_client),
    pinecone: PineconeClient = Depends(get_pinecone_client),
    db: AsyncSession = Depends(get_db_session),
    session_factory=Depends(get_db_sessionmaker),
):
    """Upload file to S3, record session/document in Postgres, start ingestion."""
    try:
        filename = file.filename or "upload"
        s3_key = await s3.upload_fileobj(file.file, filename)

        await repo.get_or_create_session(db, session_id)
        await repo.create_document(db, session_id=session_id, s3_key=s3_key, filename=filename)

        # sessionmaker (not the request session) is safe to pass to background tasks
        background_tasks.add_task(
            process_file_pipeline,
            s3_key,
            filename,
            session_id,
            s3,
            embedder,
            pinecone,
            session_factory,
        )

        return UploadResponse(
            status="uploaded",
            message=f"{file.filename} uploaded and ingestion started.",
            session_id=session_id,
            s3_key=s3_key,
        )
    except Exception as exc:
        raise AppException(status_code=500, detail="Upload failed unexpectedly.") from exc


# ========= CHAT =========


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    pinecone: PineconeClient = Depends(get_pinecone_client),
    embedder: HuggingFaceClient = Depends(get_embedding_client),
    web: DuckDuckGoClient = Depends(get_web_search_client),
    db: AsyncSession = Depends(get_db_session),
):
    """Main chat endpoint: route → relevance check → retrieve → generate."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(
            "chat_request",
            message_preview=request.message[:50],
            web_search_allowed=request.web_search_allowed,
            session_id=session_id[:8],
        )

        base_route = await route_query(request.message, session_id, request.web_search_allowed, db)

        has_docs, docs_relevant = await check_docs_relevant(
            request.message, session_id, pinecone, embedder
        )

        final_route = decide_combined_route(
            base_route,
            has_documents=has_docs,
            docs_relevant=docs_relevant,
            web_allowed=request.web_search_allowed,
        )

        logger.info(
            "routing_decision",
            base_route=base_route,
            has_docs=has_docs,
            docs_relevant=docs_relevant,
            final_route=final_route,
        )

        context = await retrieve_context(
            request.message,
            final_route,
            session_id,
            request.web_search_allowed,
            pinecone,
            embedder,
            web,
        )

        answer = await generate_final_response(
            request.message,
            context,
            final_route,  # type: ignore[arg-type]
        )

        return {
            "answer": answer,
            "route": final_route,
            "context_count": len(context),
            "session_id": session_id,
        }
    except AppException:
        raise
    except Exception as e:
        logger.error("chat_failed", exc_info=True)
        raise AppException(
            status_code=500, detail="free tier Limit Reached for API please try again later"
        ) from e


# ========= CLEANUP =========


@app.post("/api/cleanup")
async def cleanup_session(
    request: CleanupRequest,
    s3: S3Client = Depends(get_s3_client),
    pinecone: PineconeClient = Depends(get_pinecone_client),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete Pinecone vectors, S3 objects, and Postgres state for a session."""
    try:
        logger.info("cleanup_request", session_id=request.session_id)

        keys = await repo.list_s3_keys_for_session(db, request.session_id)
        await pinecone.delete_vectors_by_session(request.session_id)
        if keys:
            await s3.delete_objects(keys)
        await repo.delete_session(db, request.session_id)

        return {
            "status": "cleaned",
            "session_id": request.session_id,
            "deleted_files": len(keys),
        }
    except Exception as e:
        logger.error("cleanup_failed", exc_info=True)
        raise AppException(status_code=500, detail="Cleanup failed unexpectedly.") from e


# ========= FRONTEND + HEALTH =========


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
