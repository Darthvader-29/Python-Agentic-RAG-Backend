import uuid
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from auth.keys_router import router as keys_router
from auth.router import router as auth_router
from auth.security import decode_token
from components.generation import generate_final_response
from components.preprocessing import EMBEDDING_DIM
from components.retrieval import retrieve_context
from components.router import route_query
from config import settings
from database import repository as repo
from database.db_manager import PineconeClient
from database.models import DocumentStatus, User
from database.session import build_engine, build_sessionmaker
from dependencies import (
    get_db_session,
    get_embedding_client,
    get_pinecone_client,
    get_s3_client,
    get_web_search_client,
)
from exceptions import AppException, app_exception_handler
from integrations.duckduckgo.client import DuckDuckGoClient
from integrations.huggingface.client import HuggingFaceClient
from integrations.s3.client import S3Client
from llm.base import LLMProvider
from llm.dependencies import get_llm_provider
from logging_config import configure_logging
from worker.tasks import ingest_document

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
    # Phase 5: one pooled Redis client per process (lazy from_url — no network here)
    app.state.redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("clients_initialized", environment=settings.ENVIRONMENT)
    yield
    await app.state.redis.aclose()
    await app.state.db_engine.dispose()


app = FastAPI(
    title="Dynamic Knowledge RAG Engine",
    version="1.0.0",
    description="Multi-agent RAG with Pinecone, S3, and Gemini",
    lifespan=lifespan,
)

app.add_exception_handler(AppException, app_exception_handler)

# Phase 3: explicit allow-list — "*" + allow_credentials is rejected by browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Phase 5: per-user rate limiting (Redis-backed in prod, memory:// in tests) ──
def _rate_limit_key(request: Request) -> str:
    """Throttle per authenticated user so one tenant can't drain another's budget;
    fall back to client IP for anonymous traffic. The bearer is decoded best-effort —
    an invalid/expired token falls through to IP (the route's auth dependency still 401s)."""
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        try:
            sub = decode_token(auth[7:]).get("sub")
            if sub:
                return f"user:{sub}"
        except Exception:
            pass
    return get_remote_address(request)


# storage_uri resolves to REDIS_URL in prod (shared across instances) and "memory://"
# in tests. Per-route @limiter.limit decorators below; no global default → /health is exempt.
limiter = Limiter(key_func=_rate_limit_key, storage_uri=settings.rate_limit_storage_uri)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Phase 3: auth + key management routers
app.include_router(auth_router)
app.include_router(keys_router)


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


# ── Phase 5: presigned upload (frontend M8 contract) ──
class PresignRequest(BaseModel):
    filename: str
    content_type: str | None = None
    session_id: str | None = None  # optional; a new one is created and returned if absent


class PresignResponse(BaseModel):
    document_id: str
    upload_url: str
    s3_key: str
    session_id: str


class ConfirmRequest(BaseModel):
    document_id: str
    s3_key: str


class DocumentStatusResponse(BaseModel):
    id: str
    filename: str
    status: str  # pending | processing | ready | failed
    s3_key: str
    session_id: str
    error: str | None = None


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


async def _resolve_session(db: AsyncSession, session_id: str | None, current_user: User) -> str:
    """Create or verify a session owned by the current user; return the session_id."""
    sid = session_id or str(uuid.uuid4())
    existing = await repo.get_session(db, sid)
    if existing is None:
        await repo.create_session(db, sid, current_user.id)
    elif existing.user_id is not None and existing.user_id != current_user.id:
        raise HTTPException(403, "session does not belong to the current user")
    elif existing.user_id is None:
        existing.user_id = current_user.id
    return sid


async def _owns_document(db: AsyncSession, doc, current_user: User) -> bool:
    """A document is the caller's if its session is unowned or owned by the caller."""
    session = await repo.get_session(db, doc.session_id)
    return session is None or session.user_id is None or session.user_id == current_user.id


async def _upload_multipart(request: Request, current_user: User, s3: S3Client, db: AsyncSession):
    """Legacy flag-OFF path: bytes pass through the API, then ingestion is enqueued."""
    form = await request.form()
    file = form.get("file")
    if file is None or not hasattr(file, "filename"):
        raise HTTPException(422, "missing file")
    session_id = await _resolve_session(db, form.get("session_id"), current_user)
    filename = file.filename or "upload"
    s3_key = await s3.upload_fileobj(file.file, filename)
    doc = await repo.create_document(db, session_id=session_id, s3_key=s3_key, filename=filename)
    await db.commit()  # persist before enqueue so a separate worker can read the row
    ingest_document.delay(
        document_id=doc.id, s3_key=s3_key, filename=filename, session_id=session_id
    )
    return UploadResponse(
        status="processing",
        message=f"{filename} uploaded and ingestion started.",
        session_id=session_id,
        s3_key=s3_key,
    )


async def _upload_presign(request: Request, current_user: User, s3: S3Client, db: AsyncSession):
    """Presigned flag-ON path: issue a PUT URL; the client uploads direct to storage."""
    payload = PresignRequest.model_validate(await request.json())
    session_id = await _resolve_session(db, payload.session_id, current_user)
    s3_key = s3.make_user_key(current_user.id, payload.filename)
    upload_url = await s3.generate_presigned_url(s3_key)
    doc = await repo.create_document(
        db, session_id=session_id, s3_key=s3_key, filename=payload.filename
    )
    await db.commit()
    return PresignResponse(
        document_id=doc.id, upload_url=upload_url, s3_key=s3_key, session_id=session_id
    )


@app.post("/api/upload")
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload(
    request: Request,
    current_user: User = Depends(get_current_user),
    s3: S3Client = Depends(get_s3_client),
    db: AsyncSession = Depends(get_db_session),
):
    """Two transports on one path (the frontend's M8 flag picks one):

    - ``multipart/form-data`` → legacy passthrough upload (bytes via the API), then enqueue.
    - ``application/json``     → presigned PUT: ``{document_id, upload_url, s3_key, session_id}``.

    Both record a Postgres ``documents`` row and route ingestion through the Celery worker.
    """
    try:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            return await _upload_multipart(request, current_user, s3, db)
        return await _upload_presign(request, current_user, s3, db)
    except HTTPException:
        raise
    except Exception as exc:
        raise AppException(status_code=500, detail="Upload failed unexpectedly.") from exc


@app.post("/api/upload/confirm")
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def confirm_upload(
    request: Request,
    payload: ConfirmRequest,
    current_user: User = Depends(get_current_user),
    s3: S3Client = Depends(get_s3_client),
    db: AsyncSession = Depends(get_db_session),
):
    """Verify the presigned object landed (head_object), then enqueue ingestion (M8 step 3)."""
    try:
        doc = await repo.get_document(db, payload.document_id)
        if doc is None or not await _owns_document(db, doc, current_user):
            raise HTTPException(404, "document not found")
        if not await s3.object_exists(payload.s3_key):
            await repo.set_document_status(db, s3_key=payload.s3_key, status=DocumentStatus.FAILED)
            raise HTTPException(409, "object not uploaded")
        ingest_document.delay(
            document_id=doc.id,
            s3_key=payload.s3_key,
            filename=doc.filename,
            session_id=doc.session_id,
        )
        return {"document_id": doc.id, "status": "queued"}
    except (AppException, HTTPException):
        raise
    except Exception as exc:
        raise AppException(status_code=500, detail="Confirm failed unexpectedly.") from exc


@app.get("/api/documents/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Poll a document's ingestion status (M8 step 4). status ∈ pending|processing|ready|failed."""
    doc = await repo.get_document(db, document_id)
    if doc is None or not await _owns_document(db, doc, current_user):
        raise HTTPException(404, "document not found")
    return DocumentStatusResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status.value,
        s3_key=doc.s3_key,
        session_id=doc.session_id,
    )


# ========= CHAT =========


@app.post("/api/chat")
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def chat(
    request: Request,
    payload: ChatRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    current_user: User = Depends(get_current_user),
    pinecone: PineconeClient = Depends(get_pinecone_client),
    embedder: HuggingFaceClient = Depends(get_embedding_client),
    web: DuckDuckGoClient = Depends(get_web_search_client),
    db: AsyncSession = Depends(get_db_session),
):
    """Main chat endpoint: route → relevance check → retrieve → generate."""
    try:
        session_id = payload.session_id or str(uuid.uuid4())
        logger.info(
            "chat_request",
            message_preview=payload.message[:50],
            web_search_allowed=payload.web_search_allowed,
            session_id=session_id[:8],
        )

        # Phase 3: ownership check — create session with owner or verify existing ownership
        existing = await repo.get_session(db, session_id)
        if existing is None:
            await repo.create_session(db, session_id, current_user.id)
        elif existing.user_id is not None and existing.user_id != current_user.id:
            raise HTTPException(403, "session does not belong to the current user")
        elif existing.user_id is None:
            existing.user_id = current_user.id

        # Phase 4: get has_documents from DB for routing, then check Pinecone relevance
        has_documents_db = await repo.session_has_documents(db, session_id)

        base_route = await route_query(
            provider,
            payload.message,
            has_documents=has_documents_db,
            web_search_allowed=payload.web_search_allowed,
        )

        has_docs, docs_relevant = await check_docs_relevant(
            payload.message, session_id, pinecone, embedder
        )

        final_route = decide_combined_route(
            base_route,
            has_documents=has_docs,
            docs_relevant=docs_relevant,
            web_allowed=payload.web_search_allowed,
        )

        logger.info(
            "routing_decision",
            base_route=base_route,
            has_docs=has_docs,
            docs_relevant=docs_relevant,
            final_route=final_route,
        )

        context = await retrieve_context(
            payload.message,
            final_route,
            session_id,
            payload.web_search_allowed,
            pinecone,
            embedder,
            web,
        )

        answer = await generate_final_response(
            provider,
            payload.message,
            context,
            final_route,  # type: ignore[arg-type]
        )

        return {
            "answer": answer,
            "route": final_route,
            "context_count": len(context),
            "session_id": session_id,
        }
    except (AppException, HTTPException):
        raise
    except Exception as e:
        logger.error("chat_failed", exc_info=True)
        raise AppException(
            status_code=500, detail="free tier Limit Reached for API please try again later"
        ) from e


# ========= CLEANUP =========


@app.post("/api/cleanup")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def cleanup_session(
    request: Request,
    payload: CleanupRequest,
    current_user: User = Depends(get_current_user),
    s3: S3Client = Depends(get_s3_client),
    pinecone: PineconeClient = Depends(get_pinecone_client),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete Pinecone vectors, S3 objects, and Postgres state for a session."""
    try:
        logger.info("cleanup_request", session_id=payload.session_id)

        # Phase 3: ownership check
        session = await repo.get_session(db, payload.session_id)
        if session is None:
            raise HTTPException(404, "session not found")
        if session.user_id is not None and session.user_id != current_user.id:
            raise HTTPException(403, "session does not belong to the current user")

        keys = await repo.list_s3_keys_for_session(db, payload.session_id)
        await pinecone.delete_vectors_by_session(payload.session_id)
        if keys:
            await s3.delete_objects(keys)
        await repo.delete_session(db, payload.session_id)

        return {
            "status": "cleaned",
            "session_id": payload.session_id,
            "deleted_files": len(keys),
        }
    except (AppException, HTTPException):
        raise
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
