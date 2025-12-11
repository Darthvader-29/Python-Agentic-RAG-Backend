import os
import uuid
from typing import List, Optional

from exceptions import AppException, app_exception_handler
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from components.router import route_query
from components.retrieval import retrieve_context
from components.generation import generate_final_response
from components.preprocessing import process_file_pipeline
from database.db_manager import (
    delete_vectors_by_session,
    list_s3_keys_for_session,
    search_vectors,
)
from integrations.s3.client import upload_fileobj_to_s3, delete_s3_objects
from integrations.huggingface.client import embed_batch  # for query embedding

load_dotenv()

app = FastAPI(
    title="Dynamic Knowledge RAG Engine",
    version="1.0.0",
    description="Multi-agent RAG with Pinecone, S3, and Gemini",
)

app.add_exception_handler(AppException, app_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ========= MODELS =========
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    web_search_allowed: bool = True


class CleanupRequest(BaseModel):
    session_id: str
    file_keys: Optional[List[str]] = []


class UploadResponse(BaseModel):
    status: str
    message: str
    session_id: str
    s3_key: str


# ========= HELPERS FOR COMBINED ROUTING =========
RAG_THRESHOLD = 0.6 # cosine similarity threshold for "relevant doc"


def get_query_embedding(text: str) -> list[float]:
    """
    Get a single 384-dim embedding for the query using the same model as ingestion.
    embed_batch returns List[List[float]], so take first element.
    """
    embs = embed_batch([text], batch_size=1)
    return embs[0] if embs else [0.0] * 384


def check_docs_relevant(query: str, session_id: str) -> tuple[bool, bool]:
    """
    Returns (has_documents, docs_relevant).

    has_documents: any vectors exist for this session.
    docs_relevant: top match score >= RAG_THRESHOLD.
    """
    try:
        q_emb = get_query_embedding(query)
        results = search_vectors(q_emb, top_k=3, session_id=session_id)
        if not results:
            return False, False
        has_docs = True
        top_score = results[0]["score"]
        docs_relevant = top_score >= RAG_THRESHOLD
        print(
            f"[Routing] Pinecone relevance: top_score={top_score:.3f}, "
            f"docs_relevant={docs_relevant}"
        )
        return has_docs, docs_relevant
    except Exception as e:
        print(f"[Routing] Doc relevance check failed: {e}")
        return False, False


def decide_combined_route(
    base_route: str,
    has_documents: bool,
    docs_relevant: bool,
    web_allowed: bool,
) -> str:
    """
    Combine base route (RAG/WEB/DIRECT) with doc relevance into a final route label.

    Possible outputs: "RAG", "WEB+RAG", "DIRECT+RAG", "DIRECT+WEB", "DIRECT", "WEB".
    """
    base = base_route.upper()

    if has_documents and docs_relevant:
        # documents useful – let them participate
        if base == "WEB" and web_allowed:
            return "WEB+RAG"
        if base == "DIRECT":
            return "DIRECT+RAG"
        # for base RAG or anything else, just RAG
        return "RAG"

    # documents not useful or don't exist
    if web_allowed:
        if base in ("WEB", "RAG"):
            # question seems webby or RAG but docs not helpful -> use web only + model
            return "DIRECT+WEB"
        if base == "DIRECT":
            return "DIRECT+WEB"
        return "DIRECT+WEB"

    # no web allowed, no useful docs -> pure DIRECT
    return "DIRECT"


# ========= UPLOAD + INGEST (S3) =========
@app.post("/api/upload", response_model=UploadResponse)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Upload file to S3 and start ingestion in background.
    Uses the session_id provided by the frontend so cleanup works.
    """
    try:
        s3_key = upload_fileobj_to_s3(file.file, file.filename)

        background_tasks.add_task(
            process_file_pipeline,
            s3_key,
            file.filename,
            session_id,
        )

        return UploadResponse(
            status="uploaded",
            message=f"{file.filename} uploaded and ingestion started.",
            session_id=session_id,
            s3_key=s3_key,
        )
    except Exception:
        raise AppException(status_code=500, detail="Upload failed unexpectedly.")


# ========= CHAT =========
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint:
    1) Router decides base route: RAG / WEB / DIRECT
    2) Quick Pinecone check decides whether docs are relevant
    3) Combined route (e.g., WEB+RAG, DIRECT+WEB) is chosen
    4) Retrieval & generation use that combined route
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        print(
            f"[Chat] '{request.message[:50]}...' | web={request.web_search_allowed} | session={session_id[:8]}"
        )

        # 1. Base route from Gemini + fallback router
        base_route = await route_query(
            request.message,
            session_id,
            request.web_search_allowed,
        )

        # 2. Doc relevance check (single embed + Pinecone query)
        has_docs, docs_relevant = check_docs_relevant(
            request.message,
            session_id,
        )

        # 3. Decide combined route
        final_route = decide_combined_route(
            base_route,
            has_documents=has_docs,
            docs_relevant=docs_relevant,
            web_allowed=request.web_search_allowed,
        )

        print(
            f"[Routing] base={base_route}, has_docs={has_docs}, "
            f"docs_relevant={docs_relevant} -> final_route={final_route}"
        )

        # 4. Retrieve according to combined route
        context = await retrieve_context(
            request.message,
            final_route,
            session_id,
            request.web_search_allowed,
        )

        # 5. Generate
        answer = await generate_final_response(
            request.message,
            context,
            final_route,
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
        print(f"[Chat Error] {e}")
        raise AppException(status_code=500, detail="free tier Limit Reached for API please try again later")


# ========= CLEANUP =========
@app.post("/api/cleanup")
async def cleanup_session(request: CleanupRequest):
    """
    Cleanup endpoint called on tab close/reset:
    - Delete vectors for this session from Pinecone
    - Delete uploaded files from S3
    """
    try:
        print(f"[Cleanup] Session {request.session_id}")

        file_keys = request.file_keys or list_s3_keys_for_session(request.session_id)

        delete_vectors_by_session(request.session_id)
        delete_s3_objects(file_keys)

        return {
            "status": "cleaned",
            "session_id": request.session_id,
            "deleted_files": len(file_keys or []),
        }
    except Exception as e:
        print(f"[Cleanup Error] {e}")
        raise AppException(status_code=500, detail="Cleanup failed unexpectedly.")


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
