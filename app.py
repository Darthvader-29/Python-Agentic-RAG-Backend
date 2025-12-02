import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Local imports ---
from components.router import route_query
from components.retrieval import retrieve_context
from components.generation import generate_final_response
from components.preprocessing import process_file_pipeline
from database.db_manager import delete_vectors_by_session
from integrations.uploadthing.client import (
    UploadThingClient,
    download_file_to_temp,   # kept for completeness, not strictly needed in app.py now
)

load_dotenv()

# ============ FASTAPI APP ============
app = FastAPI(
    title="Dynamic Knowledge RAG Engine",
    version="1.0.0",
    description="Multi-agent RAG with Pinecone, UploadThing, and Gemini",
)

# CORS (relax for now; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============ REQUEST MODELS ============
class SignUploadRequest(BaseModel):
    filename: str
    file_size: int
    file_type: str


class IngestRequest(BaseModel):
    file_key: str
    filename: str
    session_id: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    web_search_allowed: bool = True


class CleanupRequest(BaseModel):
    session_id: str
    file_keys: Optional[List[str]] = []


# ============ UPLOAD FLOW ============

@app.post("/api/upload/sign")
async def sign_upload(request: SignUploadRequest):
    """
    Step 1: Get presigned URL from UploadThing so the frontend
    can upload directly to cloud storage.
    """
    try:
        client = UploadThingClient()
        files_payload = [{
            "name": request.filename,
            "size": request.file_size,
            "type": request.file_type,
        }]
        upload_data = client.request_presigned_urls(files_payload)
        return upload_data[0]
    except Exception as e:
        print("[/api/upload/sign ERROR]", repr(e))
        raise HTTPException(status_code=500, detail=f"Signing failed: {str(e)}")



@app.post("/api/ingest")
async def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Step 3: After upload completes, start async ingestion:
    - Download from UploadThing
    - Parse PDF/DOCX
    - Chunk + embed with Gemini
    - Save vectors to Pinecone
    """
    try:
        background_tasks.add_task(
            process_file_pipeline,
            request.file_key,
            request.filename,
            request.session_id,
        )
        return {
            "status": "ingestion_started",
            "message": f"Processing {request.filename} in background...",
            "session_id": request.session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ============ CHAT PIPELINE ============

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint:
    1) Router decides RAG / WEB / DIRECT
    2) Retrieval fetches context if needed
    3) Generation builds final answer
    """
    try:
        # Generate a session id if UI didn't send one yet
        session_id = request.session_id or str(uuid.uuid4())

        print(
            f"[Chat] '{request.message}' | web={request.web_search_allowed} | session={session_id[:8]}"
        )

        # 1. Route
        route_decision = route_query(request.message, request.web_search_allowed)

        # 2. Retrieve
        retrieval_result = retrieve_context(
            route_decision,
            request.message,
            request.web_search_allowed,
        )

        # 3. Generate
        answer = generate_final_response(
            route_decision,
            request.message,
            retrieval_result,
        )

        return {
            "answer": answer,
            "route": route_decision.decision,
            "reasoning": route_decision.reasoning,
            "sources": retrieval_result.sources[:3],
            "has_context": len(retrieval_result.contexts) > 0,
            "session_id": session_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ============ CLEANUP ============

@app.post("/api/cleanup")
async def cleanup_session(request: CleanupRequest):
    """
    Cleanup endpoint called when the tab closes or user resets:
    - Delete vectors for this session from Pinecone
    - Delete uploaded files from UploadThing (optional)
    """
    try:
        print(f"[Cleanup] Session {request.session_id}")

        # 1. Pinecone vectors
        delete_vectors_by_session(request.session_id)

        # 2. UploadThing files
        if request.file_keys:
            client = UploadThingClient()
            client.delete_files(request.file_keys)
            print(f"[Cleanup] Deleted {len(request.file_keys)} files from UploadThing")

        return {
            "status": "cleaned",
            "session_id": request.session_id,
            "deleted_files": len(request.file_keys or []),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# ============ FRONTEND + HEALTH ============

@app.get("/")
async def root():
    """Serve minimal HTML frontend."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
