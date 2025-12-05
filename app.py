import os
import uuid
from typing import List, Optional

from exceptions import AppException, app_exception_handler
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from components.router import route_query
from components.retrieval import retrieve_context
from components.generation import generate_final_response
from components.preprocessing import process_file_pipeline
from database.db_manager import delete_vectors_by_session
from integrations.s3.client import upload_fileobj_to_s3, delete_s3_objects

load_dotenv()

app = FastAPI(
    title="Dynamic Knowledge RAG Engine",
    version="1.0.0",
    description="Multi-agent RAG with Pinecone, S3, and Gemini",
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

# ========= UPLOAD + INGEST (S3) =========
@app.post("/api/upload", response_model=UploadResponse)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload file to S3 and start ingestion in background.
    """
    try:
        s3_key = upload_fileobj_to_s3(file.file, file.filename)
        session_id = str(uuid.uuid4())

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
    except Exception as e:
        # For production, log the error e
        raise AppException(status_code=500, detail="Upload failed unexpectedly.")

# ========= CHAT =========
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint:
    1) Router decides RAG / WEB / DIRECT
    2) Retrieval fetches context if needed
    3) Generation builds final answer
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        print(
            f"[Chat] '{request.message[:50]}...' | web={request.web_search_allowed} | session={session_id[:8]}"
        )

        # 1. Route ✅ FIXED: Correct args + await
        route_decision = await route_query(request.message, session_id, request.web_search_allowed)

        # 2. Retrieve ✅ FIXED: Correct args + await
        context = await retrieve_context(
            request.message,
            route_decision,
            session_id,
            request.web_search_allowed
        )

        # 3. Generate ✅ FIXED: Correct args + await
        answer = await generate_final_response(request.message, context, route_decision)

        return {
            "answer": answer,
            "route": route_decision,
            "context_count": len(context),
            "session_id": session_id,
        }

    except Exception as e:
        print(f"[Chat Error] {e}")
        # For production, log the error e
        raise AppException(status_code=500, detail="Chat failed unexpectedly.")

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
        delete_vectors_by_session(request.session_id)
        delete_s3_objects(request.file_keys or [])
        return {
            "status": "cleaned",
            "session_id": request.session_id,
            "deleted_files": len(request.file_keys or []),
        }
    except Exception as e:
        # For production, log the error e
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
