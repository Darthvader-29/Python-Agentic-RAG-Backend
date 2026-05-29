import os

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database.db_manager import save_vectors
from database.doc_parser import DocumentParser
from integrations.huggingface.client import embed_batch  # NEW
from integrations.s3.client import download_s3_to_temp

# IMPORTANT: Must match Pinecone index dimension (MiniLM output is 384)
EMBEDDING_DIM = 384

logger = structlog.get_logger(__name__)


async def process_file_pipeline(file_key: str, filename: str, session_id: str):
    """
    The Master Ingestion Function.
    1. Download from S3
    2. Extract Text
    3. Chunk
    4. Embed with HuggingFace
    5. Save to Pinecone
    """
    temp_path = None
    try:
        logger.info("ingestion_start", filename=filename, s3_key=file_key)

        # 1. Download from S3
        temp_path = download_s3_to_temp(file_key)
        logger.info("ingestion_downloaded", temp_path=temp_path)

        # 2. Extract
        raw_text = DocumentParser.extract_content(temp_path, filename)
        logger.info("ingestion_extracted", chars=len(raw_text))

        if not raw_text.strip():
            logger.info("ingestion_empty", reason="no text extracted from document")
            return

        # 3. Chunking (Semantic)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        logger.info("ingestion_chunked", chunks=len(chunks))

        if not chunks:
            logger.info("ingestion_empty", reason="no valid chunks created")
            return

        # 4. Embeddings with HuggingFace (single call - client handles batching)
        logger.info("ingestion_embedding_start")
        embeddings = embed_batch(chunks, batch_size=32)
        logger.debug(
            "ingestion_embeddings",
            count=len(embeddings),
            dims=len(embeddings[0]) if embeddings else 0,
        )

        if len(embeddings) != len(chunks):
            logger.error(
                "ingestion_embedding_mismatch",
                embedding_count=len(embeddings),
                chunk_count=len(chunks),
            )
            raise ValueError("Embedding mismatch")

        # 5. Save to Pinecone - CORRECT FORMAT
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            vectors.append(
                {
                    "id": f"{session_id}_{filename.replace(' ', '_')}_{i:04d}",
                    "values": embedding,  # List[float] - Pinecone expects this
                    "metadata": {
                        "text": chunk,  # Store full chunk text for retrieval
                        "filename": filename,
                        "session_id": session_id,
                        "chunk_index": i,
                        "s3_key": file_key,
                    },
                }
            )

        save_vectors(vectors)  # Updated signature: takes list of dicts
        logger.info("ingestion_complete", vectors_saved=len(vectors))

    except Exception as e:
        logger.error("ingestion_failed", error=str(e), exc_info=True)
        raise

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info("ingestion_temp_cleanup", temp_path=temp_path)
