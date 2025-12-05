import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database.doc_parser import DocumentParser
from database.db_manager import save_vectors
from integrations.s3.client import download_s3_to_temp
from integrations.huggingface.client import embed_batch  # NEW

load_dotenv()

# IMPORTANT: Must match Pinecone index dimension (MiniLM output is 384)
EMBEDDING_DIM = 384

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
        print(f"--- Starting Ingestion for {filename} (S3: {file_key}) ---")
        
        # 1. Download from S3
        temp_path = download_s3_to_temp(file_key)
        print(f"Downloaded to temp: {temp_path}")
        
        # 2. Extract
        raw_text = DocumentParser.extract_content(temp_path, filename)
        print(f"Extracted {len(raw_text)} characters.")
        
        if not raw_text.strip():
            print("No text extracted from document.")
            return
        
        # 3. Chunking (Semantic)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        print(f"Created {len(chunks)} chunks.")
        
        if not chunks:
            print("No valid chunks created.")
            return

        # 4. Embeddings with HuggingFace (single call - client handles batching)
        print("Generating embeddings...")
        embeddings = embed_batch(chunks, batch_size=32)
        print(f"Generated {len(embeddings)} embeddings (dim={len(embeddings[0]) if embeddings else 0})")
        
        if len(embeddings) != len(chunks):
            print(f"[ERROR] Embedding count ({len(embeddings)}) != chunk count ({len(chunks)})")
            raise ValueError("Embedding mismatch")

        # 5. Save to Pinecone - CORRECT FORMAT
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{session_id}_{filename.replace(' ', '_')}_{i:04d}",
                "values": embedding,  # List[float] - Pinecone expects this
                "metadata": {
                    "text": chunk,  # Store full chunk text for retrieval
                    "filename": filename,
                    "session_id": session_id,
                    "chunk_index": i,
                    "s3_key": file_key
                }
            })
        
        save_vectors(vectors)  # Updated signature: takes list of dicts
        print(f"--- Saved {len(vectors)} vectors to Pinecone ---")

    except Exception as e:
        print(f"Ingestion Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print("Temp file cleaned up.")
