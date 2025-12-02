import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

from database.doc_parser import DocumentParser
from database.db_manager import save_vectors
from integrations.uploadthing.client import download_file_to_temp

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# IMPORTANT: Must match Pinecone index dimension
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # or 1024/1536/3072, but must match Pinecone index config

async def process_file_pipeline(file_key: str, filename: str, session_id: str):
    """
    The Master Ingestion Function.
    1. Download from UploadThing
    2. Extract Text
    3. Chunk
    4. Embed with Gemini
    5. Save to Pinecone
    """
    temp_path = None
    try:
        print(f"--- Starting Ingestion for {filename} ---")
        
        # 1. Download
        temp_path = download_file_to_temp(file_key)
        
        # 2. Extract
        raw_text = DocumentParser.extract_content(temp_path, filename)
        print(f"Extracted {len(raw_text)} characters.")
        
        # 3. Chunking (Semantic)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        print(f"Created {len(chunks)} chunks.")
        
        if not chunks:
            print("No text found to chunk.")
            return

        # 4. Embeddings with Gemini
        # Batch using embed_content (Gemini supports list of contents)
        embeddings = []
        # process in batches to avoid very large requests
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document",
                    output_dimensionality=EMBEDDING_DIM,
                )
                # result.embeddings is a list of vectors for the batch
                batch_embeddings = [e.values for e in result.embeddings]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"[Embedding Error] batch {i}-{i+batch_size}: {e}")
                raise

        if len(embeddings) != len(chunks):
            print(f"[Warning] Embedding count ({len(embeddings)}) != chunk count ({len(chunks)})")

        # 5. Save to Pinecone
        metadata_list = []
        for i in range(len(chunks)):
            metadata_list.append({
                "filename": filename,
                "session_id": session_id,
                "chunk_index": i,
            })
        
        save_vectors(chunks, embeddings, metadata_list)
        print("--- Ingestion Complete ---")

    except Exception as e:
        print(f"Ingestion Failed: {str(e)}")
        raise e
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print("Temp file cleaned up.")
