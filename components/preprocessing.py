import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from database.doc_parser import DocumentParser
from database.db_manager import save_vectors
from integrations.uploadthing.client import download_file_to_temp

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def process_file_pipeline(file_key: str, filename: str, session_id: str):
    """
    The Master Ingestion Function.
    1. Download from UploadThing
    2. Extract Text
    3. Chunk
    4. Embed
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
        # We use overlap to keep context between chunks
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

        # 4. Embeddings (Batching could be added for huge files, simple for now)
        # OpenAI text-embedding-3-small
        response = openai_client.embeddings.create(
            input=chunks,
            model="text-embedding-3-small"
        )
        embeddings = [item.embedding for item in response.data]
        
        # 5. Save to Pinecone
        # Prepare metadata for each chunk
        metadata_list = []
        for i in range(len(chunks)):
            metadata_list.append({
                "filename": filename,
                "session_id": session_id, # For cleanup!
                "chunk_index": i
            })
            
        save_vectors(chunks, embeddings, metadata_list)
        print("--- Ingestion Complete ---")

    except Exception as e:
        print(f"Ingestion Failed: {str(e)}")
        raise e
        
    finally:
        # Cleanup local temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print("Temp file cleaned up.")
