import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "rag-knowledge-base"

def get_index():
    """
    Returns the Pinecone Index object.
    Creates the index if it doesn't exist (Serverless).
    """
    # Check if index exists
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536, # OpenAI embedding size
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
    
    return pc.Index(INDEX_NAME)

def save_vectors(chunks, embeddings, metadata_list):
    """
    Upserts vectors to Pinecone.
    chunks: List[str]
    embeddings: List[List[float]]
    metadata_list: List[dict]
    """
    index = get_index()
    vectors_to_upsert = []
    
    for i, (text, vector, meta) in enumerate(zip(chunks, embeddings, metadata_list)):
        # Create a unique ID (e.g., doc_id + chunk_index)
        # Using a simple hash or filename_index for now
        doc_id = f"{meta.get('filename', 'doc')}_{i}"
        
        # Pinecone metadata cannot be too large, but 1 chunk of text fits fine
        meta['text'] = text 
        
        vectors_to_upsert.append({
            "id": doc_id,
            "values": vector,
            "metadata": meta
        })
        
    # Upsert in batches of 100 to avoid limits
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)
        
    print(f"Successfully saved {len(chunks)} chunks to Pinecone.")

def search_vectors(query_vector, top_k=5):
    """
    Query Pinecone for similar vectors.
    """
    index = get_index()
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # Return list of dicts with text and score
    return [
        {"text": match.metadata["text"], "score": match.score, "source": match.metadata.get("filename")} 
        for match in results.matches
    ]
def delete_vectors_by_session(session_id: str):
    """
    Deletes all vectors associated with a specific session.
    """
    index = get_index()
    try:
        # Pinecone allows deleting by metadata filter!
        index.delete(
            filter={
                "session_id": {"$eq": session_id}
            }
        )
        print(f"Deleted vectors for session: {session_id}")
    except Exception as e:
        print(f"Pinecone Delete Error: {e}")
