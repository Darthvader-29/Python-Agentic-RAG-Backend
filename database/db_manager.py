from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = PINECONE_INDEX_NAME

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
            dimension=384,  # ✅ FIXED: MiniLM-L6-v2 = 384 dims (not 768)
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
        # Wait for index to be ready
        import time
        time.sleep(10)
    
    return pc.Index(INDEX_NAME)

def save_vectors(vectors: list[dict]):
    """
    Upserts vectors to Pinecone. ✅ UPDATED SIGNATURE
    vectors: List[dict] = [{"id": str, "values": list[float], "metadata": dict}]
    """
    index = get_index()
    batch_size = 100
    
    # Upsert in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
    
    print(f"Successfully saved {len(vectors)} vectors to Pinecone.")

def search_vectors(query_vector, top_k=5, session_id: str = None):
    """
    Query Pinecone for similar vectors with optional session filter.
    """
    index = get_index()
    
    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True
    }
    
    # Add session filter for RAG isolation
    if session_id:
        query_params["filter"] = {"session_id": {"$eq": session_id}}
    
    results = index.query(**query_params)
    
    # Return list of dicts with text and score
    return [
        {
            "text": match.metadata["text"], 
            "score": match.score, 
            "source": match.metadata.get("filename"),
            "chunk_index": match.metadata.get("chunk_index")
        } 
        for match in results.matches
    ]

def delete_vectors_by_session(session_id: str):
    """
    Deletes all vectors associated with a specific session.
    """
    index = get_index()
    try:
        index.delete(
            filter={
                "session_id": {"$eq": session_id}
            }
        )
        print(f"Deleted vectors for session: {session_id}")
    except Exception as e:
        print(f"Pinecone Delete Error: {e}")
