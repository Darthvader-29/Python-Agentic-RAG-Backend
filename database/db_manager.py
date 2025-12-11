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
    existing_indexes = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # MiniLM-L6-v2 = 384 dims
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        import time

        time.sleep(10)

    return pc.Index(INDEX_NAME)


def save_vectors(vectors: list[dict]):
    """
    Upserts vectors to Pinecone.
    vectors: List[dict] = [{"id": str, "values": list[float], "metadata": dict}]
    """
    index = get_index()
    batch_size = 100

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    print(f"Successfully saved {len(vectors)} vectors to Pinecone.")


def search_vectors(query_vector, top_k: int = 5, session_id: str | None = None):
    """
    Query Pinecone for similar vectors with optional session filter.
    """
    index = get_index()

    query_params: dict = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    if session_id:
        query_params["filter"] = {"session_id": {"$eq": session_id}}

    results = index.query(**query_params)

    return [
        {
            "text": match.metadata["text"],
            "score": match.score,
            "source": match.metadata.get("filename"),
            "chunk_index": match.metadata.get("chunk_index"),
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
                "session_id": {"$eq": session_id},
            }
        )
        print(f"Deleted vectors for session: {session_id}")
    except Exception as e:
        print(f"Pinecone Delete Error: {e}")


def list_s3_keys_for_session(session_id: str) -> list[str]:
    """
    Returns a list of unique S3 keys for all vectors in a session.
    Uses a broad query with dummy vector and filter on session_id.
    """
    index = get_index()
    try:
        # broad query: we only care about metadata.s3_key
        results = index.query(
            vector=[0.0] * 384,
            top_k=1000,
            filter={"session_id": {"$eq": session_id}},
            include_metadata=True,
        )
        keys = {
            m.metadata.get("s3_key")
            for m in results.matches
            if m.metadata and m.metadata.get("s3_key")
        }
        keys_list = list(keys)
        print(f"Found {len(keys_list)} S3 keys for session {session_id}")
        return keys_list
    except Exception as e:
        print(f"Error listing S3 keys for session {session_id}: {e}")
        return []
