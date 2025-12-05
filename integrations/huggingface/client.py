"""
Hugging Face Inference API client for sentence-transformers/all-MiniLM-L6-v2 embeddings.
Uses huggingface_hub.InferenceClient with feature-extraction pipeline.
Free tier, 384-dimensional embeddings.
"""
from typing import List
import numpy as np
from huggingface_hub import InferenceClient
import os

# Optional: Use HF token for higher rate limits (free tier still works without)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Set in .env or environment

client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=HF_TOKEN,
)

def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Embed multiple texts using HF Inference API.
    
    Args:
        texts: List of text strings to embed
        batch_size: Process in batches to respect rate limits
        
    Returns:
        List of 384-dim embedding vectors
    """
    if not texts:
        return []
    
    embeddings = []
    
    # Process in batches to avoid rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Get embeddings for batch
        batch_embeds = client.feature_extraction(
            batch,
            normalize=True,  # L2 normalize for cosine similarity
        )
        
        # Convert to list of lists if needed
        if isinstance(batch_embeds, np.ndarray):
            batch_embeds = batch_embeds.tolist()
        
        embeddings.extend(batch_embeds)
    
    return embeddings

def embed_single(text: str) -> List[float]:
    """Embed a single text (convenience wrapper)."""
    return embed_batch([text], batch_size=1)[0]

if __name__ == "__main__":
    # Test the embedding function
    test_texts = [
        "This is a test sentence.",
        "Hugging Face embeddings work great for RAG.",
        "FastAPI + Pinecone + Gemini = powerful RAG system."
    ]
    
    embeds = embed_batch(test_texts)
    print(f"Generated {len(embeds)} embeddings")
    print(f"First embedding shape: {len(embeds[0])} dims")
    print(f"Sample: {embeds[0][:5]}...")
