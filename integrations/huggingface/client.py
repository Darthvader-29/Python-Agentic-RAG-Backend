"""
Hugging Face Inference API client for sentence-transformers/all-MiniLM-L6-v2 embeddings.
Uses huggingface_hub.InferenceClient with feature-extraction pipeline.
Free tier, 384-dimensional embeddings.
"""

import numpy as np
import structlog
from huggingface_hub import InferenceClient

from config import settings

client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=settings.HUGGINGFACE_TOKEN,
)

logger = structlog.get_logger(__name__)


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
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
        batch = texts[i : i + batch_size]

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


def embed_single(text: str) -> list[float]:
    """Embed a single text (convenience wrapper)."""
    return embed_batch([text], batch_size=1)[0]
