"""
Embedding generation for the RAG pipeline.

This module converts document chunks and user questions
into numerical vectors.

Document chunks:
    Used when building the vector database.

User questions:
    Used when searching the vector database.

The same embedding model must be used for both.
"""

import logging

from sentence_transformers import SentenceTransformer

from src.models import DocumentChunk


logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# The model is loaded only once and then reused.
# This prevents loading the model every time we
# generate an embedding.
_model = None


# =========================================================
# MODEL
# =========================================================

def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    """
    Load and return the embedding model.

    The model is cached after the first load.
    """

    global _model

    if _model is None:

        logger.info(
            "Loading embedding model: %s",
            model_name,
        )

        _model = SentenceTransformer(
            model_name
        )

        logger.info(
            "Embedding model loaded successfully."
        )

    return _model


# =========================================================
# DOCUMENT CHUNK VALIDATION
# =========================================================

def validate_chunks(
    chunks: list[DocumentChunk],
) -> None:
    """
    Validate document chunks before generating embeddings.
    """

    if not isinstance(chunks, list):
        raise TypeError(
            "chunks must be a list."
        )

    for chunk in chunks:

        if not isinstance(
            chunk,
            DocumentChunk,
        ):
            raise TypeError(
                "Every item in chunks must be a DocumentChunk."
            )

        if not chunk.text.strip():
            raise ValueError(
                f"Chunk {chunk.chunk_id} contains empty text."
            )


# =========================================================
# DOCUMENT EMBEDDINGS
# =========================================================

def generate_embeddings(
    chunks: list[DocumentChunk],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[list[float]]:
    """
    Generate embeddings for document chunks.

    Each chunk receives one numerical vector.

    Example:

        chunk 1 → [0.12, -0.42, ...]
        chunk 2 → [0.31,  0.18, ...]

    Returns:
        A list of embedding vectors.
    """

    validate_chunks(
        chunks
    )

    if not chunks:
        return []

    model = get_embedding_model(
        model_name
    )

    texts = [
        chunk.text
        for chunk in chunks
    ]

    logger.info(
        "Generating embeddings for %d chunks.",
        len(texts),
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    result = [
        embedding.tolist()
        for embedding in embeddings
    ]

    logger.info(
        "Generated %d embeddings.",
        len(result),
    )

    return result


# =========================================================
# QUERY EMBEDDING
# =========================================================

def generate_query_embedding(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float]:
    """
    Generate an embedding for a user's question.

    The query must use the same embedding model as
    the document chunks.

    Example:

        "What are the payment terms?"

    becomes:

        [0.023, -0.182, 0.441, ...]
    """

    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "query cannot be empty."
        )

    model = get_embedding_model(
        model_name
    )

    logger.info(
        "Generating embedding for user query."
    )

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    return embedding.tolist()