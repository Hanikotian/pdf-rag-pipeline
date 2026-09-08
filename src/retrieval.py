"""
Retrieval layer for the RAG pipeline.

This module converts a user question into an embedding
and retrieves the most semantically similar document chunks.

Important design decision:

We do NOT use a hard-coded similarity threshold by default.

Similarity scores are model- and document-dependent.
A threshold that works well for one PDF may incorrectly
remove useful information from another PDF.

Instead, top-k retrieval is the primary retrieval strategy.

A similarity threshold remains available as an optional
configuration for applications that want additional filtering.
"""

import logging

from src.embeddings import generate_query_embedding
from src.models import DocumentChunk
from src.vector_store import VectorStore


logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 5


def validate_question(
    question: str,
) -> None:
    """
    Validate the user's question.
    """

    if not isinstance(
        question,
        str,
    ):
        raise TypeError(
            "question must be a string."
        )

    if not question.strip():
        raise ValueError(
            "question cannot be empty."
        )


def retrieve_relevant_chunks(
    question: str,
    vector_store: VectorStore,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float | None = None,
) -> list[tuple[DocumentChunk, float]]:
    """
    Retrieve the most relevant chunks for a question.

    Parameters
    ----------
    question:
        User's natural-language question.

    vector_store:
        FAISS vector store containing document chunks.

    top_k:
        Maximum number of chunks to retrieve.

    score_threshold:
        Optional minimum similarity score.

        None:
            No similarity threshold is applied.

        Example:
            score_threshold=0.15

        This is intentionally optional because similarity
        scores are dependent on the embedding model and
        document being searched.

    Returns
    -------
    list[tuple[DocumentChunk, float]]
        Retrieved document chunks with similarity scores.
    """

    # -----------------------------------------------------
    # Validate question.
    # -----------------------------------------------------

    validate_question(
        question
    )

    # -----------------------------------------------------
    # Validate vector store.
    # -----------------------------------------------------

    if not isinstance(
        vector_store,
        VectorStore,
    ):
        raise TypeError(
            "vector_store must be a VectorStore."
        )

    # -----------------------------------------------------
    # Validate top_k.
    # -----------------------------------------------------

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    # -----------------------------------------------------
    # Validate optional threshold.
    # -----------------------------------------------------

    if score_threshold is not None:

        if not 0 <= score_threshold <= 1:
            raise ValueError(
                "score_threshold must be between 0 and 1."
            )

    # -----------------------------------------------------
    # Convert the question into an embedding.
    # -----------------------------------------------------

    query_embedding = (
        generate_query_embedding(
            question
        )
    )

    # -----------------------------------------------------
    # Search the vector store.
    #
    # By default, no threshold is applied.
    #
    # FAISS returns the top-k most similar chunks.
    # -----------------------------------------------------

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    logger.info(
        "Retrieved %d chunks for question.",
        len(results),
    )

    return results