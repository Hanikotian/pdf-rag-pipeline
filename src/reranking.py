"""
Reranking layer for the RAG pipeline.

FAISS retrieval is very fast, but embedding similarity alone
does not always produce the best ranking.

This module adds a second-stage reranker.

Pipeline:

    Question
        ↓
    FAISS retrieval
        ↓
    Candidate chunks
        ↓
    Cross-Encoder reranking
        ↓
    Best chunks
        ↓
    Gemini

The Cross-Encoder directly evaluates:

    question + chunk

together, which usually gives a better relevance ranking
than embedding similarity alone.
"""

import logging

from sentence_transformers import CrossEncoder

from src.models import DocumentChunk


logger = logging.getLogger(__name__)


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_RERANK_TOP_K = 5


# The model is cached after the first load.
#
# This prevents the reranker from being loaded from disk
# every time a question is asked.
_reranker_model = None


def get_reranker_model(
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> CrossEncoder:
    """
    Load and cache the Cross-Encoder reranking model.

    Parameters
    ----------
    model_name:
        Hugging Face model name.

    Returns
    -------
    CrossEncoder
        Loaded reranking model.
    """

    global _reranker_model

    if _reranker_model is None:
        logger.info(
            "Loading reranker model: %s",
            model_name,
        )

        _reranker_model = CrossEncoder(
            model_name
        )

        logger.info(
            "Reranker model loaded successfully."
        )

    return _reranker_model


def validate_reranking_input(
    question: str,
    retrieved_chunks: list[
        tuple[DocumentChunk, float]
    ],
    top_k: int,
) -> None:
    """
    Validate reranking inputs.
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

    if not isinstance(
        retrieved_chunks,
        list,
    ):
        raise TypeError(
            "retrieved_chunks must be a list."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    for item in retrieved_chunks:

        if not isinstance(
            item,
            tuple,
        ):
            raise TypeError(
                "Each retrieved item must be a tuple."
            )

        if len(item) != 2:
            raise ValueError(
                "Each retrieved item must contain "
                "(DocumentChunk, similarity_score)."
            )

        chunk, score = item

        if not isinstance(
            chunk,
            DocumentChunk,
        ):
            raise TypeError(
                "Each retrieved chunk must be "
                "a DocumentChunk."
            )

        if not isinstance(
            score,
            (int, float),
        ):
            raise TypeError(
                "Similarity score must be numeric."
            )


def rerank_chunks(
    question: str,
    retrieved_chunks: list[
        tuple[DocumentChunk, float]
    ],
    top_k: int = DEFAULT_RERANK_TOP_K,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[
    tuple[DocumentChunk, float, float]
]:
    """
    Rerank retrieved chunks using a Cross-Encoder.

    Parameters
    ----------
    question:
        User's natural-language question.

    retrieved_chunks:
        Candidate chunks returned by FAISS.

        Format:

            [
                (chunk, similarity_score),
                ...
            ]

    top_k:
        Number of reranked chunks to keep.

    model_name:
        Cross-Encoder model used for reranking.

    Returns
    -------
    list[tuple[DocumentChunk, float, float]]

        Each result contains:

            (
                chunk,
                faiss_similarity_score,
                reranker_score,
            )

    Results are sorted by reranker score
    from highest to lowest.
    """

    validate_reranking_input(
        question=question,
        retrieved_chunks=retrieved_chunks,
        top_k=top_k,
    )

    if not retrieved_chunks:
        return []

    model = get_reranker_model(
        model_name=model_name
    )

    # A Cross-Encoder receives pairs of text:
    #
    # [
    #     [question, chunk_1],
    #     [question, chunk_2],
    #     ...
    # ]
    #
    # It then gives each pair a relevance score.
    pairs = [
        [
            question,
            chunk.text,
        ]
        for chunk, _ in retrieved_chunks
    ]

    logger.info(
        "Reranking %d retrieved chunks.",
        len(pairs),
    )

    reranker_scores = model.predict(
        pairs
    )

    reranked_results = []

    for (
        chunk,
        similarity_score,
    ), reranker_score in zip(
        retrieved_chunks,
        reranker_scores,
    ):

        reranked_results.append(
            (
                chunk,
                float(similarity_score),
                float(reranker_score),
            )
        )

    # Highest reranker score first.
    reranked_results.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    # We cannot return more chunks than exist.
    final_results = reranked_results[
        :top_k
    ]

    logger.info(
        "Reranking complete. "
        "Returning %d chunks.",
        len(final_results),
    )

    return final_results