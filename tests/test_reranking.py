"""
Tests for the RAG reranking layer.

These tests avoid downloading or running the real
Cross-Encoder model.

Instead, pytest mocks the model so the tests stay:

- fast
- deterministic
- offline
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models import DocumentChunk
from src.reranking import (
    rerank_chunks,
    validate_reranking_input,
)


def create_chunk(
    chunk_id: str,
    text: str,
    page_number: int,
) -> DocumentChunk:
    """
    Helper function used by the tests.
    """

    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        page_numbers=[page_number],
        character_count=len(text),
        source_filename="test.pdf",
    )


def test_empty_question_rejected():
    """
    Empty questions should not be accepted.
    """

    with pytest.raises(ValueError):
        validate_reranking_input(
            question="",
            retrieved_chunks=[],
            top_k=5,
        )


def test_whitespace_question_rejected():
    """
    Questions containing only whitespace
    should also be rejected.
    """

    with pytest.raises(ValueError):
        validate_reranking_input(
            question="   ",
            retrieved_chunks=[],
            top_k=5,
        )


def test_non_string_question_rejected():
    """
    The question must be a string.
    """

    with pytest.raises(TypeError):
        validate_reranking_input(
            question=123,
            retrieved_chunks=[],
            top_k=5,
        )


def test_invalid_top_k_rejected():
    """
    top_k must always be greater than zero.
    """

    with pytest.raises(ValueError):
        validate_reranking_input(
            question="What is this document about?",
            retrieved_chunks=[],
            top_k=0,
        )


def test_empty_retrieval_returns_empty_list():
    """
    If FAISS retrieved nothing,
    reranking should simply return nothing.
    """

    result = rerank_chunks(
        question="What is this document about?",
        retrieved_chunks=[],
        top_k=5,
    )

    assert result == []


@patch(
    "src.reranking.get_reranker_model"
)
def test_chunks_are_reranked(
    mock_get_model,
):
    """
    Verify that chunks are reordered using
    Cross-Encoder relevance scores.
    """

    chunk_1 = create_chunk(
        chunk_id="chunk-0001",
        text="This paragraph discusses cats.",
        page_number=1,
    )

    chunk_2 = create_chunk(
        chunk_id="chunk-0002",
        text="The proposed neural network achieved "
        "87 percent accuracy.",
        page_number=2,
    )

    chunk_3 = create_chunk(
        chunk_id="chunk-0003",
        text="This paragraph discusses weather.",
        page_number=3,
    )

    retrieved = [
        (
            chunk_1,
            0.30,
        ),
        (
            chunk_2,
            0.20,
        ),
        (
            chunk_3,
            0.25,
        ),
    ]

    # Pretend the Cross-Encoder gives these scores.
    #
    # Chunk 2 should become first even though
    # its FAISS similarity was lower.
    mock_model = MagicMock()

    mock_model.predict.return_value = [
        0.10,
        0.95,
        0.05,
    ]

    mock_get_model.return_value = (
        mock_model
    )

    results = rerank_chunks(
        question=(
            "What accuracy did the "
            "proposed model achieve?"
        ),
        retrieved_chunks=retrieved,
        top_k=3,
    )

    assert len(results) == 3

    # Chunk 2 should now be ranked first.
    assert (
        results[0][0].chunk_id
        == "chunk-0002"
    )

    assert results[0][1] == 0.20
    assert results[0][2] == 0.95


@patch(
    "src.reranking.get_reranker_model"
)
def test_reranking_respects_top_k(
    mock_get_model,
):
    """
    Verify that reranking returns only
    the requested number of chunks.
    """

    chunk_1 = create_chunk(
        "chunk-0001",
        "Text one",
        1,
    )

    chunk_2 = create_chunk(
        "chunk-0002",
        "Text two",
        2,
    )

    chunk_3 = create_chunk(
        "chunk-0003",
        "Text three",
        3,
    )

    retrieved = [
        (chunk_1, 0.3),
        (chunk_2, 0.2),
        (chunk_3, 0.1),
    ]

    mock_model = MagicMock()

    mock_model.predict.return_value = [
        0.4,
        0.9,
        0.2,
    ]

    mock_get_model.return_value = (
        mock_model
    )

    results = rerank_chunks(
        question="Test question",
        retrieved_chunks=retrieved,
        top_k=2,
    )

    assert len(results) == 2

    assert (
        results[0][0].chunk_id
        == "chunk-0002"
    )

    assert (
        results[1][0].chunk_id
        == "chunk-0001"
    )