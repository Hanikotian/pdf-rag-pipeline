"""
Tests for the retrieval layer.
"""

from unittest.mock import patch

import pytest

from src.models import DocumentChunk
from src.retrieval import retrieve_relevant_chunks
from src.vector_store import VectorStore


def create_test_chunks():
    """
    Create simple chunks for testing retrieval.
    """

    return [
        DocumentChunk(
            chunk_id="chunk-0000",
            text="Payment terms are Net 30 days.",
            page_numbers=[1],
            character_count=31,
            source_filename="test.pdf",
        ),
        DocumentChunk(
            chunk_id="chunk-0001",
            text="The agreement is governed by Indian law.",
            page_numbers=[2],
            character_count=42,
            source_filename="test.pdf",
        ),
    ]


def test_empty_question_is_rejected():

    vector_store = VectorStore()

    with pytest.raises(ValueError):

        retrieve_relevant_chunks(
            question="",
            vector_store=vector_store,
        )


def test_non_string_question_is_rejected():

    vector_store = VectorStore()

    with pytest.raises(TypeError):

        retrieve_relevant_chunks(
            question=123,
            vector_store=vector_store,
        )


def test_invalid_top_k_is_rejected():

    vector_store = VectorStore()

    with pytest.raises(ValueError):

        retrieve_relevant_chunks(
            question="What are the payment terms?",
            vector_store=vector_store,
            top_k=0,
        )


def test_invalid_threshold_is_rejected():

    vector_store = VectorStore()

    with pytest.raises(ValueError):

        retrieve_relevant_chunks(
            question="What are the payment terms?",
            vector_store=vector_store,
            score_threshold=1.5,
        )


@patch(
    "src.retrieval.generate_query_embedding"
)
def test_retrieval_returns_top_k_results(
    mock_embedding,
):

    chunks = create_test_chunks()

    vector_store = VectorStore()

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    vector_store.add(
        chunks,
        embeddings,
    )

    mock_embedding.return_value = [
        1.0,
        0.0,
    ]

    results = retrieve_relevant_chunks(
        question="What are the payment terms?",
        vector_store=vector_store,
        top_k=1,
    )

    assert len(results) == 1

    chunk, score = results[0]

    assert chunk.chunk_id == "chunk-0000"

    assert score > 0


@patch(
    "src.retrieval.generate_query_embedding"
)
def test_threshold_can_filter_results(
    mock_embedding,
):

    chunks = create_test_chunks()

    vector_store = VectorStore()

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    vector_store.add(
        chunks,
        embeddings,
    )

    mock_embedding.return_value = [
        1.0,
        0.0,
    ]

    results = retrieve_relevant_chunks(
        question="What are the payment terms?",
        vector_store=vector_store,
        top_k=2,
        score_threshold=0.9,
    )

    assert len(results) == 1

    assert (
        results[0][0].chunk_id
        == "chunk-0000"
    )


@patch(
    "src.retrieval.generate_query_embedding"
)
def test_no_threshold_returns_top_k(
    mock_embedding,
):

    chunks = create_test_chunks()

    vector_store = VectorStore()

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    vector_store.add(
        chunks,
        embeddings,
    )

    mock_embedding.return_value = [
        1.0,
        0.0,
    ]

    results = retrieve_relevant_chunks(
        question="What are the payment terms?",
        vector_store=vector_store,
        top_k=2,
    )

    assert len(results) == 2