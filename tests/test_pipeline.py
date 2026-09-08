"""
Tests for the high-level RAG pipeline.

These tests verify orchestration and validation.

External Gemini calls are NOT required here.

The lower-level RAG functionality is tested separately
in tests/test_rag.py.
"""

from pathlib import Path

import pytest

from src.pipeline import RAGPipeline
from src.vector_store import VectorStore


def test_pipeline_starts_without_index():
    """
    A newly created pipeline should not have an index.
    """

    pipeline = RAGPipeline()

    assert pipeline.vector_store is None
    assert pipeline.is_ready is False
    assert pipeline.number_of_chunks == 0


def test_pipeline_rejects_invalid_index_directory():
    """
    index_directory must be a pathlib.Path.
    """

    with pytest.raises(TypeError):

        RAGPipeline(
            index_directory="data/indexes/test"
        )


def test_pipeline_ask_without_index_is_rejected():
    """
    Asking a question before loading a document should fail.
    """

    pipeline = RAGPipeline()

    with pytest.raises(RuntimeError):

        pipeline.ask(
            "What is this document about?"
        )


def test_pipeline_rejects_empty_question():
    """
    Empty questions should be rejected.
    """

    pipeline = RAGPipeline()

    # We need an index because the pipeline checks for
    # an index before processing the question.
    pipeline.vector_store = VectorStore()

    with pytest.raises(ValueError):

        pipeline.ask("")


def test_pipeline_rejects_whitespace_question():
    """
    Whitespace-only questions should be rejected.
    """

    pipeline = RAGPipeline()

    pipeline.vector_store = VectorStore()

    with pytest.raises(ValueError):

        pipeline.ask("   ")


def test_pipeline_rejects_non_string_question():
    """
    Questions must be strings.
    """

    pipeline = RAGPipeline()

    pipeline.vector_store = VectorStore()

    with pytest.raises(TypeError):

        pipeline.ask(123)


def test_pipeline_rejects_invalid_retrieval_top_k():
    """
    retrieval_top_k must be greater than zero.
    """

    pipeline = RAGPipeline()

    pipeline.vector_store = VectorStore()

    with pytest.raises(ValueError):

        pipeline.ask(
            "What is this document about?",
            retrieval_top_k=0,
        )


def test_pipeline_rejects_invalid_rerank_top_k():
    """
    rerank_top_k must be greater than zero.
    """

    pipeline = RAGPipeline()

    pipeline.vector_store = VectorStore()

    with pytest.raises(ValueError):

        pipeline.ask(
            "What is this document about?",
            rerank_top_k=0,
        )


def test_pipeline_requires_index_directory():
    """
    index_document() must know where the persistent
    index should be stored.
    """

    pipeline = RAGPipeline()

    with pytest.raises(ValueError):

        pipeline.index_document(
            Path("data/raw/IEEE_Paper1.pdf")
        )


def test_pipeline_accepts_path_object_for_index_directory():
    """
    A Path object should be accepted during initialization.
    """

    pipeline = RAGPipeline(
        index_directory=Path(
            "data/indexes/test"
        )
    )

    assert pipeline.index_directory == Path(
        "data/indexes/test"
    )