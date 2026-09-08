from pathlib import Path

import pytest

from src.chunking import chunk_document
from src.embeddings import generate_embeddings
from src.ingestion import extract_text_from_pdf
from src.rag import (
    LLMRAGResponse,
    RAGResponse,
    build_context,
    build_rag_prompt,
    map_cited_sources,
    parse_rag_response,
)
from src.vector_store import VectorStore


PDF_PATH = Path(
    "data/raw/rag_sample_vendor_contract.pdf"
)


def build_test_store():
    """
    Build a vector store from the sample PDF.
    """

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    embeddings = generate_embeddings(
        chunks
    )

    store = VectorStore()

    store.add(
        chunks,
        embeddings,
    )

    return store


# =========================================================
# CONTEXT TESTS
# =========================================================

def test_context_contains_chunk_text():

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    context = build_context(
        [
            (
                chunks[0],
                0.9,
            )
        ]
    )

    assert chunks[0].text in context


def test_context_contains_source_number():

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    context = build_context(
        [
            (
                chunks[0],
                0.9,
            )
        ]
    )

    assert "SOURCE 1" in context


def test_context_contains_page_number():

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    context = build_context(
        [
            (
                chunks[0],
                0.9,
            )
        ]
    )

    for page in chunks[0].page_numbers:

        assert str(page) in context


# =========================================================
# PROMPT TESTS
# =========================================================

def test_rag_prompt_contains_question():

    prompt = build_rag_prompt(
        question="What are the payment terms?",
        context="SOURCE 1\nPages: 2\nNet 30 days.",
    )

    assert (
        "What are the payment terms?"
        in prompt
    )


def test_rag_prompt_contains_context():

    prompt = build_rag_prompt(
        question="What are the payment terms?",
        context="SOURCE 1\nPages: 2\nNet 30 days.",
    )

    assert (
        "Net 30 days."
        in prompt
    )


def test_rag_prompt_requires_source_numbers():

    prompt = build_rag_prompt(
        question="What are the payment terms?",
        context="SOURCE 1\nPages: 2\nNet 30 days.",
    )

    assert (
        "source_numbers"
        in prompt
    )


# =========================================================
# RESPONSE TESTS
# =========================================================

def test_rag_response_page_numbers():

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    response = RAGResponse(
        answer="Test answer.",
        sources=[
            (
                chunks[0],
                0.9,
            ),
            (
                chunks[1],
                0.8,
            ),
        ],
        cited_sources=[
            (
                chunks[0],
                0.9,
            ),
        ],
    )

    assert response.page_numbers == sorted(
        chunks[0].page_numbers
    )


def test_rag_response_does_not_use_uncited_sources():

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    response = RAGResponse(
        answer="Test answer.",
        sources=[
            (
                chunks[0],
                0.9,
            ),
            (
                chunks[1],
                0.8,
            ),
        ],
        cited_sources=[
            (
                chunks[1],
                0.8,
            ),
        ],
    )

    assert response.page_numbers == sorted(
        chunks[1].page_numbers
    )


# =========================================================
# SOURCE MAPPING TESTS
# =========================================================

def test_source_numbers_map_to_correct_chunks():

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    results = [
        (
            chunks[0],
            0.9,
        ),
        (
            chunks[1],
            0.8,
        ),
    ]

    cited_sources = map_cited_sources(
        results=results,
        source_numbers=[2],
    )

    assert len(
        cited_sources
    ) == 1

    assert (
        cited_sources[0][0].chunk_id
        == chunks[1].chunk_id
    )


# =========================================================
# PARSING TESTS
# =========================================================

def test_valid_rag_response_is_parsed():

    class MockResponse:
        parsed = LLMRAGResponse(
            answer="The payment terms are Net 30 days.",
            source_numbers=[1],
        )

    result = parse_rag_response(
        response=MockResponse(),
        number_of_sources=3,
    )

    assert (
        result.answer
        == "The payment terms are Net 30 days."
    )

    assert (
        result.source_numbers
        == [1]
    )


def test_invalid_source_number_is_rejected():

    class MockResponse:
        parsed = LLMRAGResponse(
            answer="Test answer.",
            source_numbers=[5],
        )

    with pytest.raises(
        RuntimeError
    ):

        parse_rag_response(
            response=MockResponse(),
            number_of_sources=3,
        )


def test_duplicate_source_numbers_are_removed():

    class MockResponse:
        parsed = LLMRAGResponse(
            answer="Test answer.",
            source_numbers=[1, 1, 2, 2],
        )

    result = parse_rag_response(
        response=MockResponse(),
        number_of_sources=3,
    )

    assert (
        result.source_numbers
        == [1, 2]
    )