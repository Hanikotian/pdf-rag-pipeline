from pathlib import Path

import pytest

from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_document
from src.embeddings import generate_embeddings
from src.vector_store import VectorStore


PDF_PATH = Path(
    "data/raw/rag_sample_vendor_contract.pdf"
)


def build_test_store():
    """
    Build a vector store from the sample contract.
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

    return store, chunks, embeddings


def test_chunks_can_be_added():

    store, chunks, embeddings = (
        build_test_store()
    )

    assert len(store) == len(
        chunks
    )

    assert store.dimension == len(
        embeddings[0]
    )


def test_search_returns_results():

    store, chunks, embeddings = (
        build_test_store()
    )

    results = store.search(
        embeddings[0],
        top_k=3,
    )

    assert len(results) > 0

    returned_chunk, score = results[0]

    assert returned_chunk.chunk_id == (
        chunks[0].chunk_id
    )

    assert isinstance(
        score,
        float,
    )


def test_top_k_is_respected():

    store, chunks, embeddings = (
        build_test_store()
    )

    results = store.search(
        embeddings[0],
        top_k=2,
    )

    assert len(results) <= 2


def test_empty_store_rejected():

    store = VectorStore()

    with pytest.raises(
        RuntimeError
    ):
        store.search(
            [0.1, 0.2]
        )


def test_mismatched_chunks_and_embeddings_rejected():

    store = VectorStore()

    with pytest.raises(
        ValueError
    ):
        store.add(
            chunks=[
                chunks
                for chunks in []
            ],
            embeddings=[
                [0.1, 0.2]
            ],
        )