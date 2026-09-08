from pathlib import Path

from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_document
from src.embeddings import generate_embeddings


PDF_PATH = Path(
    "data/raw/rag_sample_vendor_contract.pdf"
)


def test_embeddings_generated():

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    embeddings = generate_embeddings(
        chunks
    )

    assert len(embeddings) == len(
        chunks
    )

    assert len(embeddings) > 0


def test_embedding_dimensions_are_consistent():

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    embeddings = generate_embeddings(
        chunks
    )

    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    assert len(dimensions) == 1


def test_embeddings_are_numeric():

    document = extract_text_from_pdf(
        PDF_PATH
    )

    chunks = chunk_document(
        document
    )

    embeddings = generate_embeddings(
        chunks
    )

    for embedding in embeddings:

        assert all(
            isinstance(value, float)
            for value in embedding
        )


def test_empty_chunks_return_empty_embeddings():

    embeddings = generate_embeddings(
        []
    )

    assert embeddings == []