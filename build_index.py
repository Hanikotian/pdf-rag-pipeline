"""
Build a persistent RAG index from a PDF.

This is a small command-line entry point used during
development.

Later, the Streamlit application will call the same
indexing functionality directly when a user uploads
a PDF.
"""

from pathlib import Path

from src.indexing import build_index


PDF_PATH = Path(
    "data/raw/IEEE_Paper1.pdf"
)

INDEX_DIRECTORY = Path(
    "data/indexes/IEEE_Paper1"
)


if __name__ == "__main__":

    vector_store = build_index(
        pdf_path=PDF_PATH,
        index_directory=INDEX_DIRECTORY,
    )

    print()
    print("=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)

    print(
        f"PDF: {PDF_PATH}"
    )

    print(
        f"Chunks indexed: {len(vector_store.chunks)}"
    )

    print(
        f"Index location: {INDEX_DIRECTORY}"
    )