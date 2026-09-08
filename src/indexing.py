"""
Document indexing pipeline for the RAG application.

This module is responsible for processing a PDF once and
creating a persistent vector index.

Pipeline:

    PDF
     ↓
    Ingestion
     ↓
    Chunking
     ↓
    Embeddings
     ↓
    FAISS Vector Store
     ↓
    Save to disk

The indexing pipeline is intentionally generic.

It does NOT know whether the PDF is:
- a contract
- a research paper
- a technical document
- a report
- a manual
- or any other type of document.
"""

import logging
from pathlib import Path

from src.chunking import chunk_document
from src.embeddings import generate_embeddings
from src.ingestion import extract_text_from_pdf
from src.vector_store import VectorStore


logger = logging.getLogger(__name__)


DEFAULT_INDEX_DIRECTORY = Path(
    "data/indexes"
)


def validate_pdf_path(
    pdf_path: Path,
) -> None:
    """
    Validate the input PDF path.
    """

    if not isinstance(
        pdf_path,
        Path,
    ):
        raise TypeError(
            "pdf_path must be a pathlib.Path."
        )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"PDF path is not a file: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {pdf_path.suffix}"
        )


def build_index(
    pdf_path: Path,
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
) -> VectorStore:
    """
    Process a PDF and create a persistent vector index.

    Parameters
    ----------
    pdf_path:
        Path to the PDF document.

    index_directory:
        Directory where the FAISS index and chunk metadata
        will be saved.

    Returns
    -------
    VectorStore
        The populated vector store.

    Processing steps:

        1. Extract PDF text.
        2. Split text into page-aware chunks.
        3. Generate embeddings.
        4. Build FAISS vector store.
        5. Save vector store to disk.
    """

    validate_pdf_path(
        pdf_path
    )

    logger.info(
        "Starting document indexing: %s",
        pdf_path,
    )

    # =====================================================
    # STEP 1 — INGESTION
    # =====================================================

    logger.info(
        "Step 1/4: Extracting PDF text."
    )

    document = extract_text_from_pdf(
        pdf_path
    )

    if not document.pages:
        raise ValueError(
            "PDF contains no extractable pages."
        )

    logger.info(
        "Extracted %d pages.",
        len(document.pages),
    )

    # =====================================================
    # STEP 2 — CHUNKING
    # =====================================================

    logger.info(
        "Step 2/4: Creating document chunks."
    )

    chunks = chunk_document(
        document
    )

    if not chunks:
        raise ValueError(
            "No text chunks were generated from the PDF."
        )

    logger.info(
        "Created %d chunks.",
        len(chunks),
    )

    # =====================================================
    # STEP 3 — EMBEDDINGS
    # =====================================================

    logger.info(
        "Step 3/4: Generating embeddings."
    )

    embeddings = generate_embeddings(
        chunks
    )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    logger.info(
        "Generated %d embeddings.",
        len(embeddings),
    )

    # =====================================================
    # STEP 4 — VECTOR STORE
    # =====================================================

    logger.info(
        "Step 4/4: Building FAISS vector store."
    )

    vector_store = VectorStore()

    vector_store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    # -----------------------------------------------------
    # Save the vector store.
    # -----------------------------------------------------

    vector_store.save(
        index_directory
    )

    logger.info(
        "Index successfully saved to: %s",
        index_directory,
    )

    return vector_store


def load_index(
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
) -> VectorStore:
    """
    Load an existing vector index from disk.

    This is used after the document has already been
    processed.

    The PDF does NOT need to be embedded again.
    """

    if not isinstance(
        index_directory,
        Path,
    ):
        raise TypeError(
            "index_directory must be a pathlib.Path."
        )

    if not index_directory.exists():
        raise FileNotFoundError(
            f"Index directory not found: {index_directory}"
        )

    logger.info(
        "Loading vector index from: %s",
        index_directory,
    )

    vector_store = VectorStore.load(
        index_directory
    )

    logger.info(
        "Vector index loaded successfully."
    )

    return vector_store