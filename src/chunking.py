"""
Generic document chunking for RAG.

This module converts an ingested Document into smaller
overlapping chunks suitable for:

- embeddings
- vector search
- retrieval
- RAG

Important design decision:

Chunks are page-aware.

A chunk will NEVER contain text from multiple PDF pages.

This makes source attribution much more reliable because
every chunk knows exactly which page it came from.
"""


# =========================================================
# IMPORTS
# =========================================================

import logging

from src.models import (
    Document,
    DocumentChunk,
)


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# DEFAULT SETTINGS
# =========================================================

DEFAULT_CHUNK_SIZE = 1200

DEFAULT_CHUNK_OVERLAP = 200


# =========================================================
# VALIDATION
# =========================================================

def validate_chunk_settings(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """
    Validate chunking configuration.
    """

    if chunk_size <= 0:

        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:

        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:

        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )


# =========================================================
# PAGE SEGMENTS
# =========================================================

def build_document_segments(
    document: Document,
) -> list[tuple[int, str]]:
    """
    Extract non-empty page text from a document.

    Returns:

        [
            (page_number, page_text),
            ...
        ]

    Each page remains an independent segment.

    This is important because the chunker must not combine
    text from different PDF pages.
    """

    segments = []

    for page in document.pages:

        text = page.text.strip()

        if not text:
            continue

        segments.append(
            (
                page.page_number,
                text,
            )
        )

    return segments


# =========================================================
# PAGE CHUNKING
# =========================================================

def chunk_page(
    page_number: int,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    source_filename: str | None,
    starting_chunk_index: int,
) -> tuple[
    list[DocumentChunk],
    int,
]:
    """
    Split one page into overlapping chunks.

    IMPORTANT:

    This function operates on ONE page only.

    Therefore no resulting chunk can contain text
    from another page.
    """

    chunks = []

    start = 0

    chunk_index = (
        starting_chunk_index
    )

    text_length = len(text)

    while start < text_length:

        # -------------------------------------------------
        # Determine the end of this chunk.
        # -------------------------------------------------

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk_text = text[
            start:end
        ].strip()

        if chunk_text:

            chunk = DocumentChunk(
                chunk_id=(
                    f"chunk-{chunk_index:04d}"
                ),
                text=chunk_text,
                page_numbers=[
                    page_number
                ],
                character_count=len(
                    chunk_text
                ),
                source_filename=source_filename,
            )

            chunks.append(
                chunk
            )

            chunk_index += 1

        # -------------------------------------------------
        # Stop after the final chunk.
        # -------------------------------------------------

        if end >= text_length:

            break

        # -------------------------------------------------
        # Move backwards by the overlap amount.
        #
        # Example:
        #
        # chunk size = 1200
        # overlap    = 200
        #
        # next start = 1200 - 200
        #            = 1000
        # -------------------------------------------------

        start = (
            end - chunk_overlap
        )

    return (
        chunks,
        chunk_index,
    )


# =========================================================
# MAIN CHUNKING FUNCTION
# =========================================================

def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """
    Convert a Document into page-aware chunks.

    Each chunk belongs to exactly one PDF page.

    Example:

        Page 1
          ↓
        chunk-0000
        chunk-0001

        Page 2
          ↓
        chunk-0002
        chunk-0003

    Returns:
        List of DocumentChunk objects.
    """

    # -----------------------------------------------------
    # Validate settings.
    # -----------------------------------------------------

    validate_chunk_settings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # -----------------------------------------------------
    # Get individual page segments.
    # -----------------------------------------------------

    segments = build_document_segments(
        document
    )

    if not segments:

        return []

    # -----------------------------------------------------
    # Get source filename from document metadata.
    # -----------------------------------------------------

    filename = None

    if document.metadata:

        filename = (
            document.metadata.filename
        )

    # -----------------------------------------------------
    # Chunk each page independently.
    # -----------------------------------------------------

    all_chunks = []

    chunk_index = 0

    for page_number, text in segments:

        page_chunks, chunk_index = (
            chunk_page(
                page_number=page_number,
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_filename=filename,
                starting_chunk_index=chunk_index,
            )
        )

        all_chunks.extend(
            page_chunks
        )

    # -----------------------------------------------------
    # Logging.
    # -----------------------------------------------------

    logger.info(
        "Chunking completed. Generated %d page-aware chunks.",
        len(all_chunks),
    )

    return all_chunks