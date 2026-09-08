"""
Generic PDF ingestion.

This module is responsible only for reading PDFs and turning
them into our generic Document model.

It does NOT:
- understand contracts
- extract vendor/customer information
- call an LLM
- perform semantic interpretation

Those responsibilities belong to later stages.

Pipeline:

    PDF
     |
     v
    PDF ingestion
     |
     +--> Metadata
     |
     +--> Page text
     |
     +--> Extraction quality
     |
     v
    Document object
"""

from pathlib import Path
import logging

import pymupdf

from src.models import (
    Document,
    DocumentMetadata,
    Page,
)


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

# Minimum number of characters we consider meaningful text.
#
# This is intentionally small because even a short page
# can contain useful information.
MIN_TEXT_CHARACTERS = 10


# =========================================================
# PAGE TEXT EXTRACTION
# =========================================================

def extract_page_text(
    pdf_page,
) -> str:
    """
    Extract text from one PDF page.

    Parameters
    ----------
    pdf_page:
        A PyMuPDF page object.

    Returns
    -------
    str
        Extracted page text.

    The extraction is deliberately generic. It does not
    attempt to interpret the content.
    """

    # Extract normal text from the PDF page.
    text = pdf_page.get_text(
        "text"
    )

    # Normalize excessive whitespace while preserving
    # meaningful line breaks.
    lines = []

    for line in text.splitlines():

        # Remove unnecessary whitespace around each line.
        cleaned_line = line.strip()

        if cleaned_line:
            lines.append(
                cleaned_line
            )

    # Reconstruct the page text.
    return "\n".join(lines)


# =========================================================
# DOCUMENT METADATA
# =========================================================

def build_document_metadata(
    pdf_document,
    pdf_path: Path,
    pages: list[Page],
) -> DocumentMetadata:
    """
    Build generic metadata for the PDF.

    This information is useful regardless of whether the
    document is a contract, research paper, invoice, etc.
    """

    # -----------------------------------------------------
    # Basic PDF information
    # -----------------------------------------------------

    page_count = len(pdf_document)

    # Get metadata supplied by the PDF itself.
    pdf_metadata = pdf_document.metadata or {}

    # -----------------------------------------------------
    # Extraction statistics
    # -----------------------------------------------------

    total_character_count = sum(
        page.character_count
        for page in pages
    )

    text_pages = sum(
        1
        for page in pages
        if page.has_text
    )

    empty_pages = page_count - text_pages

    # Avoid division by zero for an empty PDF.
    if page_count > 0:

        text_extraction_ratio = (
            text_pages / page_count
        )

    else:

        text_extraction_ratio = 0.0

    # A document has usable text if at least one page
    # contains meaningful extracted text.
    has_usable_text = (
        text_pages > 0
    )

    # -----------------------------------------------------
    # Create metadata model
    # -----------------------------------------------------

    return DocumentMetadata(

        filename=pdf_path.name,

        page_count=page_count,

        title=pdf_metadata.get(
            "title"
        ) or None,

        author=pdf_metadata.get(
            "author"
        ) or None,

        subject=pdf_metadata.get(
            "subject"
        ) or None,

        creator=pdf_metadata.get(
            "creator"
        ) or None,

        producer=pdf_metadata.get(
            "producer"
        ) or None,

        total_character_count=(
            total_character_count
        ),

        text_pages=text_pages,

        empty_pages=empty_pages,

        text_extraction_ratio=(
            text_extraction_ratio
        ),

        has_usable_text=(
            has_usable_text
        ),
    )


# =========================================================
# MAIN INGESTION FUNCTION
# =========================================================

def extract_text_from_pdf(
    pdf_path: str | Path,
) -> Document:
    """
    Extract a PDF into our generic Document model.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    Document
        Generic document containing metadata and pages.

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist.

    ValueError
        If the supplied path is not a PDF or the PDF
        contains no pages.
    """

    # -----------------------------------------------------
    # STEP 1: Normalize the path
    # -----------------------------------------------------

    pdf_path = Path(
        pdf_path
    )

    logger.info(
        "Starting PDF ingestion: %s",
        pdf_path,
    )

    # -----------------------------------------------------
    # STEP 2: Validate the file
    # -----------------------------------------------------

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():

        raise ValueError(
            f"Path is not a file: {pdf_path}"
        )

    # Make sure the file has a PDF extension.
    if pdf_path.suffix.lower() != ".pdf":

        raise ValueError(
            f"Expected a PDF file, got: "
            f"{pdf_path.suffix}"
        )

    # -----------------------------------------------------
    # STEP 3: Open the PDF
    # -----------------------------------------------------

    try:

        pdf_document = pymupdf.open(
            pdf_path
        )

    except Exception as exc:

        logger.exception(
            "Failed to open PDF: %s",
            pdf_path,
        )

        raise ValueError(
            f"Could not open PDF: {pdf_path}"
        ) from exc

    # -----------------------------------------------------
    # STEP 4: Check that the PDF contains pages
    # -----------------------------------------------------

    if len(pdf_document) == 0:

        pdf_document.close()

        raise ValueError(
            f"PDF contains no pages: {pdf_path}"
        )

    # -----------------------------------------------------
    # STEP 5: Extract every page
    # -----------------------------------------------------

    pages: list[Page] = []

    try:

        for index, pdf_page in enumerate(
            pdf_document
        ):

            # PDF pages are internally zero-indexed.
            #
            # Our application uses human-friendly
            # one-based page numbers.
            page_number = index + 1

            # Extract text from this page.
            text = extract_page_text(
                pdf_page
            )

            # Count extracted characters.
            character_count = len(
                text
            )

            # Determine whether the page contains
            # meaningful machine-readable text.
            has_text = (
                character_count
                >= MIN_TEXT_CHARACTERS
            )

            # Create our generic Page model.
            page = Page(

                page_number=page_number,

                text=text,

                character_count=(
                    character_count
                ),

                has_text=has_text,
            )

            pages.append(
                page
            )

            logger.debug(
                "Page %d extracted: %d characters",
                page_number,
                character_count,
            )

    finally:

        # Always close the PDF, even if page extraction
        # encounters an unexpected error.
        pdf_document.close()

    # -----------------------------------------------------
    # STEP 6: Reopen PDF for metadata
    # -----------------------------------------------------
    #
    # We closed the PDF above to ensure proper resource
    # management.
    #
    # Open it again only to safely retrieve metadata.
    # -----------------------------------------------------

    try:

        metadata_document = pymupdf.open(
            pdf_path
        )

        metadata = build_document_metadata(
            metadata_document,
            pdf_path,
            pages,
        )

        metadata_document.close()

    except Exception as exc:

        logger.exception(
            "Failed to read PDF metadata."
        )

        raise ValueError(
            f"Could not read PDF metadata: "
            f"{pdf_path}"
        ) from exc

    # -----------------------------------------------------
    # STEP 7: Create final Document
    # -----------------------------------------------------

    document = Document(

        metadata=metadata,

        pages=pages,
    )

    logger.info(
        "PDF ingestion completed successfully. "
        "Pages: %d, text pages: %d, characters: %d",
        metadata.page_count,
        metadata.text_pages,
        metadata.total_character_count,
    )

    return document