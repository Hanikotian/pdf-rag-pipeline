"""
Tests for the document indexing pipeline.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.indexing import (
    build_index,
    load_index,
    validate_pdf_path,
)
from src.models import (
    Document,
    DocumentChunk,
    Page,
)


# =========================================================
# TEST HELPERS
# =========================================================

def create_test_pdf(
    tmp_path: Path,
) -> Path:
    """
    Create a minimal fake PDF path.

    The actual PDF contents are not needed for the
    validation tests.
    """

    pdf_path = (
        tmp_path / "test.pdf"
    )

    pdf_path.write_bytes(
        b"fake pdf content"
    )

    return pdf_path


# =========================================================
# VALIDATION TESTS
# =========================================================

def test_pdf_path_must_be_path_object():

    with pytest.raises(TypeError):

        validate_pdf_path(
            "test.pdf"
        )


def test_missing_pdf_is_rejected(
    tmp_path,
):

    pdf_path = (
        tmp_path / "missing.pdf"
    )

    with pytest.raises(
        FileNotFoundError
    ):

        validate_pdf_path(
            pdf_path
        )


def test_non_pdf_is_rejected(
    tmp_path,
):

    file_path = (
        tmp_path / "document.txt"
    )

    file_path.write_text(
        "test"
    )

    with pytest.raises(
        ValueError
    ):

        validate_pdf_path(
            file_path
        )


# =========================================================
# INDEXING PIPELINE TEST
# =========================================================

@patch(
    "src.indexing.extract_text_from_pdf"
)
@patch(
    "src.indexing.generate_embeddings"
)
def test_build_index(
    mock_embeddings,
    mock_ingestion,
    tmp_path,
):

    pdf_path = create_test_pdf(
        tmp_path
    )

    index_directory = (
        tmp_path / "index"
    )

    # -----------------------------------------------------
    # Fake document returned by ingestion.
    # -----------------------------------------------------

    document = Document(
        pages=[
            Page(
                page_number=1,
                text=(
                    "Payment terms are "
                    "Net 30 days."
                ),
            )
        ]
    )

    mock_ingestion.return_value = (
        document
    )

    # -----------------------------------------------------
    # Fake chunk.
    # -----------------------------------------------------

    chunk = DocumentChunk(
        chunk_id="chunk-0000",
        text="Payment terms are Net 30 days.",
        page_numbers=[1],
        character_count=31,
        source_filename="test.pdf",
    )

    # -----------------------------------------------------
    # Fake embedding.
    # -----------------------------------------------------

    mock_embeddings.return_value = [
        [1.0, 0.0]
    ]

    # -----------------------------------------------------
    # Run indexing.
    # -----------------------------------------------------

    vector_store = build_index(
        pdf_path=pdf_path,
        index_directory=index_directory,
    )

    # -----------------------------------------------------
    # Verify ingestion was called.
    # -----------------------------------------------------

    mock_ingestion.assert_called_once_with(
        pdf_path
    )

    # -----------------------------------------------------
    # Verify the resulting vector store.
    # -----------------------------------------------------

    assert vector_store.index is not None


def test_load_missing_index_is_rejected(
    tmp_path,
):

    index_directory = (
        tmp_path / "missing_index"
    )

    with pytest.raises(
        FileNotFoundError
    ):

        load_index(
            index_directory
        )