"""
Tests for page-aware document chunking.
"""


import pytest

from src.chunking import (
    build_document_segments,
    chunk_document,
    chunk_page,
)
from src.models import (
    Document,
    Page,
)


# =========================================================
# TEST HELPERS
# =========================================================

def create_test_document():
    """
    Create a small two-page document for testing.
    """

    return Document(
        pages=[
            Page(
                page_number=1,
                text=(
                    "Page one contains information "
                    "about the vendor."
                ),
            ),
            Page(
                page_number=2,
                text=(
                    "Page two contains information "
                    "about payment terms."
                ),
            ),
        ]
    )


# =========================================================
# PAGE SEGMENT TESTS
# =========================================================

def test_build_document_segments():

    document = create_test_document()

    segments = build_document_segments(
        document
    )

    assert len(
        segments
    ) == 2

    assert segments[0][0] == 1

    assert segments[1][0] == 2


# =========================================================
# PAGE CHUNK TESTS
# =========================================================

def test_chunk_page_keeps_page_number():

    chunks, next_index = chunk_page(
        page_number=3,
        text=(
            "This is some sample text "
            "from page three."
        ),
        chunk_size=100,
        chunk_overlap=20,
        source_filename="test.pdf",
        starting_chunk_index=0,
    )

    assert len(chunks) == 1

    assert chunks[0].page_numbers == [3]

    assert (
        chunks[0].source_filename
        == "test.pdf"
    )


def test_chunk_page_creates_multiple_chunks():

    text = (
        "A" * 250
    )

    chunks, next_index = chunk_page(
        page_number=1,
        text=text,
        chunk_size=100,
        chunk_overlap=20,
        source_filename="test.pdf",
        starting_chunk_index=0,
    )

    assert len(chunks) > 1

    for chunk in chunks:

        assert chunk.page_numbers == [1]


# =========================================================
# PAGE-AWARE CHUNKING TESTS
# =========================================================

def test_chunks_never_cross_page_boundaries():

    document = create_test_document()

    chunks = chunk_document(
        document,
        chunk_size=20,
        chunk_overlap=5,
    )

    for chunk in chunks:

        assert len(
            chunk.page_numbers
        ) == 1


def test_all_pages_are_represented():

    document = create_test_document()

    chunks = chunk_document(
        document,
        chunk_size=100,
        chunk_overlap=20,
    )

    pages = set()

    for chunk in chunks:

        pages.update(
            chunk.page_numbers
        )

    assert pages == {
        1,
        2,
    }


def test_chunk_ids_are_unique():

    document = create_test_document()

    chunks = chunk_document(
        document,
        chunk_size=20,
        chunk_overlap=5,
    )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(
        chunk_ids
    ) == len(
        set(chunk_ids)
    )


def test_empty_document_returns_empty_list():

    document = Document(
        pages=[]
    )

    chunks = chunk_document(
        document
    )

    assert chunks == []


# =========================================================
# VALIDATION TESTS
# =========================================================

def test_invalid_chunk_size():

    document = create_test_document()

    with pytest.raises(
        ValueError
    ):

        chunk_document(
            document,
            chunk_size=0,
        )


def test_negative_overlap():

    document = create_test_document()

    with pytest.raises(
        ValueError
    ):

        chunk_document(
            document,
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_larger_than_chunk():

    document = create_test_document()

    with pytest.raises(
        ValueError
    ):

        chunk_document(
            document,
            chunk_size=100,
            chunk_overlap=100,
        )