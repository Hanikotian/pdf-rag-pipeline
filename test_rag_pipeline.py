"""
End-to-end test for the RAGPipeline.

This test uses an already-built FAISS index and verifies
that the complete RAG pipeline can:

1. Load an existing index.
2. Accept a user question.
3. Retrieve relevant chunks.
4. Rerank the retrieved chunks.
5. Generate a grounded Gemini answer.
6. Return source information and page numbers.

This is an integration test because it makes a real
Gemini API call.
"""

import pytest
from pathlib import Path

from src.pipeline import RAGPipeline


@pytest.mark.integration
def test_rag_pipeline_with_ieee_paper():
    """
    Test the complete RAG pipeline using IEEE_Paper1.
    """

    # ---------------------------------------------------------
    # Existing persistent index
    # ---------------------------------------------------------
    #
    # We already built this index earlier using:
    #
    #     python build_index.py
    #
    # Therefore we do NOT need to process the PDF again.
    # ---------------------------------------------------------

    index_directory = Path(
        "data/indexes/IEEE_Paper1"
    )

    assert index_directory.exists(), (
        f"Index directory not found: {index_directory}"
    )

    # ---------------------------------------------------------
    # Create the RAG pipeline
    # ---------------------------------------------------------

    pipeline = RAGPipeline(
        index_directory=index_directory
    )

    # ---------------------------------------------------------
    # Explicitly load the existing vector index
    # ---------------------------------------------------------

    pipeline.load_existing_index()

    # The pipeline should now be ready to answer questions.

    assert pipeline.is_ready is True

    # The IEEE paper index currently contains 32 chunks.

    assert pipeline.number_of_chunks == 32

    # ---------------------------------------------------------
    # Ask a real question
    # ---------------------------------------------------------

    response = pipeline.ask(
        "What is the main objective of this paper?"
    )

    # ---------------------------------------------------------
    # Validate the response
    # ---------------------------------------------------------

    assert response.answer

    assert isinstance(
        response.answer,
        str,
    )

    assert response.sources

    assert response.cited_sources

    # At least one page should have been cited.

    assert response.page_numbers

    # Page numbers should be positive integers.

    assert all(
        page_number > 0
        for page_number in response.page_numbers
    )

    print()
    print("=" * 60)
    print("RAG PIPELINE TEST")
    print("=" * 60)

    print()
    print("Question:")
    print(
        "What is the main objective of this paper?"
    )

    print()
    print("Answer:")
    print(response.answer)

    print()
    print("Cited pages:")
    print(response.page_numbers)

    print()
    print("Number of retrieved sources:")
    print(len(response.sources))

    print()
    print("=" * 60)