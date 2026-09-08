"""
Manual end-to-end test of the RAG pipeline
with Cross-Encoder reranking.

This script:

1. Loads the existing IEEE FAISS index
2. Asks several questions
3. Shows the final answer
4. Shows Gemini-selected citations
5. Shows retrieval and reranking scores

This is a development script and can be
deleted after verification.
"""

from pathlib import Path

from src.indexing import load_index
from src.rag import answer_question


INDEX_DIRECTORY = Path(
    "data/indexes/IEEE_Paper1"
)


QUESTIONS = [
    (
        "What is the main objective "
        "of this paper?"
    ),
    (
        "What methodology or approach "
        "does the paper propose?"
    ),
    (
        "What are the main results "
        "or findings of the paper?"
    ),
]


def main():
    """
    Run several questions against the
    previously created IEEE index.
    """

    print()
    print("=" * 70)
    print("LOADING VECTOR INDEX")
    print("=" * 70)

    vector_store = load_index(
        INDEX_DIRECTORY
    )

    print(
        f"Loaded {len(vector_store.chunks)} chunks."
    )

    for question in QUESTIONS:

        print()
        print("=" * 70)
        print("QUESTION")
        print("=" * 70)

        print(question)

        response = answer_question(
            question=question,
            vector_store=vector_store,
        )

        print()
        print("ANSWER:")
        print(
            response.answer
        )

        print()
        print("CITED PAGES:")
        print(
            response.page_numbers
        )

        print()
        print(
            "ALL RERANKED SOURCES:"
        )

        for index, source in enumerate(
            response.sources,
            start=1,
        ):

            print()
            print(
                f"SOURCE {index}"
            )

            print(
                "Pages:",
                source.chunk.page_numbers,
            )

            print(
                "FAISS similarity:",
                round(
                    source.similarity_score,
                    4,
                ),
            )

            print(
                "Reranker score:",
                round(
                    source.reranker_score,
                    4,
                ),
            )

            print(
                source.chunk.text[:500]
            )

            print("-" * 70)

        print()
        print(
            "GEMINI-CITED SOURCES:"
        )

        for source in (
            response.cited_sources
        ):

            print(
                "Pages:",
                source.chunk.page_numbers,
            )

            print(
                "Reranker score:",
                round(
                    source.reranker_score,
                    4,
                ),
            )

            print()


if __name__ == "__main__":
    main()