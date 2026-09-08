"""
High-level RAG application pipeline.

This module provides one simple interface for the complete
PDF RAG system.

The pipeline has two main operations:

1. index_document()
   PDF -> ingestion -> chunking -> embeddings -> FAISS

2. ask()
   question -> retrieval -> reranking -> Gemini -> answer

The lower-level implementation remains separated into
specialized modules.

This module only orchestrates those components.

Architecture:

    PDF
     |
     v
    Indexing Pipeline
     |
     v
    Persistent FAISS Index
     |
     v
    User Question
     |
     v
    RAG Answer Pipeline
     |
     +--> Retrieval
     |
     +--> Reranking
     |
     +--> Gemini
     |
     v
    Grounded Answer + Citations
"""

import logging
from pathlib import Path

from src.indexing import build_index, load_index
from src.rag import RAGResponse, answer_question
from src.vector_store import VectorStore


logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    High-level interface for the PDF RAG system.

    The class manages the currently loaded vector index.

    Example
    -------
    pipeline = RAGPipeline()

    pipeline.index_document(
        Path("data/raw/example.pdf")
    )

    response = pipeline.ask(
        "What is the main objective?"
    )

    print(response.answer)
    """

    def __init__(
        self,
        index_directory: Path | None = None,
    ) -> None:
        """
        Initialize the RAG pipeline.

        Parameters
        ----------
        index_directory:
            Optional directory that should be used for the
            persistent vector index.

            The directory is only remembered at initialization.

            An existing index is NOT loaded automatically.

            To load an existing index explicitly, use:

                pipeline.load_existing_index()

            This design keeps pipeline initialization lightweight
            and allows a new pipeline to be created before a
            document has been indexed.
        """

        self.vector_store: VectorStore | None = None
        self.index_directory: Path | None = None

        # ---------------------------------------------------------
        # Validate and remember the index directory.
        #
        # We intentionally do NOT load the index here.
        # ---------------------------------------------------------

        if index_directory is not None:

            if not isinstance(
                index_directory,
                Path,
            ):
                raise TypeError(
                    "index_directory must be a pathlib.Path."
                )

            self.index_directory = index_directory

            logger.info(
                "Configured RAG index directory: %s",
                index_directory,
            )
    # =========================================================
    # INDEXING
    # =========================================================

    def index_document(
        self,
        pdf_path: Path,
        index_directory: Path | None = None,
    ) -> VectorStore:
        """
        Index a PDF for later question answering.

        Pipeline:

            PDF
             ↓
            Ingestion
             ↓
            Chunking
             ↓
            Embeddings
             ↓
            FAISS
             ↓
            Persistent index

        Parameters
        ----------
        pdf_path:
            Path to the PDF that should be indexed.

        index_directory:
            Optional location where the FAISS index
            should be stored.

            If omitted, the pipeline uses the index
            directory already configured for this instance.

        Returns
        -------
        VectorStore
            The newly created vector store.
        """

        # -----------------------------------------------------
        # Validate PDF path.
        # -----------------------------------------------------

        if not isinstance(
            pdf_path,
            Path,
        ):
            raise TypeError(
                "pdf_path must be a pathlib.Path."
            )

        # -----------------------------------------------------
        # Determine where the index should be stored.
        # -----------------------------------------------------

        if index_directory is not None:

            if not isinstance(
                index_directory,
                Path,
            ):
                raise TypeError(
                    "index_directory must be a pathlib.Path."
                )

            self.index_directory = index_directory

        # -----------------------------------------------------
        # If no directory was supplied, require one.
        # -----------------------------------------------------

        if self.index_directory is None:

            raise ValueError(
                "index_directory must be provided when "
                "no index directory was configured."
            )

        logger.info(
            "Starting RAG document indexing."
        )

        logger.info(
            "PDF: %s",
            pdf_path,
        )

        logger.info(
            "Index directory: %s",
            self.index_directory,
        )

        # -----------------------------------------------------
        # Delegate the actual indexing work to indexing.py.
        #
        # Keeping this logic there means pipeline.py does
        # not need to know how ingestion, chunking,
        # embeddings, or FAISS work internally.
        # -----------------------------------------------------

        self.vector_store = build_index(
            pdf_path=pdf_path,
            index_directory=self.index_directory,
        )

        logger.info(
            "Document indexing completed successfully."
        )

        return self.vector_store

    # =========================================================
    # LOAD EXISTING INDEX
    # =========================================================

    def load_existing_index(
        self,
        index_directory: Path | None = None,
    ) -> VectorStore:
        """
        Load an existing persistent FAISS index.

        This is useful when the application is restarted
        and we want to reuse an already-created knowledge base
        instead of embedding the PDF again.

        Parameters
        ----------
        index_directory:
            Directory containing the saved FAISS index.

            If omitted, the pipeline's configured index
            directory is used.

        Returns
        -------
        VectorStore
            Loaded vector store.
        """

        # -----------------------------------------------------
        # Update the configured directory if one was supplied.
        # -----------------------------------------------------

        if index_directory is not None:

            if not isinstance(
                index_directory,
                Path,
            ):
                raise TypeError(
                    "index_directory must be a pathlib.Path."
                )

            self.index_directory = index_directory

        # -----------------------------------------------------
        # We cannot load an index without knowing where it is.
        # -----------------------------------------------------

        if self.index_directory is None:

            raise ValueError(
                "index_directory must be provided."
            )

        logger.info(
            "Loading existing RAG index."
        )

        self.vector_store = load_index(
            self.index_directory
        )

        logger.info(
            "Existing RAG index loaded successfully."
        )

        return self.vector_store

    # =========================================================
    # QUESTION ANSWERING
    # =========================================================

    def ask(
        self,
        question: str,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> RAGResponse:
        """
        Ask a question about the currently indexed document.

        Pipeline:

            Question
                ↓
            FAISS retrieval
                ↓
            Cross-Encoder reranking
                ↓
            Gemini
                ↓
            Grounded answer
                ↓
            Page-aware citations

        Parameters
        ----------
        question:
            User's question.

        retrieval_top_k:
            Number of candidate chunks retrieved from FAISS.

        rerank_top_k:
            Number of chunks retained after reranking
            and sent to Gemini.

        Returns
        -------
        RAGResponse
            Grounded answer with cited sources.
        """

        # -----------------------------------------------------
        # Make sure a document has been indexed.
        # -----------------------------------------------------

        if self.vector_store is None:

            raise RuntimeError(
                "No document index is loaded. "
                "Call index_document() or "
                "load_existing_index() first."
            )

        # -----------------------------------------------------
        # Validate the question here as well.
        #
        # answer_question() performs its own validation too,
        # but checking at this high-level boundary gives the
        # caller a clear error before entering the lower-level
        # RAG implementation.
        # -----------------------------------------------------

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "question must be a string."
            )

        if not question.strip():

            raise ValueError(
                "question cannot be empty."
            )

        # -----------------------------------------------------
        # Validate retrieval configuration.
        # -----------------------------------------------------

        if retrieval_top_k <= 0:

            raise ValueError(
                "retrieval_top_k must be greater than zero."
            )

        if rerank_top_k <= 0:

            raise ValueError(
                "rerank_top_k must be greater than zero."
            )

        logger.info(
            "Processing RAG question."
        )

        logger.info(
            "Question: %s",
            question,
        )

        # -----------------------------------------------------
        # Delegate the complete question-answering pipeline
        # to rag.py.
        #
        # rag.py handles:
        #
        # FAISS retrieval
        # Cross-Encoder reranking
        # context construction
        # Gemini generation
        # citation mapping
        # -----------------------------------------------------

        response = answer_question(
            question=question,
            vector_store=self.vector_store,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
        )

        logger.info(
            "RAG question completed successfully."
        )

        return response

    # =========================================================
    # STATUS
    # =========================================================

    @property
    def is_ready(self) -> bool:
        """
        Return True if a usable vector index is loaded.
        """

        return (
            self.vector_store is not None
            and len(self.vector_store) > 0
        )

    @property
    def number_of_chunks(self) -> int:
        """
        Return the number of chunks currently indexed.
        """

        if self.vector_store is None:
            return 0

        return len(self.vector_store)