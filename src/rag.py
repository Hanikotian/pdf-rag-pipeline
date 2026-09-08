"""
RAG answer generation.

This module combines:

1. Semantic retrieval using FAISS
2. Cross-Encoder reranking
3. Gemini answer generation
4. Source citation mapping
5. Page-aware citations

Final pipeline:

    User Question
        ↓
    Query Embedding
        ↓
    FAISS retrieves candidate chunks
        ↓
    Cross-Encoder reranks candidates
        ↓
    Best chunks
        ↓
    Gemini
        ↓
    Grounded answer
        ↓
    Source/page citations

Important:

The public helper functions in this module preserve the
original RAG API used by the existing test suite.

Reranking is added internally without breaking the
existing DocumentChunk + score tuple structure.
"""

import json
import logging
import os
import time 
RERANK_TOP_K = 5
# Number of times Gemini will be called if a temporary
# server-side failure occurs.
GEMINI_MAX_RETRIES = 3

# Initial delay before retrying a temporary Gemini failure.
# The delay increases after each failed attempt.
GEMINI_RETRY_DELAY = 2

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.models import DocumentChunk
from src.reranking import rerank_chunks
from src.retrieval import retrieve_relevant_chunks
from src.vector_store import VectorStore


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

logger = logging.getLogger(__name__)


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


# FAISS performs broad candidate retrieval.
#
# We retrieve more chunks than Gemini ultimately needs
# because the Cross-Encoder will perform a second-stage
# relevance ranking.
RETRIEVAL_TOP_K = 10


# Number of chunks sent to Gemini after reranking.
RERANK_TOP_K = 5


# =========================================================
# GEMINI RESPONSE MODEL
# =========================================================

class LLMRAGResponse(BaseModel):
    """
    Structured response expected from Gemini.

    source_numbers refers to the numbered SOURCE blocks
    included in the prompt.
    """

    answer: str

    source_numbers: list[int]


# =========================================================
# INTERNAL RERANKED SOURCE MODEL
# =========================================================

class RAGSource(BaseModel):
    """
    Internal representation of a reranked source.

    A source contains:

    - the original document chunk
    - its FAISS similarity score
    - its Cross-Encoder reranker score

    This model is used internally by the new reranking
    pipeline.

    The public RAGResponse continues to use the original
    tuple-based format so existing tests and code remain
    compatible.
    """

    chunk: DocumentChunk

    similarity_score: float

    reranker_score: float


# =========================================================
# FINAL RAG RESPONSE
# =========================================================

class RAGResponse(BaseModel):
    """
    Final response returned by the RAG pipeline.

    The sources and cited_sources fields intentionally
    preserve the original tuple-based API:

        (DocumentChunk, score)

    This prevents the reranking upgrade from breaking
    existing code and tests.
    """

    answer: str

    sources: list[
        tuple[
            DocumentChunk,
            float,
        ]
    ]

    cited_sources: list[
        tuple[
            DocumentChunk,
            float,
        ]
    ]

    @property
    def page_numbers(
        self,
    ) -> list[int]:
        """
        Return unique page numbers from cited sources only.

        Uncited retrieved chunks are deliberately ignored.

        This ensures that the final answer does not claim
        support from pages that Gemini did not actually cite.
        """

        pages = []

        for chunk, _score in (
            self.cited_sources
        ):

            for page_number in (
                chunk.page_numbers
            ):

                if page_number not in pages:

                    pages.append(
                        page_number
                    )

        return sorted(pages)


# =========================================================
# GEMINI CLIENT
# =========================================================

def create_gemini_client():
    """
    Create the Gemini client.

    The API key is read from the GEMINI_API_KEY
    environment variable.

    The key should never be hard-coded in this file.
    """

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY is not configured."
        )

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# CONTEXT BUILDING
# =========================================================

def build_context(
    results: list[
        tuple[
            DocumentChunk,
            float,
        ]
    ],
) -> str:
    """
    Build numbered document context for Gemini.

    This is the original public helper used by the
    RAG test suite.

    Parameters
    ----------
    results:
        List of:

            (DocumentChunk, score)

    Example:

        [
            (chunk_1, 0.91),
            (chunk_2, 0.82),
        ]

    Returns
    -------
    str
        Numbered context such as:

        SOURCE 1
        Pages: 1
        ...

        SOURCE 2
        Pages: 2
        ...
    """

    if not isinstance(
        results,
        list,
    ):

        raise TypeError(
            "results must be a list."
        )

    context_parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        if not isinstance(
            result,
            tuple,
        ):

            raise TypeError(
                "Each result must be a tuple."
            )

        if len(result) != 2:

            raise ValueError(
                "Each result must contain "
                "(DocumentChunk, score)."
            )

        chunk, score = result

        if not isinstance(
            chunk,
            DocumentChunk,
        ):

            raise TypeError(
                "The first item of each result "
                "must be a DocumentChunk."
            )

        pages = ", ".join(
            str(page)
            for page in chunk.page_numbers
        )

        source_text = f"""
SOURCE {index}
Document: {chunk.source_filename or "Unknown"}
Pages: {pages}
Similarity: {score:.4f}

{chunk.text}
""".strip()

        context_parts.append(
            source_text
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# NEW INTERNAL CONTEXT BUILDER
# =========================================================

def build_rag_context(
    sources: list[RAGSource],
) -> str:
    """
    Build context from internally reranked sources.

    This function converts the richer RAGSource representation
    into the original tuple-based representation and delegates
    to build_context().

    Keeping one actual context-building implementation avoids
    duplicated formatting logic.
    """

    results = []

    for source in sources:

        # For Gemini's context we use the reranker score as
        # the ranking score because these are the scores that
        # determined the final ordering.
        results.append(
            (
                source.chunk,
                source.reranker_score,
            )
        )

    return build_context(
        results
    )


# =========================================================
# PROMPT CONSTRUCTION
# =========================================================

def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the prompt sent to Gemini.

    Gemini is explicitly instructed to:

    - use only supplied document context
    - avoid hallucination
    - identify supporting source numbers
    - never invent page numbers
    """

    return f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document sources.

IMPORTANT RULES:

1. Use only information present in the supplied sources.
2. Do not use outside knowledge.
3. Do not invent, infer, or hallucinate information.
4. If the answer cannot be supported by the sources,
   clearly say that the document does not contain enough
   information to answer the question.
5. Return the source_numbers that directly support your answer.
6. source_numbers must refer only to the SOURCE numbers
   provided below.
7. Do NOT invent page numbers.
8. Only cite sources that actually support the answer.
9. Keep the answer clear and concise unless the question
   requires a detailed explanation.

The response must follow the required structured schema.

USER QUESTION:

{question}

DOCUMENT SOURCES:

{context}
""".strip()


# =========================================================
# SOURCE NUMBER VALIDATION
# =========================================================

def validate_source_numbers(
    source_numbers: list[int],
    total_sources: int,
) -> list[int]:
    """
    Validate source numbers returned by Gemini.

    Invalid source numbers are removed.

    Duplicate source numbers are removed while preserving
    their original order.

    This function is intentionally kept separate from
    parse_rag_response() so it can be tested independently
    if needed.
    """

    if not isinstance(
        source_numbers,
        list,
    ):

        raise TypeError(
            "source_numbers must be a list."
        )

    if total_sources < 0:

        raise ValueError(
            "total_sources cannot be negative."
        )

    valid_numbers = []

    for source_number in (
        source_numbers
    ):

        if not isinstance(
            source_number,
            int,
        ):

            continue

        if not (
            1
            <= source_number
            <= total_sources
        ):

            continue

        if (
            source_number
            not in valid_numbers
        ):

            valid_numbers.append(
                source_number
            )

    return valid_numbers


# =========================================================
# GEMINI RESPONSE PARSING
# =========================================================

def parse_rag_response(
    response,
    number_of_sources: int,
) -> LLMRAGResponse:
    """
    Parse and validate the response returned by Gemini.

    The Gemini SDK can provide structured output through:

        response.parsed

    If that is unavailable, this function falls back to:

        response.text

    followed by JSON parsing.

    Source numbers are validated against the number of
    sources supplied to Gemini.

    Invalid source numbers result in RuntimeError.

    Duplicate source numbers are removed.
    """

    # -----------------------------------------------------
    # First choice:
    # Gemini structured response.
    # -----------------------------------------------------

    parsed_response = getattr(
        response,
        "parsed",
        None,
    )

    # -----------------------------------------------------
    # Fallback:
    # Parse raw response text as JSON.
    # -----------------------------------------------------

    if parsed_response is None:

        response_text = getattr(
            response,
            "text",
            None,
        )

        if not response_text:

            raise RuntimeError(
                "Gemini returned an empty RAG response."
            )

        try:

            response_data = json.loads(
                response_text
            )

            parsed_response = (
                LLMRAGResponse(
                    **response_data
                )
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                "Unable to parse Gemini RAG response."
            ) from exc

    # -----------------------------------------------------
    # Ensure the parsed object follows our schema.
    # -----------------------------------------------------

    if not isinstance(
        parsed_response,
        LLMRAGResponse,
    ):

        try:

            parsed_response = (
                LLMRAGResponse.model_validate(
                    parsed_response
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                "Gemini returned an invalid RAG response."
            ) from exc

    # -----------------------------------------------------
    # Validate source numbers.
    #
    # Unlike validate_source_numbers(), this function
    # treats an invalid citation as a response error.
    #
    # This matches the original test contract.
    # -----------------------------------------------------

    for source_number in (
        parsed_response.source_numbers
    ):

        if not isinstance(
            source_number,
            int,
        ):

            raise RuntimeError(
                "Gemini returned an invalid source number."
            )

        if not (
            1
            <= source_number
            <= number_of_sources
        ):

            raise RuntimeError(
                "Gemini returned a source number "
                "outside the available source range."
            )

    # -----------------------------------------------------
    # Remove duplicate source numbers.
    # -----------------------------------------------------

    cleaned_source_numbers = []

    for source_number in (
        parsed_response.source_numbers
    ):

        if (
            source_number
            not in cleaned_source_numbers
        ):

            cleaned_source_numbers.append(
                source_number
            )

    return LLMRAGResponse(
        answer=parsed_response.answer,
        source_numbers=(
            cleaned_source_numbers
        ),
    )


# =========================================================
# SOURCE MAPPING
# =========================================================

def map_cited_sources(
    results: list[
        tuple[
            DocumentChunk,
            float,
        ]
    ],
    source_numbers: list[int],
) -> list[
    tuple[
        DocumentChunk,
        float,
    ]
]:
    """
    Map Gemini source numbers back to actual chunks.

    Parameters
    ----------
    results:
        The exact source list supplied to Gemini.

    source_numbers:
        Numbers returned by Gemini.

    Example:

        results = [
            (chunk_1, 0.91),
            (chunk_2, 0.82),
        ]

        source_numbers = [2]

    Returns:

        [
            (chunk_2, 0.82)
        ]

    Source numbers are one-based because Gemini sees:

        SOURCE 1
        SOURCE 2
        SOURCE 3
    """

    cited_sources = []

    for source_number in source_numbers:

        if not isinstance(
            source_number,
            int,
        ):

            continue

        if not (
            1
            <= source_number
            <= len(results)
        ):

            continue

        source = results[
            source_number - 1
        ]

        if source not in cited_sources:

            cited_sources.append(
                source
            )

    return cited_sources


# =========================================================
# GEMINI ANSWER GENERATION
# =========================================================

def generate_grounded_answer(
    question: str,
    sources: list[RAGSource],
) -> RAGResponse:
    """
    Generate a grounded answer using Gemini.

    The Gemini prompt receives only the reranked sources.

    The public RAGResponse remains compatible with the
    original tuple-based representation.
    """

    if not sources:

        return RAGResponse(
            answer=(
                "The document does not contain enough "
                "relevant information to answer this question."
            ),
            sources=[],
            cited_sources=[],
        )

    # -----------------------------------------------------
    # Convert internal sources into the public tuple format.
    #
    # We use reranker score for the final ranking score.
    # -----------------------------------------------------

    results = [
        (
            source.chunk,
            source.reranker_score,
        )
        for source in sources
    ]

    # -----------------------------------------------------
    # Build context.
    # -----------------------------------------------------

    context = build_context(
        results
    )

    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    # -----------------------------------------------------
    # Call Gemini.
    # -----------------------------------------------------

    client = create_gemini_client()

    logger.info(
        "Sending RAG request to Gemini."
    )

    # -----------------------------------------------------
    # Call Gemini with retry handling.
    #
    # Gemini can occasionally return temporary server errors
    # such as HTTP 503 when the model is experiencing high
    # demand.
    #
    # These errors are usually transient, so retrying is
    # appropriate before giving up.
    # -----------------------------------------------------

    response = None

    for attempt in range(
        1,
        GEMINI_MAX_RETRIES + 1,
    ):

        try:

            logger.info(
                "Calling Gemini (attempt %d/%d).",
                attempt,
                GEMINI_MAX_RETRIES,
            )

            response = (
                client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=LLMRAGResponse,
                    ),
                )
            )

            # Gemini responded successfully.
            break

        except Exception as exc:

            # -------------------------------------------------
            # We currently treat server-side failures as
            # potentially temporary.
            #
            # We deliberately do not retry indefinitely.
            # -------------------------------------------------

            logger.warning(
                "Gemini request failed on attempt %d/%d: %s",
                attempt,
                GEMINI_MAX_RETRIES,
                exc,
            )

            # If this was the final attempt, expose a clean
            # application-level error instead of allowing the
            # raw SDK traceback to reach the user.
            if attempt == GEMINI_MAX_RETRIES:

                raise RuntimeError(
                    "Gemini is temporarily unavailable. "
                    "Please try again in a few moments."
                ) from exc

            # Exponential backoff:
            #
            # attempt 1 -> 2 seconds
            # attempt 2 -> 4 seconds
            #
            # This avoids immediately hammering the API again.
            delay = (
                GEMINI_RETRY_DELAY
                * (2 ** (attempt - 1))
            )

            logger.info(
                "Retrying Gemini request in %d seconds.",
                delay,
            )

            time.sleep(delay)

    # -----------------------------------------------------
    # Parse Gemini's structured response.
    # -----------------------------------------------------

    parsed_response = parse_rag_response(
        response=response,
        number_of_sources=len(results),
    )

    # -----------------------------------------------------
    # Convert source numbers into actual chunks.
    # -----------------------------------------------------

    cited_sources = map_cited_sources(
        results=results,
        source_numbers=(
            parsed_response.source_numbers
        ),
    )

    return RAGResponse(
        answer=parsed_response.answer,
        sources=results,
        cited_sources=cited_sources,
    )


# =========================================================
# COMPLETE RAG PIPELINE
# =========================================================

def answer_question(
    question: str,
    vector_store: VectorStore,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
) -> RAGResponse:
    """
    Run the complete RAG pipeline.

    Stage 1:
        FAISS retrieves broad candidate chunks.

    Stage 2:
        Cross-Encoder reranks those candidates.

    Stage 3:
        Gemini answers using only the strongest chunks.

    No fixed similarity threshold is used.

    This is important because the earlier IEEE test
    demonstrated that useful chunks can have relatively
    low FAISS similarity scores.
    """

    # -----------------------------------------------------
    # Validate question.
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
    # Validate retrieval settings.
    # -----------------------------------------------------

    if retrieval_top_k <= 0:

        raise ValueError(
            "retrieval_top_k must be greater than zero."
        )

    if rerank_top_k <= 0:

        raise ValueError(
            "rerank_top_k must be greater than zero."
        )

    # -----------------------------------------------------
    # Stage 1:
    # Broad FAISS retrieval.
    # -----------------------------------------------------

    logger.info(
        "Retrieving top %d candidate chunks.",
        retrieval_top_k,
    )

    retrieved_chunks = (
        retrieve_relevant_chunks(
            question=question,
            vector_store=vector_store,
            top_k=retrieval_top_k,
            score_threshold=None,
        )
    )

    if not retrieved_chunks:

        return RAGResponse(
            answer=(
                "The document does not contain enough "
                "relevant information to answer this question."
            ),
            sources=[],
            cited_sources=[],
        )

    # -----------------------------------------------------
    # Stage 2:
    # Cross-Encoder reranking.
    # -----------------------------------------------------

    logger.info(
        "Reranking %d candidate chunks.",
        len(retrieved_chunks),
    )

    reranked_chunks = rerank_chunks(
        question=question,
        retrieved_chunks=retrieved_chunks,
        top_k=rerank_top_k,
    )

    if not reranked_chunks:

        return RAGResponse(
            answer=(
                "The document does not contain enough "
                "relevant information to answer this question."
            ),
            sources=[],
            cited_sources=[],
        )

    # -----------------------------------------------------
    # Convert reranked results into internal RAGSource
    # objects.
    # -----------------------------------------------------

    sources = [
        RAGSource(
            chunk=chunk,
            similarity_score=similarity_score,
            reranker_score=reranker_score,
        )
        for (
            chunk,
            similarity_score,
            reranker_score,
        )
        in reranked_chunks
    ]

    logger.info(
        "Sending %d reranked chunks to Gemini.",
        len(sources),
    )

    # -----------------------------------------------------
    # Stage 3:
    # Grounded Gemini generation.
    # -----------------------------------------------------

    return generate_grounded_answer(
        question=question,
        sources=sources,
    )