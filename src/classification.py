"""
Document classification using Google Gemini.

This module classifies an ingested PDF into a broad document type.
It does not perform detailed extraction.
"""

import os
import logging

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from src.llm_validation import LLMInputError


logger = logging.getLogger(__name__)

# Load variables from .env
load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "gemini-3.6-flash"


# =========================================================
# RESPONSE MODEL
# =========================================================

class DocumentClassification(BaseModel):
    """
    Structured classification returned by Gemini.
    """

    document_type: str = Field(
        description=(
            "Broad document type. Must be one of: "
            "contract, research_paper, invoice, report, "
            "resume, or other."
        )
    )

    confidence: float = Field(
        description="Confidence score between 0 and 1."
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

def create_gemini_client() -> genai.Client:
    """
    Create a Gemini client using GEMINI_API_KEY.
    """

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_classification_input(
    document_text: str
) -> None:
    """
    Validate text before sending it to Gemini.
    """

    if not isinstance(document_text, str):
        raise LLMInputError(
            "Document text must be a string."
        )

    if not document_text.strip():
        raise LLMInputError(
            "Document text cannot be empty."
        )


# =========================================================
# PROMPT
# =========================================================

def build_classification_prompt(
    document_text: str
) -> str:
    """
    Build the classification prompt.
    """

    return f"""
You are a document classification system.

Classify the following document into exactly ONE of
these categories:

- contract
- research_paper
- invoice
- report
- resume
- other

Rules:

1. Choose the category that best represents the document.
2. Do not invent information.
3. Return only the requested structured response.
4. Confidence must be a number between 0 and 1.

DOCUMENT:

{document_text}
"""


# =========================================================
# CLASSIFICATION
# =========================================================

def classify_document(
    document_text: str
) -> DocumentClassification:
    """
    Classify a document using Gemini.

    Pipeline:

        Validate input
              ↓
        Gemini
              ↓
        Structured response
              ↓
        Pydantic validation
              ↓
        DocumentClassification
    """

    # ---------------------------------------------------------
    # STEP 1: Validate input
    # ---------------------------------------------------------

    validate_classification_input(
        document_text
    )

    logger.info(
        "Document classification input validated."
    )

    # ---------------------------------------------------------
    # STEP 2: Create Gemini client
    # ---------------------------------------------------------

    client = create_gemini_client()

    # ---------------------------------------------------------
    # STEP 3: Build prompt
    # ---------------------------------------------------------

    prompt = build_classification_prompt(
        document_text
    )

    # ---------------------------------------------------------
    # STEP 4: Call Gemini
    # ---------------------------------------------------------

    logger.info(
        "Sending document classification request to Gemini."
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": DocumentClassification,
        },
    )

    # ---------------------------------------------------------
    # STEP 5: Validate response
    # ---------------------------------------------------------

    if response is None:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    if not getattr(response, "parsed", None):
        raise RuntimeError(
            "Gemini did not return a structured classification."
        )

    classification = response.parsed

    # ---------------------------------------------------------
    # STEP 6: Validate category
    # ---------------------------------------------------------

    allowed_types = {
        "contract",
        "research_paper",
        "invoice",
        "report",
        "resume",
        "other",
    }

    if classification.document_type not in allowed_types:
        raise ValueError(
            f"Invalid document type: "
            f"{classification.document_type}"
        )

    # ---------------------------------------------------------
    # STEP 7: Validate confidence
    # ---------------------------------------------------------

    if not 0 <= classification.confidence <= 1:
        raise ValueError(
            "Classification confidence must be between 0 and 1."
        )

    logger.info(
        "Document classified as '%s' with confidence %.2f.",
        classification.document_type,
        classification.confidence,
    )

    return classification