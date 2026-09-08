"""
LLM-based structured extraction using Google Gemini.

This module is responsible for:

1. Loading configuration from .env.
2. Validating document input.
3. Sending document text to Gemini.
4. Asking Gemini to extract contract information.
5. Requesting structured JSON from Gemini.
6. Validating the returned structure with Pydantic.
7. Applying application-level contract validation.
8. Retrying temporary Gemini API failures.

Important:
This module does NOT decide provenance.

Provenance remains the responsibility of the application
and our deterministic extraction layer.
"""

# =========================================================
# STANDARD LIBRARY IMPORTS
# =========================================================

import json
import logging
import os
from pathlib import Path


# =========================================================
# THIRD-PARTY IMPORTS
# =========================================================

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types

from pydantic import BaseModel, Field

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# =========================================================
# INTERNAL IMPORTS
# =========================================================

from src.llm_validation import (
    LLMExtractionError,
    LLMInputError,
    LLMResponseError,
    validate_contract,
    validate_llm_input,
)


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# Find the project root.
#
# __file__ points to:
#
#     pdf-rag-pipeline/src/llm_extraction.py
#
# .parent:
#
#     pdf-rag-pipeline/src
#
# .parent again:
#
#     pdf-rag-pipeline
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ---------------------------------------------------------
# Load .env from the project root.
#
# This makes the API key available when:
#
# - running main.py
# - running pytest
# - importing this module
# ---------------------------------------------------------

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_FILE
)


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# Keep the model configurable.
#
# If GEMINI_MODEL is present in .env, it will be used.
#
# Otherwise, use the model that worked in your project.
# ---------------------------------------------------------

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


# =========================================================
# LLM RESPONSE MODELS
# =========================================================

class LLMParty(BaseModel):
    """
    Represents a party extracted by Gemini.
    """

    name: str = Field(
        description=(
            "Full legal name of the party exactly "
            "as written in the document."
        )
    )

    address: str | None = Field(
        default=None,
        description=(
            "Full address of the party if the document "
            "contains an address."
        )
    )


class LLMContractDetails(BaseModel):
    """
    Represents commercial contract information
    extracted by Gemini.
    """

    effective_date: str = Field(
        description=(
            "Contract effective date exactly as stated "
            "in the document."
        )
    )

    expiry_date: str = Field(
        description=(
            "Contract expiry date exactly as stated "
            "in the document."
        )
    )

    contract_value: float = Field(
        description=(
            "Total contract value as a numeric amount. "
            "Remove currency symbols and thousands separators."
        )
    )

    currency: str = Field(
        description=(
            "Three-letter currency code such as INR, USD, "
            "or EUR."
        )
    )

    billing_frequency: str = Field(
        description=(
            "Billing frequency such as Monthly, Quarterly, "
            "or Annually."
        )
    )

    payment_terms: str = Field(
        description=(
            "Payment terms such as Net 30 days."
        )
    )


class LLMContract(BaseModel):
    """
    Structured contract response expected from Gemini.
    """

    document_id: str = Field(
        description=(
            "Unique document or contract identifier."
        )
    )

    vendor: LLMParty = Field(
        description=(
            "Vendor or service provider information."
        )
    )

    customer: LLMParty = Field(
        description=(
            "Customer information."
        )
    )

    contract: LLMContractDetails = Field(
        description=(
            "Main commercial contract information."
        )
    )


# =========================================================
# PROMPT CONSTRUCTION
# =========================================================

def build_extraction_prompt(
    document_text: str
) -> str:
    """
    Build the prompt sent to Gemini.

    The prompt tells the LLM:

    - what the task is
    - which fields to extract
    - not to invent information
    - to use only the supplied document
    - how to handle missing information
    """

    return f"""
You are an information extraction system.

Your task is to extract structured information from
the contract document provided below.

IMPORTANT RULES:

1. Use ONLY information present in the document.
2. Do NOT invent, infer, or hallucinate missing information.
3. Preserve names exactly as they appear in the document.
4. Extract the complete legal names of the vendor and customer.
5. Contract value must be returned as a numeric value.
6. Currency must be returned as a currency code.
7. Preserve payment terms such as "Net 30 days".
8. Preserve dates as written in the document.
9. If an optional address is not present, return null.
10. Do not add explanations outside the structured response.

The application will validate your response against a
strict structured schema.

DOCUMENT:

---------------- BEGIN DOCUMENT ----------------

{document_text}

----------------- END DOCUMENT -----------------
"""


# =========================================================
# GEMINI CLIENT
# =========================================================

def create_gemini_client() -> genai.Client:
    """
    Create a Gemini API client using GEMINI_API_KEY.

    The API key is loaded from the project's .env file.

    The API key is never hard-coded in source code.
    """

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    logger.debug(
        "Gemini API key successfully loaded."
    )

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# GEMINI API CALL WITH RETRIES
# =========================================================

@retry(
    retry=retry_if_exception_type(
        errors.APIError
    ),
    wait=wait_exponential(
        multiplier=1,
        min=2,
        max=10,
    ),
    stop=stop_after_attempt(3),
    reraise=True,
)
def generate_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
):
    """
    Call Gemini with automatic retry handling.

    Temporary API failures are retried up to three times.

    The response schema is supplied directly to Gemini so
    the model is constrained to our LLMContract structure.
    """

    logger.info(
        "Sending request to Gemini model: %s",
        model,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LLMContract,
        ),
    )

    return response


# =========================================================
# RESPONSE PARSING
# =========================================================

def parse_llm_response(
    response
) -> LLMContract:
    """
    Convert Gemini's response into an LLMContract.

    The Gemini SDK can provide a parsed Pydantic object
    when structured output is requested.

    We still keep a JSON fallback so that this function
    remains robust if the SDK returns only response.text.
    """

    # ---------------------------------------------------------
    # Preferred path:
    #
    # Gemini SDK has already parsed the response according
    # to the Pydantic schema.
    # ---------------------------------------------------------

    parsed = getattr(
        response,
        "parsed",
        None
    )

    if parsed is not None:

        if isinstance(
            parsed,
            LLMContract
        ):
            return parsed

        try:

            return LLMContract.model_validate(
                parsed
            )

        except Exception as exc:

            raise LLMResponseError(
                "Gemini returned structured data that "
                "could not be validated."
            ) from exc


    # ---------------------------------------------------------
    # Fallback path:
    #
    # Parse the response text manually.
    # ---------------------------------------------------------

    response_text = getattr(
        response,
        "text",
        None
    )

    if not response_text:

        raise LLMResponseError(
            "Gemini response contained no text."
        )


    try:

        parsed_json = json.loads(
            response_text
        )

    except json.JSONDecodeError as exc:

        raise LLMResponseError(
            "Gemini returned invalid JSON."
        ) from exc


    # ---------------------------------------------------------
    # Validate the JSON against our Pydantic model.
    # ---------------------------------------------------------

    try:

        return LLMContract.model_validate(
            parsed_json
        )

    except Exception as exc:

        raise LLMResponseError(
            "Gemini JSON did not match the expected "
            "contract schema."
        ) from exc


# =========================================================
# MAIN LLM EXTRACTION FUNCTION
# =========================================================

def extract_contract_with_llm(
    document_text: str
) -> LLMContract:
    """
    Extract structured contract information using Gemini.

    Pipeline:

        Document text
              ↓
        Input validation
              ↓
        Prompt construction
              ↓
        Gemini API
              ↓
        Structured JSON
              ↓
        Pydantic validation
              ↓
        Business validation
              ↓
        Valid LLMContract
    """

    # =====================================================
    # STEP 1: VALIDATE INPUT
    # =====================================================

    try:

        validate_llm_input(
            document_text
        )

    except LLMInputError:

        # Preserve the original validation error.
        raise

    logger.info(
        "LLM input validation successful."
    )


    # =====================================================
    # STEP 2: CREATE GEMINI CLIENT
    # =====================================================

    try:

        client = create_gemini_client()

    except RuntimeError as exc:

        logger.exception(
            "Could not create Gemini client."
        )

        raise LLMExtractionError(
            "Could not initialize Gemini client."
        ) from exc


    # =====================================================
    # STEP 3: BUILD EXTRACTION PROMPT
    # =====================================================

    prompt = build_extraction_prompt(
        document_text
    )


    # =====================================================
    # STEP 4: CALL GEMINI
    # =====================================================

    try:

        response = generate_with_retry(
            client=client,
            model=MODEL_NAME,
            prompt=prompt,
        )

    except Exception as exc:

        logger.exception(
            "Gemini API request failed after retries."
        )

        raise LLMExtractionError(
            "Gemini API request failed after retries."
        ) from exc


    # =====================================================
    # STEP 5: CHECK RESPONSE
    # =====================================================

    if response is None:

        raise LLMResponseError(
            "Gemini returned an empty response."
        )


    # -----------------------------------------------------
    # Some Gemini responses expose parsed structured data
    # without requiring response.text.
    #
    # Therefore we check BOTH.
    # -----------------------------------------------------

    parsed_response = getattr(
        response,
        "parsed",
        None
    )

    response_text = getattr(
        response,
        "text",
        None
    )

    if (
        parsed_response is None
        and not response_text
    ):

        raise LLMResponseError(
            "Gemini response contained no usable content."
        )


    # =====================================================
    # STEP 6: PARSE + PYDANTIC VALIDATION
    # =====================================================

    try:

        contract = parse_llm_response(
            response
        )

    except LLMResponseError:

        raise

    except Exception as exc:

        logger.exception(
            "Failed to parse Gemini response."
        )

        raise LLMResponseError(
            "Gemini returned an invalid structured response."
        ) from exc


    # =====================================================
    # STEP 7: BUSINESS VALIDATION
    # =====================================================

    try:

        validate_contract(
            contract
        )

    except Exception as exc:

        logger.exception(
            "LLM contract failed business validation."
        )

        raise LLMResponseError(
            "Extracted contract failed business validation."
        ) from exc


    # =====================================================
    # STEP 8: RETURN VALIDATED CONTRACT
    # =====================================================

    logger.info(
        "LLM contract extraction completed successfully."
    )

    return contract