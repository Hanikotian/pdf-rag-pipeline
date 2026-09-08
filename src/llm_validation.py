"""
Validation and error handling for LLM-based extraction.

This module is intentionally separate from the Gemini client so that
validation logic can be tested without making API calls.
"""

from typing import Any


class LLMExtractionError(Exception):
    """
    Base exception for controlled LLM extraction failures.
    """

    pass


class LLMInputError(LLMExtractionError):
    """
    Raised when the document sent to the LLM is invalid.
    """

    pass


class LLMResponseError(LLMExtractionError):
    """
    Raised when the LLM response cannot be parsed or validated.
    """

    pass


def validate_llm_input(document_text: str) -> None:
    """
    Validate document text before sending it to Gemini.
    """

    # The extraction pipeline expects text.
    if not isinstance(document_text, str):
        raise LLMInputError(
            "Document text must be a string."
        )

    # Do not send empty documents to the LLM.
    if not document_text.strip():
        raise LLMInputError(
            "Document text is empty."
        )

    # Basic protection against accidentally sending
    # an extremely large document in one request.
    max_characters = 100_000

    if len(document_text) > max_characters:
        raise LLMInputError(
            f"Document exceeds the maximum allowed size "
            f"of {max_characters} characters."
        )


def validate_contract(contract: Any) -> None:
    """
    Validate the structured Contract returned by the LLM.

    This is business-level validation. Schema validation alone
    cannot guarantee that the extracted values make sense.
    """

    if contract is None:
        raise LLMResponseError(
            "LLM returned no contract."
        )

    # ---------------------------------------------------------
    # Document ID
    # ---------------------------------------------------------

    if not getattr(contract, "document_id", None):
        raise LLMResponseError(
            "Document ID was not extracted."
        )

    # ---------------------------------------------------------
    # Vendor
    # ---------------------------------------------------------

    vendor = getattr(
        contract,
        "vendor",
        None
    )

    if vendor is None:
        raise LLMResponseError(
            "Vendor information was not extracted."
        )

    if not getattr(vendor, "name", None):
        raise LLMResponseError(
            "Vendor name was not extracted."
        )

    # ---------------------------------------------------------
    # Customer
    # ---------------------------------------------------------

    customer = getattr(
        contract,
        "customer",
        None
    )

    if customer is None:
        raise LLMResponseError(
            "Customer information was not extracted."
        )

    if not getattr(customer, "name", None):
        raise LLMResponseError(
            "Customer name was not extracted."
        )

    # ---------------------------------------------------------
    # Contract information
    # ---------------------------------------------------------

    contract_info = getattr(
        contract,
        "contract",
        None
    )

    if contract_info is None:
        raise LLMResponseError(
            "Contract information was not extracted."
        )

    # ---------------------------------------------------------
    # Contract value
    # ---------------------------------------------------------

    contract_value = getattr(
        contract_info,
        "contract_value",
        None
    )

    if contract_value is None:
        raise LLMResponseError(
            "Contract value was not extracted."
        )

    if contract_value <= 0:
        raise LLMResponseError(
            "Contract value must be greater than zero."
        )

    # ---------------------------------------------------------
    # Currency
    # ---------------------------------------------------------

    currency = getattr(
        contract_info,
        "currency",
        None
    )

    if not currency:
        raise LLMResponseError(
            "Currency was not extracted."
        )

    # ---------------------------------------------------------
    # Billing frequency
    # ---------------------------------------------------------

    billing_frequency = getattr(
        contract_info,
        "billing_frequency",
        None
    )

    if not billing_frequency:
        raise LLMResponseError(
            "Billing frequency was not extracted."
        )

    # ---------------------------------------------------------
    # Payment terms
    # ---------------------------------------------------------

    payment_terms = getattr(
        contract_info,
        "payment_terms",
        None
    )

    if not payment_terms:
        raise LLMResponseError(
            "Payment terms were not extracted."
        )