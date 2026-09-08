"""
Unit tests for LLM input and output validation.

These tests do NOT call Gemini.
"""

import pytest

from src.llm_validation import (
    LLMInputError,
    validate_llm_input,
)


def test_empty_document_rejected():
    """
    Empty input must be rejected.
    """

    with pytest.raises(
        LLMInputError
    ):
        validate_llm_input("")


def test_whitespace_document_rejected():
    """
    Whitespace-only input must be rejected.
    """

    with pytest.raises(
        LLMInputError
    ):
        validate_llm_input("     ")


def test_non_string_document_rejected():
    """
    Non-string input must be rejected.
    """

    with pytest.raises(
        LLMInputError
    ):
        validate_llm_input(None)


def test_valid_document_accepted():
    """
    Normal document text should pass.
    """

    validate_llm_input(
        "VENDOR SERVICE AGREEMENT"
    )