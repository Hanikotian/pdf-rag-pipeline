"""
Integration tests for Gemini LLM extraction.

These tests make REAL calls to the Gemini API.

They are intentionally marked as "integration" tests so that
the normal test suite does not execute them.

Run normal tests:
    pytest -v

Run Gemini integration tests explicitly:
    pytest -m integration -v

IMPORTANT:
These tests consume Gemini API quota.
Do not run them unnecessarily.
"""

from pathlib import Path

import pytest

from src.ingestion import extract_text_from_pdf
from src.llm_extraction import extract_contract_with_llm


# Path to the actual PDF used by the project.
PDF_PATH = Path(
    "data/raw/rag_sample_vendor_contract.pdf"
)


@pytest.mark.integration
def test_llm_contract_extraction():
    """
    Test Gemini's ability to extract structured information
    from a simple contract text.

    This test makes a REAL Gemini API call.
    """

    document_text = """
    VENDOR SERVICE AGREEMENT

    Document ID: VSA-2026-00417

    Effective Date: 1 July 2026

    Expiry Date: 30 June 2027

    Vendor:
    BluePeak Data Systems Pvt. Ltd.

    Customer:
    Northstar Analytics Pvt. Ltd.

    Contract Value INR 1,860,000

    Billing Frequency Monthly

    Payment Terms Net 30 days
    """

    # Send the document text to Gemini.
    contract = extract_contract_with_llm(
        document_text
    )

    # ---------------------------------------------------------
    # Validate the extracted structured information.
    # ---------------------------------------------------------

    assert (
        contract.document_id
        == "VSA-2026-00417"
    )

    assert (
        contract.vendor.name
        == "BluePeak Data Systems Pvt. Ltd."
    )

    assert (
        contract.customer.name
        == "Northstar Analytics Pvt. Ltd."
    )

    assert (
        contract.contract.contract_value
        == 1860000
    )

    assert (
        contract.contract.currency
        == "INR"
    )

    assert (
        contract.contract.payment_terms
        == "Net 30 days"
    )


@pytest.mark.integration
def test_llm_contract_extraction_from_actual_pdf():
    """
    Test the complete:

        PDF → text extraction → Gemini → structured data

    pipeline.

    This test makes a REAL Gemini API call.
    """

    # ---------------------------------------------------------
    # STEP 1: Make sure the PDF exists.
    # ---------------------------------------------------------

    assert PDF_PATH.exists(), (
        f"PDF file not found: {PDF_PATH}"
    )

    # ---------------------------------------------------------
    # STEP 2: Extract text from the actual PDF.
    # ---------------------------------------------------------

    document = extract_text_from_pdf(
        PDF_PATH
    )

    # ---------------------------------------------------------
    # STEP 3: Make sure the PDF contains pages.
    # ---------------------------------------------------------

    assert len(document.pages) > 0, (
        "PDF contains no pages."
    )

    # ---------------------------------------------------------
    # STEP 4: Combine text from all PDF pages.
    #
    # Gemini receives the extracted text rather than the
    # original PDF binary.
    # ---------------------------------------------------------

    document_text = "\n".join(
        page.text
        for page in document.pages
    )

    assert document_text.strip(), (
        "No text was extracted from the PDF."
    )

    # ---------------------------------------------------------
    # STEP 5: Send the extracted PDF text to Gemini.
    # ---------------------------------------------------------

    contract = extract_contract_with_llm(
        document_text
    )

    # ---------------------------------------------------------
    # STEP 6: Validate the extracted information.
    # ---------------------------------------------------------

    assert contract.document_id == (
        "VSA-2026-00417"
    )

    assert contract.vendor.name == (
        "BluePeak Data Systems Pvt. Ltd."
    )

    assert contract.customer.name == (
        "Northstar Analytics Pvt. Ltd."
    )

    assert contract.contract.contract_value == (
        1860000
    )

    assert contract.contract.currency == (
        "INR"
    )

    assert contract.contract.payment_terms == (
        "Net 30 days"
    )

    assert contract.contract.billing_frequency == (
        "Monthly"
    )