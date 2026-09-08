"""
Tests for contract extraction and provenance.

These tests verify two things:

1. The extracted contract values are correct.
2. The extracted values contain correct provenance/evidence.
"""

from pathlib import Path

from src.ingestion import extract_text_from_pdf
from src.extraction import extract_contract


# ---------------------------------------------------------
# Test data
# ---------------------------------------------------------

# Locate the sample PDF from the project root.
PDF_PATH = Path(
    "data/raw/rag_sample_vendor_contract.pdf"
)


# ---------------------------------------------------------
# Main extraction test
# ---------------------------------------------------------

def test_contract_extraction():
    """
    Verify that the contract extractor correctly
    extracts the main contract information.
    """

    # Extract the PDF into our Document model.
    document = extract_text_from_pdf(
        PDF_PATH
    )

    # Extract the structured contract.
    contract = extract_contract(
        document
    )

    # -----------------------------------------------------
    # Verify document ID
    # -----------------------------------------------------

    assert (
        contract.document_id
        == "VSA-2026-00417"
    )

    # -----------------------------------------------------
    # Verify vendor
    # -----------------------------------------------------

    assert (
        contract.vendor.name
        == "BluePeak Data Systems Pvt. Ltd."
    )

    # -----------------------------------------------------
    # Verify customer
    # -----------------------------------------------------

    assert (
        contract.customer.name
        == "Northstar Analytics Pvt. Ltd."
    )

    # -----------------------------------------------------
    # Verify contract value
    # -----------------------------------------------------

    assert (
        contract.contract.contract_value
        == 1860000
    )

    # -----------------------------------------------------
    # Verify currency
    # -----------------------------------------------------

    assert (
        contract.contract.currency
        == "INR"
    )

    # -----------------------------------------------------
    # Verify payment terms
    # -----------------------------------------------------

    assert (
        contract.contract.payment_terms
        == "Net 30 days"
    )


# ---------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------

def test_document_id_provenance():
    """
    Verify that the document ID has provenance
    pointing to the correct PDF page and source text.
    """

    document = extract_text_from_pdf(
        PDF_PATH
    )

    contract = extract_contract(
        document
    )

    evidence = (
        contract.document_id_evidence
    )

    # Evidence must exist.
    assert evidence is not None

    # Document ID appears on page 1.
    assert evidence.page_number == 1

    # The evidence must contain the actual ID.
    assert (
        "VSA-2026-00417"
        in evidence.text
    )


def test_vendor_provenance():
    """
    Verify that the vendor name has provenance.
    """

    document = extract_text_from_pdf(
        PDF_PATH
    )

    contract = extract_contract(
        document
    )

    evidence = (
        contract.vendor.name_evidence
    )

    # Evidence must exist.
    assert evidence is not None

    # Vendor information appears on page 1.
    assert evidence.page_number == 1

    # Evidence must contain the vendor name.
    assert (
        "BluePeak Data Systems Pvt. Ltd."
        in evidence.text
    )


def test_contract_value_provenance():
    """
    Verify that the contract value has provenance.
    """

    document = extract_text_from_pdf(
        PDF_PATH
    )

    contract = extract_contract(
        document
    )

    evidence = (
        contract.contract.evidence[
            "contract_value"
        ]
    )

    # Evidence must exist.
    assert evidence is not None

    # Contract value appears on page 1.
    assert evidence.page_number == 1

    # Evidence must contain the extracted value.
    assert (
        "1,860,000"
        in evidence.text
    )


def test_payment_terms_provenance():
    """
    Verify that the payment terms have provenance.
    """

    document = extract_text_from_pdf(
        PDF_PATH
    )

    contract = extract_contract(
        document
    )

    evidence = (
        contract.contract.evidence[
            "payment_terms"
        ]
    )

    # Evidence must exist.
    assert evidence is not None

    # Payment terms appear on page 1.
    assert evidence.page_number == 1

    # Evidence must contain the extracted value.
    assert (
        "Net 30 days"
        in evidence.text
    )


def test_sla_provenance():
    """
    Verify that SLA records contain provenance.
    """

    document = extract_text_from_pdf(
        PDF_PATH
    )

    contract = extract_contract(
        document
    )

    # Make sure at least one SLA was extracted.
    assert len(contract.sla) > 0

    # Check the first SLA.
    sla = contract.sla[0]

    # SLA evidence must exist.
    assert sla.evidence is not None

    # Evidence must point to a valid PDF page.
    assert sla.evidence.page_number >= 1

    # Evidence must contain the SLA metric.
    assert (
        sla.metric
        in sla.evidence.text
    )