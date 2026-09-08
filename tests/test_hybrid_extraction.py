"""
Tests for Stage 1 hybrid extraction.
"""

from pathlib import Path

from src.ingestion import extract_text_from_pdf
from src.hybrid_extraction import extract_contract_hybrid


# ---------------------------------------------------------
# Test PDF
# ---------------------------------------------------------

PDF_PATH = Path(
    "data/raw/rag_sample_vendor_contract.pdf"
)


def test_hybrid_extraction():
    """
    Verify that Stage 1 hybrid extraction
    successfully uses the deterministic extractor.
    """

    # ---------------------------------------------------------
    # STEP 1: Extract the PDF
    # ---------------------------------------------------------

    document = extract_text_from_pdf(
        PDF_PATH
    )

    # ---------------------------------------------------------
    # STEP 2: Run hybrid extraction
    # ---------------------------------------------------------

    contract = extract_contract_hybrid(
        document
    )

    # ---------------------------------------------------------
    # STEP 3: Verify the extracted contract
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


def test_hybrid_extraction_preserves_provenance():
    """
    Verify that deterministic provenance is preserved
    by the hybrid layer.
    """

    document = extract_text_from_pdf(
        PDF_PATH
    )

    contract = extract_contract_hybrid(
        document
    )

    # Evidence should still exist.
    assert contract.contract.evidence[
        "contract_value"
    ] is not None

    assert contract.contract.evidence[
        "payment_terms"
    ] is not None   