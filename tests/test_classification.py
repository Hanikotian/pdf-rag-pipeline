import pytest
from src.classification import classify_document


def test_contract_classification():

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

    Payment Terms Net 30 days
    """

    result = classify_document(
        document_text
    )

    assert result.document_type == "contract"

    assert 0 <= result.confidence <= 1

def test_research_paper_classification():

    document_text = """
    Deep Learning for Medical Image Analysis

    Abstract

    This research paper presents a novel deep learning
    architecture for detecting abnormalities in MRI images.

    1. Introduction

    Medical image analysis has become an important research
    area in artificial intelligence.

    2. Methodology

    The proposed model was evaluated on a publicly available
    medical imaging dataset.

    3. Results

    Experimental results demonstrate improved classification
    accuracy compared with existing approaches.

    References

    [1] Author et al., Deep Learning Research, 2025.
    """

    result = classify_document(
        document_text
    )

    assert result.document_type == "research_paper"
    assert 0 <= result.confidence <= 1

@pytest.mark.integration
def test_invoice_classification():

    document_text = """
    INVOICE

    Invoice Number: INV-2026-00142
    Invoice Date: 15 August 2026

    Vendor:
    ABC Technologies Pvt. Ltd.

    Bill To:
    XYZ Solutions Pvt. Ltd.

    Description:
    Software Development Services

    Quantity: 1
    Amount: INR 150,000

    Subtotal: INR 150,000
    GST: INR 27,000
    Total Amount: INR 177,000

    Payment Due: 30 August 2026
    """

    result = classify_document(
        document_text
    )

    assert result.document_type == "invoice"
    assert 0 <= result.confidence <= 1