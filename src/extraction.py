"""
Contract extraction module.

Responsibilities:
1. Read an ingested Document.
2. Extract structured contract fields.
3. Capture provenance for every extracted field.
4. Validate the extracted information using Pydantic.
"""

import re
from datetime import datetime

from src.models import (
    Contract,
    ContractDetails,
    Evidence,
    Party,
    SLA,
)


# ---------------------------------------------------------
# Generic extraction helpers
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize whitespace in extracted PDF text.

    PDFs often contain unnecessary line breaks and spaces.
    """

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def find_value(
    pattern: str,
    text: str
) -> str | None:
    """
    Search text using a regular expression.

    Returns the first captured group.
    """

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return None


def find_value_with_evidence(
    pattern: str,
    document
):
    """
    Search every PDF page for a value.

    Returns:
        (
            extracted_value,
            page_number,
            matched_source_text
        )

    Returns None if the value cannot be found.
    """

    for page in document.pages:

        # Normalize this individual page.
        page_text = normalize_text(
            page.text
        )

        match = re.search(
            pattern,
            page_text,
            re.IGNORECASE
        )

        if match:
            return (
                match.group(1).strip(),
                page.page_number,
                match.group(0).strip()
            )

    return None


def parse_date(date_string: str):
    """
    Convert a date such as:

        15 July 2026

    into a Python date object.
    """

    return datetime.strptime(
        date_string.strip(),
        "%d %B %Y"
    ).date()


def make_evidence(
    result
) -> Evidence:
    """
    Convert the result returned by
    find_value_with_evidence() into an Evidence object.
    """

    return Evidence(
        page_number=result[1],
        text=result[2]
    )


# ---------------------------------------------------------
# Main contract extraction
# ---------------------------------------------------------

def extract_contract(document) -> Contract:
    """
    Extract a complete structured contract from a Document.

    Every extracted field is accompanied by provenance
    whenever the field is present in the source PDF.
    """

    # -----------------------------------------------------
    # Combine all pages.
    #
    # We still use the complete document for fields where
    # the existing extraction logic depends on the complete
    # text.
    # -----------------------------------------------------

    full_text = "\n".join(
        page.text
        for page in document.pages
    )

    normalized_text = normalize_text(
        full_text
    )

    # -----------------------------------------------------
    # 1. Document ID
    # -----------------------------------------------------

    document_id_result = find_value_with_evidence(
        r"Document ID:\s*([A-Za-z0-9-]+)",
        document
    )

    if not document_id_result:
        raise ValueError(
            "Could not extract Document ID"
        )

    document_id = document_id_result[0]

    document_id_evidence = make_evidence(
        document_id_result
    )

    # -----------------------------------------------------
    # 2. Effective date
    # -----------------------------------------------------

    effective_date_result = find_value_with_evidence(
        r"Effective Date:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        document
    )

    if not effective_date_result:
        raise ValueError(
            "Could not extract effective date"
        )

    effective_date = parse_date(
        effective_date_result[0]
    )

    effective_date_evidence = make_evidence(
        effective_date_result
    )

    # -----------------------------------------------------
    # 3. Expiry date
    # -----------------------------------------------------

    expiry_date_result = find_value_with_evidence(
        r"Expiry Date:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        document
    )

    if not expiry_date_result:
        raise ValueError(
            "Could not extract expiry date"
        )

    expiry_date = parse_date(
        expiry_date_result[0]
    )

    expiry_date_evidence = make_evidence(
        expiry_date_result
    )

    # -----------------------------------------------------
    # 4. Contract value
    # -----------------------------------------------------

    contract_value_result = find_value_with_evidence(
        r"Contract Value\s+INR\s+([\d,]+)",
        document
    )

    if not contract_value_result:
        raise ValueError(
            "Could not extract contract value"
        )

    contract_value = float(
        contract_value_result[0].replace(",", "")
    )

    contract_value_evidence = make_evidence(
        contract_value_result
    )

    # -----------------------------------------------------
    # 5. Currency
    # -----------------------------------------------------

    currency_result = find_value_with_evidence(
        r"Contract Value\s+(INR)\s+[\d,]+",
        document
    )

    if not currency_result:
        raise ValueError(
            "Could not extract currency"
        )

    currency = currency_result[0]

    currency_evidence = make_evidence(
        currency_result
    )

    # -----------------------------------------------------
    # 6. Billing frequency
    # -----------------------------------------------------

    billing_frequency_result = find_value_with_evidence(
        r"Billing Frequency\s+(Monthly|Quarterly|Annually)",
        document
    )

    if not billing_frequency_result:
        raise ValueError(
            "Could not extract billing frequency"
        )

    billing_frequency = billing_frequency_result[0]

    billing_frequency_evidence = make_evidence(
        billing_frequency_result
    )

    # -----------------------------------------------------
    # 7. Payment terms
    # -----------------------------------------------------

    payment_terms_result = find_value_with_evidence(
        r"Payment Terms\s+(Net\s+\d+\s+days)",
        document
    )

    if not payment_terms_result:
        raise ValueError(
            "Could not extract payment terms"
        )

    payment_terms = payment_terms_result[0]

    payment_terms_evidence = make_evidence(
        payment_terms_result
    )

    # -----------------------------------------------------
    # 8. Vendor name
    # -----------------------------------------------------

    vendor_name_result = find_value_with_evidence(
        r"\band\s+([A-Za-z0-9&.,' -]+?)\s+\(the Vendor\)",
        document
    )

    if not vendor_name_result:
        raise ValueError(
            "Could not extract vendor name"
        )

    vendor_name = vendor_name_result[0]

    vendor_name_evidence = make_evidence(
        vendor_name_result
    )

    # -----------------------------------------------------
    # 9. Vendor address
    # -----------------------------------------------------

    vendor_address_result = find_value_with_evidence(
        r"\(the Vendor\),\s*registered at\s+(.+?)(?:,\s*and\s+)",
        document
    )

    vendor_address = None
    vendor_address_evidence = None

    if vendor_address_result:

        vendor_address = vendor_address_result[0]

        vendor_address_evidence = make_evidence(
            vendor_address_result
        )

    # -----------------------------------------------------
    # 10. Customer name
    # -----------------------------------------------------

    customer_name_result = find_value_with_evidence(
        r"\bbetween\s+(.+?)\s+\(the Customer\)",
        document
    )

    if not customer_name_result:

        # Fallback for documents where the customer
        # appears after a different introductory phrase.
        customer_name_result = find_value_with_evidence(
            r"and\s+([A-Za-z0-9&.,' -]+?)\s+\(the Customer\)",
            document
        )

    if not customer_name_result:
        raise ValueError(
            "Could not extract customer name"
        )

    customer_name = customer_name_result[0]

    customer_name_evidence = make_evidence(
        customer_name_result
    )

    # -----------------------------------------------------
    # 11. Customer address
    # -----------------------------------------------------

    customer_address_result = find_value_with_evidence(
        r"\(the Customer\),\s*registered at\s+(.+?)(?:\.|,)",
        document
    )

    customer_address = None
    customer_address_evidence = None

    if customer_address_result:

        customer_address = customer_address_result[0]

        customer_address_evidence = make_evidence(
            customer_address_result
        )

    # -----------------------------------------------------
    # 12. Vendor model
    # -----------------------------------------------------

    vendor = Party(
        name=vendor_name,
        address=vendor_address,
        name_evidence=vendor_name_evidence,
        address_evidence=vendor_address_evidence,
    )

    # -----------------------------------------------------
    # 13. Customer model
    # -----------------------------------------------------

    customer = Party(
        name=customer_name,
        address=customer_address,
        name_evidence=customer_name_evidence,
        address_evidence=customer_address_evidence,
    )

    # -----------------------------------------------------
    # 14. Contract details
    # -----------------------------------------------------

    contract_details = ContractDetails(
        effective_date=effective_date,
        expiry_date=expiry_date,
        contract_value=contract_value,
        currency=currency,
        billing_frequency=billing_frequency,
        payment_terms=payment_terms,

        evidence={
            "effective_date": effective_date_evidence,
            "expiry_date": expiry_date_evidence,
            "contract_value": contract_value_evidence,
            "currency": currency_evidence,
            "billing_frequency": billing_frequency_evidence,
            "payment_terms": payment_terms_evidence,
        }
    )

    # -----------------------------------------------------
    # 15. SLA extraction
    # -----------------------------------------------------

    sla = []

    # API Availability
    api_result = find_value_with_evidence(
        r"API Availability\s+([\d.]+%)\s+Monthly",
        document
    )

    if api_result:

        sla.append(
            SLA(
                metric="API Availability",
                target=api_result[0],
                measurement="Monthly",
                evidence=make_evidence(
                    api_result
                )
            )
        )

    # Critical Incident Response
    critical_result = find_value_with_evidence(
        r"Critical Incident Response\s+(Within\s+\d+\s+hour)\s+Per incident",
        document
    )

    if critical_result:

        sla.append(
            SLA(
                metric="Critical Incident Response",
                target=critical_result[0],
                measurement="Per incident",
                evidence=make_evidence(
                    critical_result
                )
            )
        )

    # High Priority Incident Response
    high_priority_result = find_value_with_evidence(
        r"High Priority Incident Response\s+(Within\s+\d+\s+hours)\s+Per incident",
        document
    )

    if high_priority_result:

        sla.append(
            SLA(
                metric="High Priority Incident Response",
                target=high_priority_result[0],
                measurement="Per incident",
                evidence=make_evidence(
                    high_priority_result
                )
            )
        )

    # Document Processing Accuracy
    accuracy_result = find_value_with_evidence(
        r"Document Processing Accuracy\s+(≥\s*98%)\s+Monthly sample",
        document
    )

    if accuracy_result:

        sla.append(
            SLA(
                metric="Document Processing Accuracy",
                target=accuracy_result[0],
                measurement="Monthly sample",
                evidence=make_evidence(
                    accuracy_result
                )
            )
        )

    # -----------------------------------------------------
    # 16. Build final validated contract
    # -----------------------------------------------------

    return Contract(
        document_id=document_id,
        document_id_evidence=document_id_evidence,
        vendor=vendor,
        customer=customer,
        contract=contract_details,
        sla=sla,
    )