"""
Data models used by the PDF-RAG pipeline.

These models define:
- Generic PDF documents and pages
- PDF metadata and extraction quality
- Provenance/evidence
- Contract parties
- Contract details
- SLA information

The generic document models are intentionally separated
from contract-specific models so that the ingestion layer
can work with ANY PDF type.
"""

from datetime import date

from pydantic import BaseModel, Field


# =========================================================
# GENERIC PDF MODELS
# =========================================================

class Page(BaseModel):
    """
    Represents one page extracted from a PDF.

    This model is document-type independent.

    It does not care whether the PDF is:
    - a contract
    - research paper
    - invoice
    - report
    - resume
    - etc.
    """

    # Page number in the original PDF.
    #
    # We use 1-based numbering because PDF readers
    # normally display the first page as page 1.
    page_number: int

    # Text extracted from this page.
    text: str

    # Number of characters successfully extracted.
    #
    # Useful for determining whether a page contains
    # meaningful text.
    character_count: int = 0

    # Indicates whether this page appears to contain
    # usable machine-readable text.
    #
    # False can indicate an image-only/scanned page.
    has_text: bool = True


class DocumentMetadata(BaseModel):
    """
    Metadata and quality information about a PDF.

    This information is generated during the ingestion
    stage and is independent of the document's business
    type.
    """

    # Original filename of the PDF.
    filename: str

    # Number of pages in the PDF.
    page_count: int

    # PDF metadata fields.
    #
    # These may not exist in every PDF, so they are optional.
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    producer: str | None = None

    # -----------------------------------------------------
    # Extraction statistics
    # -----------------------------------------------------

    # Total number of characters extracted from the
    # entire document.
    total_character_count: int = 0

    # Number of pages containing usable text.
    text_pages: int = 0

    # Number of pages that appear to have no usable text.
    empty_pages: int = 0

    # Percentage of pages containing usable text.
    #
    # Example:
    #
    # 4 pages total
    # 3 pages with text
    #
    # text_extraction_ratio = 0.75
    text_extraction_ratio: float = 0.0

    # Indicates whether the document contains enough
    # machine-readable text to continue with normal
    # text-based processing.
    has_usable_text: bool = False


class Document(BaseModel):
    """
    Represents a complete PDF document.

    A document contains:
    - metadata about the PDF
    - extracted pages

    This is the main object produced by our ingestion layer.
    """

    # Generic PDF metadata and extraction statistics.
    #
    # A default is provided so existing code that creates:
    #
    #     Document(pages=[...])
    #
    # continues to work.
    metadata: DocumentMetadata | None = None

    # All pages extracted from the PDF.
    pages: list[Page]

class DocumentChunk(BaseModel):
    """
    Represents one chunk of text created from a document.

    Chunks are the units that will later be:
    - embedded
    - stored in a vector database
    - retrieved during RAG
    """

    chunk_id: str

    text: str

    page_numbers: list[int]

    character_count: int

    source_filename: str | None = None
# =========================================================
# PROVENANCE
# =========================================================

class Evidence(BaseModel):
    """
    Records where an extracted value came from.

    page_number:
        The page in the original PDF.

    text:
        The exact text matched during extraction.
    """

    page_number: int
    text: str


# =========================================================
# CONTRACT MODELS
# =========================================================

class Party(BaseModel):
    """
    Represents a party involved in the contract.

    Examples:
    - Vendor
    - Customer
    """

    # Legal name of the party.
    name: str

    # Address is optional because some documents
    # may not contain an address.
    address: str | None = None

    # Evidence showing where the party name came from.
    name_evidence: Evidence | None = None

    # Evidence showing where the party address came from.
    address_evidence: Evidence | None = None


class ContractDetails(BaseModel):
    """
    Represents the main commercial details of a contract.
    """

    # Contract start date.
    effective_date: date

    # Contract end date.
    expiry_date: date

    # Total monetary value of the contract.
    contract_value: float

    # Currency code, for example INR or USD.
    currency: str

    # Billing frequency.
    billing_frequency: str

    # Payment terms.
    payment_terms: str

    # Evidence for each individual contract field.
    #
    # Example:
    #
    # {
    #     "contract_value": Evidence(...),
    #     "payment_terms": Evidence(...)
    # }
    evidence: dict[str, Evidence] = Field(
        default_factory=dict
    )


class SLA(BaseModel):
    """
    Represents one Service Level Agreement metric.
    """

    # Name of the SLA metric.
    metric: str

    # Required target.
    target: str

    # How the metric is measured.
    measurement: str

    # Evidence showing where this SLA
    # information was extracted from.
    evidence: Evidence | None = None


class Contract(BaseModel):
    """
    Complete structured representation of a contract.
    """

    # Unique contract/document identifier.
    document_id: str

    # Evidence showing where the document ID
    # was extracted from.
    document_id_evidence: Evidence | None = None

    # Contract parties.
    vendor: Party
    customer: Party

    # Commercial contract information.
    contract: ContractDetails

    # Service-level agreements.
    sla: list[SLA]