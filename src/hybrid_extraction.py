"""
Hybrid contract extraction.

This module combines two extraction strategies:

1. Deterministic extraction
   - Fast
   - Cheap
   - Predictable
   - Uses our existing regex-based extractor

2. LLM extraction
   - Used only when deterministic extraction
     produces missing or low-confidence fields

The goal is to avoid sending the entire document
to the LLM when deterministic extraction is already
good enough.

Pipeline:

    Document
       |
       v
    Deterministic extraction
       |
       v
    Check fields
       |
       +---- All fields reliable ----> Return result
       |
       +---- Missing/low confidence
                    |
                    v
              Gemini extraction
                    |
                    v
                 Merge
                    |
                    v
              Final validation
                    |
                    v
                 Result
"""

import logging
from copy import deepcopy

from src.extraction import extract_contract
from src.llm_extraction import extract_contract_with_llm
from src.llm_validation import validate_contract


logger = logging.getLogger(__name__)


# =========================================================
# FIELD CONFIGURATION
# =========================================================

# These are the fields that our hybrid pipeline currently
# considers important enough to evaluate.
#
# The names correspond to the structure of our Contract model.

REQUIRED_FIELDS = [
    "document_id",
    "vendor.name",
    "customer.name",
    "contract.effective_date",
    "contract.expiry_date",
    "contract.contract_value",
    "contract.currency",
    "contract.billing_frequency",
    "contract.payment_terms",
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_nested_value(
    obj,
    field_path: str,
):
    """
    Retrieve a nested field from an object.

    Example:

        get_nested_value(
            contract,
            "vendor.name"
        )

    returns:

        contract.vendor.name

    This helper allows us to work with field paths
    without hard-coding every field individually.
    """

    current = obj

    for part in field_path.split("."):

        if current is None:
            return None

        # Handle Pydantic models.
        if hasattr(current, part):
            current = getattr(
                current,
                part,
            )

        # Handle dictionaries.
        elif isinstance(current, dict):
            current = current.get(part)

        else:
            return None

    return current


def is_missing(value) -> bool:
    """
    Determine whether an extracted value is missing.

    None and empty strings are considered missing.
    """

    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    return False


def detect_missing_fields(contract) -> list[str]:
    """
    Identify fields that were not successfully extracted.

    Returns a list of field paths.

    Example:

        [
            "vendor.name",
            "contract.payment_terms"
        ]
    """

    missing_fields = []

    for field in REQUIRED_FIELDS:

        value = get_nested_value(
            contract,
            field,
        )

        if is_missing(value):
            missing_fields.append(field)

    return missing_fields


# =========================================================
# CONFIDENCE DETECTION
# =========================================================

def detect_low_confidence_fields(contract) -> list[str]:
    """
    Identify values that technically exist but may not
    be trustworthy enough to use without LLM verification.

    This is intentionally conservative.

    We do NOT mark every field as low confidence.

    The deterministic extractor already has tests proving
    that the current contract document can be extracted
    successfully.

    Low-confidence rules can be expanded later as we process
    more real-world documents.
    """

    low_confidence_fields = []

    # -----------------------------------------------------
    # Contract value
    # -----------------------------------------------------
    #
    # A contract value should be a positive numeric amount.

    contract_value = get_nested_value(
        contract,
        "contract.contract_value",
    )

    if contract_value is not None:

        try:

            if float(contract_value) <= 0:
                low_confidence_fields.append(
                    "contract.contract_value"
                )

        except (TypeError, ValueError):

            low_confidence_fields.append(
                "contract.contract_value"
            )

    # -----------------------------------------------------
    # Currency
    # -----------------------------------------------------

    currency = get_nested_value(
        contract,
        "contract.currency",
    )

    if currency:

        valid_currencies = {
            "INR",
            "USD",
            "EUR",
            "GBP",
            "AED",
            "CAD",
            "AUD",
        }

        if currency.upper() not in valid_currencies:

            low_confidence_fields.append(
                "contract.currency"
            )

    # -----------------------------------------------------
    # Billing frequency
    # -----------------------------------------------------

    billing_frequency = get_nested_value(
        contract,
        "contract.billing_frequency",
    )

    if billing_frequency:

        valid_frequencies = {
            "monthly",
            "quarterly",
            "annually",
            "annual",
            "yearly",
        }

        if billing_frequency.lower() not in valid_frequencies:

            low_confidence_fields.append(
                "contract.billing_frequency"
            )

    # -----------------------------------------------------
    # Payment terms
    # -----------------------------------------------------

    payment_terms = get_nested_value(
        contract,
        "contract.payment_terms",
    )

    if payment_terms:

        # Payment terms should normally contain something
        # such as "Net 30 days".
        #
        # If there is no number, the value is suspicious.

        if not any(
            character.isdigit()
            for character in payment_terms
        ):
            low_confidence_fields.append(
                "contract.payment_terms"
            )

    return low_confidence_fields


# =========================================================
# FIELD ANALYSIS
# =========================================================

def identify_fields_requiring_llm(
    contract,
) -> list[str]:
    """
    Determine which fields require LLM assistance.

    A field requires Gemini if:

        missing
        OR
        low confidence

    Duplicate fields are removed automatically.
    """

    missing_fields = detect_missing_fields(
        contract
    )

    low_confidence_fields = detect_low_confidence_fields(
        contract
    )

    fields = list(
        dict.fromkeys(
            missing_fields
            + low_confidence_fields
        )
    )

    logger.info(
        "Missing fields: %s",
        missing_fields,
    )

    logger.info(
        "Low-confidence fields: %s",
        low_confidence_fields,
    )

    logger.info(
        "Fields requiring LLM extraction: %s",
        fields,
    )

    return fields


# =========================================================
# MERGING
# =========================================================

def merge_llm_result(
    deterministic_contract,
    llm_contract,
    fields_to_replace: list[str],
):
    """
    Merge selected fields from the LLM result into the
    deterministic result.

    IMPORTANT:

    We do NOT blindly replace the entire deterministic
    contract with the LLM response.

    Only fields explicitly identified as missing or
    low-confidence are replaced.

    This is one of the most important principles of a
    production hybrid extraction pipeline.
    """

    # Create a copy so that the original deterministic
    # result is not modified unexpectedly.
    merged_contract = deepcopy(
        deterministic_contract
    )

    for field_path in fields_to_replace:

        llm_value = get_nested_value(
            llm_contract,
            field_path,
        )

        # If Gemini failed to provide a value,
        # keep the deterministic value.
        if is_missing(llm_value):

            logger.warning(
                "LLM did not provide value for %s. "
                "Keeping deterministic value.",
                field_path,
            )

            continue

        # -------------------------------------------------
        # Split nested path
        # -------------------------------------------------

        parts = field_path.split(".")

        target = merged_contract

        # Navigate to the parent object.
        for part in parts[:-1]:

            if hasattr(target, part):

                target = getattr(
                    target,
                    part,
                )

            elif isinstance(target, dict):

                target = target[part]

            else:

                logger.warning(
                    "Could not navigate field path: %s",
                    field_path,
                )

                target = None
                break

        if target is None:
            continue

        final_field = parts[-1]

        # -------------------------------------------------
        # Set the new value
        # -------------------------------------------------

        if hasattr(target, final_field):

            setattr(
                target,
                final_field,
                llm_value,
            )

        elif isinstance(target, dict):

            target[final_field] = llm_value

        logger.info(
            "Replaced field '%s' using LLM result.",
            field_path,
        )

    return merged_contract


# =========================================================
# MAIN HYBRID EXTRACTION
# =========================================================

def extract_contract_hybrid(
    document,
):
    """
    Perform hybrid contract extraction.

    Parameters
    ----------
    document:
        The Document object produced by the PDF ingestion
        layer.

    Returns
    -------
    Contract
        Final validated contract.

    Process
    -------
    1. Deterministic extraction.
    2. Identify missing/low-confidence fields.
    3. If necessary, call Gemini.
    4. Merge only required fields.
    5. Validate final contract.
    """

    logger.info(
        "Starting hybrid contract extraction."
    )

    # =====================================================
    # STEP 1: Deterministic extraction
    # =====================================================

    try:

        deterministic_contract = extract_contract(
            document
        )

        logger.info(
            "Deterministic extraction completed successfully."
        )

    except Exception as exc:

        logger.exception(
            "Deterministic extraction failed."
        )

        # -------------------------------------------------
        # Current Stage 1 behavior:
        #
        # If deterministic extraction completely fails,
        # we fall back to full LLM extraction.
        #
        # Later we can make this more sophisticated by
        # identifying exactly which fields failed.
        # -------------------------------------------------

        document_text = "\n".join(
            page.text
            for page in document.pages
        )

        logger.warning(
            "Falling back to full LLM extraction."
        )

        return extract_contract_with_llm(
            document_text
        )

    # =====================================================
    # STEP 2: Identify problematic fields
    # =====================================================

    fields_to_extract = identify_fields_requiring_llm(
        deterministic_contract
    )

    # =====================================================
    # STEP 3: No LLM required
    # =====================================================

    if not fields_to_extract:

        logger.info(
            "All deterministic fields are reliable. "
            "Skipping LLM call."
        )

        validate_contract(
            deterministic_contract
        )

        return deterministic_contract

    # =====================================================
    # STEP 4: Prepare document text
    # =====================================================

    document_text = "\n".join(
        page.text
        for page in document.pages
    )

    # =====================================================
    # STEP 5: LLM extraction
    # =====================================================

    logger.info(
        "Calling Gemini for %d field(s).",
        len(fields_to_extract),
    )

    llm_contract = extract_contract_with_llm(
        document_text
    )

    # =====================================================
    # STEP 6: Merge
    # =====================================================

    merged_contract = merge_llm_result(
        deterministic_contract=deterministic_contract,
        llm_contract=llm_contract,
        fields_to_replace=fields_to_extract,
    )

    # =====================================================
    # STEP 7: Final validation
    # =====================================================

    validate_contract(
        merged_contract
    )

    logger.info(
        "Hybrid extraction completed successfully."
    )

    return merged_contract