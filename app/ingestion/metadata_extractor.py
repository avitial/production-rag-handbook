"""Deterministic metadata extraction from OCR or native text.

Day 4 begins with rule-based extraction because it is:

- Reproducible
- Easy to test
- Transparent
- Suitable for clearly labeled synthetic forms

Pseudo-code:

    normalize text
    for each metadata field:
        run one or more regular expressions
        if a match exists:
            normalize the captured value
            add evidence with exact offsets
    infer document type from headings
    combine evidence into DocumentMetadata
    calculate simple confidence
    return metadata, evidence, and warnings
"""

from __future__ import annotations

from datetime import date, datetime
import re

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.domain.models import (
    DocumentMetadata,
    DocumentType,
    MetadataEvidence,
    MetadataExtractionResult,
)


PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "patient_id": (
        re.compile(
            r"\bPatient\s*ID\s*:\s*([A-Za-z0-9_-]+)",
            re.IGNORECASE,
        ),
        # OCR may occasionally remove the space between Patient and ID.
        re.compile(
            r"\bPatientID\s*:\s*([A-Za-z0-9_-]+)",
            re.IGNORECASE,
        ),
    ),
    "document_date": (
        re.compile(
            r"\b(?:Document\s*Date|Date)\s*:\s*"
            r"(\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
    ),
    "provider_name": (
        re.compile(
            r"\b(?:Provider|Physician|Clinician)\s*:\s*([^\n]+)",
            re.IGNORECASE,
        ),
    ),
    "preferred_language": (
        re.compile(
            r"\bPreferred\s*Language\s*:\s*([^\n]+)",
            re.IGNORECASE,
        ),
    ),
}


def normalize_ocr_text(text: str) -> str:
    """Normalize line endings and remove repeated blank lines."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]

    output: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        output.append(line)
        previous_blank = blank

    return "\n".join(output).strip()


def _first_match(
    text: str,
    field_name: str,
) -> tuple[str, MetadataEvidence] | None:
    """Return the first successful rule and its source evidence."""
    for pattern in PATTERNS[field_name]:
        match = pattern.search(text)
        if match:
            raw_value = match.group(1).strip()
            evidence = MetadataEvidence(
                field_name=field_name,
                value=raw_value,
                matched_text=match.group(0),
                start_offset=match.start(),
                end_offset=match.end(),
            )
            return raw_value, evidence
    return None


def infer_document_type(
    text: str,
    *,
    is_handwritten: bool = False,
) -> DocumentType:
    """Infer a broad document type from explicit headings.

    This is a deterministic baseline, not a clinical classifier.
    """
    lowered = text.lower()

    if is_handwritten:
        return DocumentType.HANDWRITTEN_NOTE
    if "discharge summary" in lowered:
        return DocumentType.DISCHARGE_SUMMARY
    if "laboratory" in lowered or "lab report" in lowered:
        return DocumentType.LAB_REPORT
    if "imaging" in lowered or "radiology" in lowered:
        return DocumentType.IMAGING_REPORT
    if "clinical notes" in lowered or "soap" in lowered:
        return DocumentType.CLINICAL_NOTE
    if "prescription" in lowered:
        return DocumentType.PRESCRIPTION
    if "referral" in lowered:
        return DocumentType.REFERRAL

    return DocumentType.OTHER


def parse_iso_date(value: str) -> date | None:
    """Parse YYYY-MM-DD; return None rather than raising on OCR corruption."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def extract_metadata(
    text: str,
    *,
    is_handwritten: bool = False,
) -> MetadataExtractionResult:
    """Extract medical-document metadata with evidence.

    Confidence is intentionally simple:

        extracted core fields / expected core fields

    Core fields for Day 4:
    - patient ID
    - document type
    - date

    Because document type is always inferred, the score should not be treated
    as a calibrated probability.
    """
    normalized = normalize_ocr_text(text)
    evidence: list[MetadataEvidence] = []
    warnings: list[str] = []

    patient_id: str | None = None
    preferred_language: str | None = None
    provider_name: str | None = None
    document_date: date | None = None

    patient_match = _first_match(normalized, "patient_id")
    if patient_match:
        patient_id, item = patient_match
        evidence.append(item)
    else:
        warnings.append("Patient ID was not found.")

    language_match = _first_match(normalized, "preferred_language")
    if language_match:
        preferred_language, item = language_match
        evidence.append(item)

    provider_match = _first_match(normalized, "provider_name")
    if provider_match:
        provider_name, item = provider_match
        evidence.append(item)

    date_match = _first_match(normalized, "document_date")
    if date_match:
        raw_date, item = date_match
        document_date = parse_iso_date(raw_date)
        evidence.append(item)
        if document_date is None:
            warnings.append(
                f"Document date could not be parsed: {raw_date}"
            )
    else:
        warnings.append("Document date was not found.")

    document_type = infer_document_type(
        normalized,
        is_handwritten=is_handwritten,
    )

    metadata = DocumentMetadata(
        patient_id=patient_id,
        document_type=document_type,
        document_date=document_date,
        provider_name=provider_name,
        preferred_language=preferred_language,
        is_handwritten=is_handwritten,
    )

    core_found = sum(
        [
            patient_id is not None,
            document_type != DocumentType.OTHER,
            document_date is not None,
        ]
    )
    confidence = core_found / 3

    return MetadataExtractionResult(
        metadata=metadata,
        confidence=confidence,
        evidence=evidence,
        warnings=warnings,
    )