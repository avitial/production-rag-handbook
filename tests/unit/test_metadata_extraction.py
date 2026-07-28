"""Unit tests for deterministic metadata extraction."""

from datetime import date

import pytest

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.domain.models import DocumentType
from app.ingestion.metadata_extractor import (
    extract_metadata,
    infer_document_type,
    normalize_ocr_text,
    parse_iso_date,
)


SAMPLE_TEXT = """
*** SYNTHETIC MEDICAL RECORD ***
Patient Demographics
Patient ID: SYN-200849
Date: 2026-07-20
Preferred Language: Vietnamese

Clinical Notes (SOAP)
S: Patient reports routine follow-up.
"""


def test_extracts_patient_id_language_date_and_type() -> None:
    result = extract_metadata(SAMPLE_TEXT)

    assert result.metadata.patient_id == "SYN-200849"
    assert result.metadata.preferred_language == "Vietnamese"
    assert result.metadata.document_date == date(2026, 7, 20)
    assert result.metadata.document_type == DocumentType.CLINICAL_NOTE
    assert result.confidence == pytest.approx(1.0)


def test_evidence_offsets_match_original_normalized_text() -> None:
    normalized = normalize_ocr_text(SAMPLE_TEXT)
    result = extract_metadata(SAMPLE_TEXT)

    for evidence in result.evidence:
        assert (
            normalized[evidence.start_offset:evidence.end_offset]
            == evidence.matched_text
        )


def test_handwritten_flag_forces_handwritten_document_type() -> None:
    result = extract_metadata(
        "Patient ID: SYN-HAND-001\nPLAN\nReturn in two weeks.",
        is_handwritten=True,
    )

    assert result.metadata.is_handwritten is True
    assert (
        result.metadata.document_type
        == DocumentType.HANDWRITTEN_NOTE
    )


def test_missing_patient_and_date_add_warnings() -> None:
    result = extract_metadata("Clinical Notes (SOAP)\nNo identifiers.")

    assert result.metadata.patient_id is None
    assert len(result.warnings) == 2
    assert "Patient ID was not found." in result.warnings
    assert "Document date was not found." in result.warnings


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("DISCHARGE SUMMARY", DocumentType.DISCHARGE_SUMMARY),
        ("Laboratory Results", DocumentType.LAB_REPORT),
        ("Imaging: MRI lumbar", DocumentType.IMAGING_REPORT),
        ("Clinical Notes (SOAP)", DocumentType.CLINICAL_NOTE),
        ("Prescription", DocumentType.PRESCRIPTION),
        ("Referral", DocumentType.REFERRAL),
        ("Unknown form", DocumentType.OTHER),
    ],
)
def test_document_type_inference(text: str, expected: DocumentType) -> None:
    assert infer_document_type(text) == expected


def test_invalid_date_returns_none() -> None:
    assert parse_iso_date("2026-99-99") is None


def test_normalize_text_collapses_repeated_blank_lines() -> None:
    assert normalize_ocr_text("A\r\n\r\n\r\nB\r\n") == "A\n\nB"