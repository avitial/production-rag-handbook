"""Unit tests for API JSON validation."""

from app.api.schemas import (
    CitationSchema,
    RAGAnswerResponse,
    ValidationSummarySchema,
)
from app.validation.json_validator import (
    validate_rag_response_json,
)


def valid_response() -> RAGAnswerResponse:
    return RAGAnswerResponse(
        question="What allergies are documented?",
        answer="Latex is documented. [SOURCE 1]",
        citations=(
            CitationSchema(
                source_number=1,
                chunk_id="chunk-1",
                filename="record.pdf",
                page_number=1,
                section="Allergies",
                patient_id="SYN-001",
                citation_label=(
                    "record.pdf, page 1, section Allergies"
                ),
            ),
        ),
        abstained=False,
        validation=ValidationSummarySchema(
            valid=True,
            citation_valid=True,
            answer_grounded=True,
            json_valid=True,
        ),
        diagnostics={"latency_ms": 10.0},
    )


def test_valid_schema_round_trips() -> None:
    result = validate_rag_response_json(valid_response())

    assert result.valid is True
    assert result.json_text is not None
    assert result.normalized["question"].startswith("What allergies")


def test_missing_required_fields_fail() -> None:
    result = validate_rag_response_json(
        {
            "question": "Q",
            "answer": "A",
        }
    )

    assert result.valid is False
    assert any(
        issue.code == "missing_field"
        for issue in result.issues
    )


def test_non_serializable_value_fails() -> None:
    result = validate_rag_response_json(
        {
            "question": "Q",
            "answer": "A",
            "citations": [],
            "abstained": False,
            "validation": {},
            "diagnostics": {"bad": object()},
        }
    )

    assert result.valid is False
    assert result.issues[0].code == "not_json_serializable"
