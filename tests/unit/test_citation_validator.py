"""Unit tests for citation validation."""

from app.generation.context_builder import (
    BuiltContext,
    ContextSource,
)
from app.generation.rag_generator import AnswerCitation
from app.validation.citation_validator import (
    parse_citation_numbers,
    validate_citations,
)


def make_context() -> BuiltContext:
    source = ContextSource(
        source_number=1,
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="record.pdf",
        page_number=2,
        section="Allergies",
        patient_id="SYN-001",
        text="Allergies\nLatex",
        citation_label="record.pdf, page 2, section Allergies",
        rerank_score=0.9,
    )
    return BuiltContext(
        text="[SOURCE 1]\nAllergies\nLatex",
        sources=(source,),
        included_characters=30,
        omitted_source_count=0,
        truncated=False,
    )


def make_citation() -> AnswerCitation:
    return AnswerCitation(
        source_number=1,
        chunk_id="chunk-1",
        filename="record.pdf",
        page_number=2,
        section="Allergies",
        patient_id="SYN-001",
        citation_label="record.pdf, page 2, section Allergies",
    )


def test_parses_unique_citations_in_order() -> None:
    assert parse_citation_numbers(
        "Latex [SOURCE 2] and Shellfish [SOURCE 1] [SOURCE 2]"
    ) == (2, 1)


def test_valid_citation_passes() -> None:
    result = validate_citations(
        answer="Latex is documented. [SOURCE 1]",
        context=make_context(),
        resolved_citations=(make_citation(),),
        abstained=False,
    )

    assert result.valid is True
    assert result.valid_source_numbers == (1,)
    assert result.issues == ()


def test_invalid_source_number_is_rejected() -> None:
    result = validate_citations(
        answer="Latex is documented. [SOURCE 9]",
        context=make_context(),
        resolved_citations=(),
        abstained=False,
    )

    assert result.valid is False
    assert result.invalid_source_numbers == (9,)
    assert any(
        issue.code == "invalid_source_number"
        for issue in result.issues
    )


def test_non_abstaining_answer_requires_citation() -> None:
    result = validate_citations(
        answer="Latex is documented.",
        context=make_context(),
        resolved_citations=(),
        abstained=False,
    )

    assert result.valid is False
    assert any(
        issue.code == "missing_citation"
        for issue in result.issues
    )


def test_abstention_does_not_require_citation() -> None:
    result = validate_citations(
        answer=(
            "I could not find enough explicit evidence in the "
            "provided sources to answer this question."
        ),
        context=make_context(),
        resolved_citations=(),
        abstained=True,
    )

    assert result.valid is True


def test_unresolved_valid_source_is_reported() -> None:
    result = validate_citations(
        answer="Latex is documented. [SOURCE 1]",
        context=make_context(),
        resolved_citations=(),
        abstained=False,
    )

    assert result.valid is False
    assert result.missing_resolved_numbers == (1,)
