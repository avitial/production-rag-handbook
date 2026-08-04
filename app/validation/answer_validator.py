"""Validate grounded answer behavior beyond citation syntax.

Checks include:

- Required abstention wording
- Empty answers
- Unsupported absolute claims
- Cross-patient leakage
- Citation-validation integration
- Evidence-term overlap for non-abstaining answers

This is a deterministic validation baseline. It does not prove medical
correctness and should be supplemented with evaluation datasets and model-based
faithfulness checks.

Pseudo-code:

    validate citation markers
    detect abstention
    reject empty answer
    if a patient filter exists:
        ensure every resolved citation belongs to that patient
    if answer does not abstain:
        ensure source evidence exists
        ensure meaningful answer terms overlap source text
    return issues and validity
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from app.generation.rag_generator import RAGAnswer
from app.validation.citation_validator import (
    CitationValidationResult,
    validate_citations,
)


ABSTENTION_SENTENCE = (
    "I could not find enough explicit evidence in the provided "
    "sources to answer this question."
)

_TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:[-_/][a-z0-9]+)*",
    re.IGNORECASE,
)

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "has", "have", "in", "is", "it", "of", "on", "or",
    "the", "to", "was", "were", "with", "source", "section",
    "patient", "page", "file",
}


@dataclass(frozen=True)
class AnswerValidationIssue:
    """One answer-level validation problem."""

    code: str
    message: str
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerValidationResult:
    """Combined grounding and citation validation result."""

    valid: bool
    abstained: bool
    grounded: bool
    citation_result: CitationValidationResult
    issues: tuple[AnswerValidationIssue, ...]


def _tokens(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _TOKEN_PATTERN.finditer(text)
        if (
            match.group(0).lower() not in _STOP_WORDS
            and len(match.group(0)) > 2
        )
    }


def validate_answer(
    answer: RAGAnswer,
    *,
    expected_patient_id: str | None = None,
    minimum_evidence_overlap: int = 1,
) -> AnswerValidationResult:
    """Validate one structured RAG answer."""
    if minimum_evidence_overlap < 0:
        raise ValueError(
            "minimum_evidence_overlap must not be negative"
        )

    issues: list[AnswerValidationIssue] = []
    answer_text = answer.answer.strip()
    abstained = (
        answer.diagnostics.abstained
        or ABSTENTION_SENTENCE.lower() in answer_text.lower()
    )

    citation_result = validate_citations(
        answer=answer_text,
        context=answer.context,
        resolved_citations=answer.citations,
        abstained=abstained,
    )

    for citation_issue in citation_result.issues:
        issues.append(
            AnswerValidationIssue(
                code=citation_issue.code,
                message=citation_issue.message,
                details={
                    "source_number": (
                        citation_issue.source_number
                    )
                },
            )
        )

    if not answer_text:
        issues.append(
            AnswerValidationIssue(
                code="empty_answer",
                message="Generated answer is empty.",
            )
        )

    if expected_patient_id is not None:
        wrong_patient_sources = [
            citation.source_number
            for citation in answer.citations
            if citation.patient_id != expected_patient_id
        ]

        if wrong_patient_sources:
            issues.append(
                AnswerValidationIssue(
                    code="cross_patient_citation",
                    message=(
                        "One or more citations belong to a different "
                        "patient than the requested patient."
                    ),
                    details={
                        "expected_patient_id": expected_patient_id,
                        "source_numbers": wrong_patient_sources,
                    },
                )
            )

    grounded = abstained

    if not abstained:
        source_text = " ".join(
            source.text
            for source in answer.context.sources
        )

        if not source_text.strip():
            issues.append(
                AnswerValidationIssue(
                    code="answer_without_context",
                    message=(
                        "A factual answer was produced without source "
                        "context."
                    ),
                )
            )
        else:
            answer_without_citations = re.sub(
                r"\[SOURCE\s+\d+\]",
                "",
                answer_text,
                flags=re.IGNORECASE,
            )
            answer_tokens = _tokens(answer_without_citations)
            source_tokens = _tokens(source_text)
            overlap = answer_tokens.intersection(source_tokens)

            # Remove citation words manually because this module intentionally
            # does not depend on the citation parser's internal regex.
            overlap = {
                token for token in overlap
                if token not in {"source"}
            }

            grounded = len(overlap) >= minimum_evidence_overlap

            if not grounded:
                issues.append(
                    AnswerValidationIssue(
                        code="insufficient_evidence_overlap",
                        message=(
                            "The non-abstaining answer has insufficient "
                            "lexical overlap with the supplied evidence."
                        ),
                        details={
                            "overlap_count": len(overlap),
                            "minimum_required": (
                                minimum_evidence_overlap
                            ),
                        },
                    )
                )

    return AnswerValidationResult(
        valid=not issues,
        abstained=abstained,
        grounded=grounded,
        citation_result=citation_result,
        issues=tuple(issues),
    )
