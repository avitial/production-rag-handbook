"""Validate numbered citations in grounded RAG answers.

Validation checks:

- Citation markers use the expected ``[SOURCE N]`` format.
- Every cited source number exists in the built context.
- Resolved citation objects agree with the source map.
- Non-abstaining factual answers contain at least one citation.
- Duplicate citation markers are normalized for reporting.

Pseudo-code:

    parse all [SOURCE N] markers from answer
    compare marker numbers with available context source numbers
    compare markers with resolved citations
    if answer is not an abstention and has no valid citation:
        add missing-citation issue
    return typed validation result
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from app.generation.context_builder import BuiltContext
from app.generation.rag_generator import AnswerCitation


_CITATION_PATTERN = re.compile(
    r"\[SOURCE\s+(\d+)\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationValidationIssue:
    """One citation-specific problem."""

    code: str
    message: str
    source_number: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CitationValidationResult:
    """Complete citation-validation result."""

    valid: bool
    cited_source_numbers: tuple[int, ...]
    valid_source_numbers: tuple[int, ...]
    invalid_source_numbers: tuple[int, ...]
    missing_resolved_numbers: tuple[int, ...]
    issues: tuple[CitationValidationIssue, ...]


def parse_citation_numbers(answer: str) -> tuple[int, ...]:
    """Parse unique source numbers in first-occurrence order."""
    output: list[int] = []

    for match in _CITATION_PATTERN.finditer(answer):
        number = int(match.group(1))
        if number not in output:
            output.append(number)

    return tuple(output)


def validate_citations(
    *,
    answer: str,
    context: BuiltContext,
    resolved_citations: Iterable[AnswerCitation] = (),
    abstained: bool = False,
    require_citation_for_answer: bool = True,
) -> CitationValidationResult:
    """Validate citation markers against context and resolved citations."""
    cited = parse_citation_numbers(answer)
    available = set(context.citation_map)
    resolved = {
        citation.source_number
        for citation in resolved_citations
    }

    invalid = tuple(
        number for number in cited if number not in available
    )
    valid_numbers = tuple(
        number for number in cited if number in available
    )
    missing_resolved = tuple(
        number for number in valid_numbers if number not in resolved
    )

    issues: list[CitationValidationIssue] = []

    for number in invalid:
        issues.append(
            CitationValidationIssue(
                code="invalid_source_number",
                message=(
                    f"Answer cites SOURCE {number}, but that source "
                    "is not present in the context."
                ),
                source_number=number,
            )
        )

    for number in missing_resolved:
        issues.append(
            CitationValidationIssue(
                code="unresolved_source_number",
                message=(
                    f"SOURCE {number} exists in context but was not "
                    "resolved into the response citation list."
                ),
                source_number=number,
            )
        )

    if (
        require_citation_for_answer
        and not abstained
        and answer.strip()
        and not valid_numbers
    ):
        issues.append(
            CitationValidationIssue(
                code="missing_citation",
                message=(
                    "A non-abstaining answer must cite at least one "
                    "valid source."
                ),
            )
        )

    return CitationValidationResult(
        valid=not issues,
        cited_source_numbers=cited,
        valid_source_numbers=valid_numbers,
        invalid_source_numbers=invalid,
        missing_resolved_numbers=missing_resolved,
        issues=tuple(issues),
    )
