"""Typed API schemas for ingestion, retrieval, and grounded answers.

These schemas define the stable boundary between the internal RAG pipeline and
downstream systems.

Pseudo-code:

    receive an API request
    validate required fields and constraints
    convert internal answer objects into transport-safe dictionaries
    serialize to JSON
    return consistent success or validation-error payloads
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dc_field
from typing import Any


@dataclass(frozen=True)
class IngestRequest:
    """Request to ingest one local file or directory."""

    source_path: str
    recursive: bool = True
    force_reindex: bool = False

    def __post_init__(self) -> None:
        if not self.source_path.strip():
            raise ValueError("source_path must not be blank")


@dataclass(frozen=True)
class IngestResponse:
    """Summary of one ingestion operation."""

    discovered_files: int
    processed_files: int
    skipped_files: int
    failed_files: int
    page_count: int
    chunk_count: int
    indexed_chunk_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryFilters:
    """API-safe retrieval filters."""

    patient_id: str | None = None
    document_id: str | None = None
    filename: str | None = None
    source_format: str | None = None
    page_number: int | None = None

    def __post_init__(self) -> None:
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be at least 1")


@dataclass(frozen=True)
class RAGQueryRequest:
    """Question-answering request."""

    question: str
    filters: QueryFilters = dc_field(default_factory=QueryFilters)
    candidate_k: int = 10
    final_k: int = 5

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be blank")
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        if self.final_k <= 0:
            raise ValueError("final_k must be greater than zero")
        if self.final_k > self.candidate_k:
            raise ValueError("final_k cannot exceed candidate_k")


@dataclass(frozen=True)
class CitationSchema:
    """Citation returned to downstream systems."""

    source_number: int
    chunk_id: str
    filename: str
    page_number: int
    section: str | None
    patient_id: str | None
    citation_label: str

    def __post_init__(self) -> None:
        if self.source_number < 1:
            raise ValueError("source_number must be at least 1")
        if self.page_number < 1:
            raise ValueError("page_number must be at least 1")
        if not self.filename.strip():
            raise ValueError("filename must not be blank")


@dataclass(frozen=True)
class ValidationIssueSchema:
    """One machine-readable validation issue."""

    code: str
    message: str
    severity: str = "error"
    field: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class ValidationSummarySchema:
    """Combined validation state."""

    valid: bool
    citation_valid: bool
    answer_grounded: bool
    json_valid: bool
    issues: tuple[ValidationIssueSchema, ...] = ()


@dataclass(frozen=True)
class RAGAnswerResponse:
    """Structured answer returned to API clients."""

    question: str
    answer: str
    citations: tuple[CitationSchema, ...]
    abstained: bool
    validation: ValidationSummarySchema
    diagnostics: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses to a JSON-safe dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ErrorResponse:
    """Consistent API error payload."""

    error_code: str
    message: str
    details: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
