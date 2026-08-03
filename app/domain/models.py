"""Core domain models for local medical-document ingestion and chunking."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field


from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceFormat(StrEnum):
    PDF = "pdf"
    JPEG = "jpeg"
    PNG = "png"
    JPG = "jpg"
    HANDWRITTEN_NOTE = "handwritten_note"
    UNKNOWN = "unknown"


class ExtractionMethod(StrEnum):
    NATIVE_PDF = "native_pdf"
    OCR_PDF = "ocr_pdf"
    OCR_IMAGE = "ocr_image"
    OCR_HANDWRITTEN = "ocr_handwritten"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class DocumentType(StrEnum):
    CLINICAL_NOTE = "clinical_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    LAB_REPORT = "lab_report"
    IMAGING_REPORT = "imaging_report"
    PRESCRIPTION = "prescription"
    REFERRAL = "referral"
    HANDWRITTEN_NOTE = "handwritten_note"
    OTHER = "other"


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    patient_id: str | None = None
    document_type: DocumentType | None = None
    document_date: date | None = None
    provider_name: str | None = None
    preferred_language: str | None = None
    is_handwritten: bool = False

    @field_validator("patient_id", "provider_name", "preferred_language")
    @classmethod
    def blank_strings_become_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class LocalSource(BaseModel):
    """Validated local file source."""

    model_config = ConfigDict(extra="allow")

    path: Path
    source_format: SourceFormat
    is_handwritten: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> "LocalSource":
        if not self.path.exists():
            raise ValueError(f"local source does not exist: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"local source is not a file: {self.path}")
        return self


class DocumentPage(BaseModel):
    """One extracted page or one image treated as page 1."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_format: SourceFormat
    page_number: int = Field(ge=1)
    text: str
    extraction_method: ExtractionMethod = ExtractionMethod.UNKNOWN
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    source_hash: str | None = None

    @field_validator("text")
    @classmethod
    def normalize_line_endings(cls, value: str) -> str:
        return value.replace("\r\n", "\n").replace("\r", "\n")


class DocumentChunk(BaseModel):
    """Retrievable page-bounded passage."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_format: SourceFormat
    page_number: int = Field(ge=1)
    section: str | None = None
    text: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    metadata: dict[str, Any] = field(default_factory=dict)

    @field_validator("section")
    @classmethod
    def blank_section_becomes_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text must not be blank")
        return value

    @model_validator(mode="after")
    def validate_offsets(self) -> "DocumentChunk":
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if self.end_offset - self.start_offset != len(self.text):
            raise ValueError("offset span must equal chunk text length")
        return self


class OCRResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    confidence: float | None = Field(default=None, ge=0, le=100)
    extraction_method: ExtractionMethod
    language: str = "eng"
    duration_ms: float = Field(ge=0)
    source_path: str
    page_number: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)



class MetadataEvidence(BaseModel):
    field_name: str
    value: str
    matched_text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class MetadataExtractionResult(BaseModel):
    metadata: DocumentMetadata
    confidence: float = Field(ge=0, le=1)
    evidence: list[MetadataEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class IngestionReport:
    """Counters and warnings returned from one ingestion run."""

    discovered_files: int
    processed_files: int
    skipped_files: int
    failed_files: int
    page_count: int
    chunk_count: int
    indexed_chunk_count: int
    document_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()