"""Synchronized local ingestion pipeline.

Supported sources:
- Native PDFs
- Scanned PDFs with OCR fallback
- JPEG/JPG/PNG images
- Handwritten-note images
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Callable

from PIL import Image, ImageOps

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.domain.models import (
    DocumentChunk,
    DocumentPage,
    ExtractionMethod,
    IngestionReport,
    SourceFormat,
)
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry
from app.utils.hashing import (
    create_chunk_id,
    create_document_id,
    sha256_file,
)


SUPPORTED_SUFFIXES = {".pdf", ".jpeg", ".jpg", ".png"}

PATIENT_ID_PATTERN = re.compile(
    r"Patient\s*ID\s*:\s*([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)

KNOWN_HEADINGS = {
    "patient demographics",
    "clinical notes",
    "clinical notes (soap)",
    "medication",
    "medications",
    "allergy",
    "allergies",
    "problem",
    "problems",
    "vital signs",
    "procedures & results",
    "assessment",
    "plan",
    "follow-up",
    "follow up",
    "diagnosis",
    "diagnoses",
}


@dataclass(frozen=True)
class IngestionConfig:
    max_characters: int = 800
    overlap_characters: int = 150
    minimum_native_characters: int = 40
    recursive: bool = True
    ocr_language: str = "eng"

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("max_characters must be greater than zero")
        if not 0 <= self.overlap_characters < self.max_characters:
            raise ValueError(
                "overlap_characters must be non-negative and "
                "smaller than max_characters"
            )
        if self.minimum_native_characters < 0:
            raise ValueError(
                "minimum_native_characters must not be negative"
            )

    @property
    def signature(self) -> str:
        payload = (
            f"max={self.max_characters}|"
            f"overlap={self.overlap_characters}|"
            f"native={self.minimum_native_characters}|"
            f"language={self.ocr_language}"
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


def discover_sources(
    path: str | Path,
    *,
    recursive: bool = True,
) -> list[Path]:
    """Discover supported local source files."""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"source path does not exist: {source}"
        )

    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"unsupported source format: {source.suffix}"
            )
        return [source]

    iterator = source.rglob("*") if recursive else source.glob("*")
    return [
        candidate
        for candidate in sorted(iterator)
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def extract_patient_id(text: str) -> str | None:
    match = PATIENT_ID_PATTERN.search(text)
    return match.group(1) if match else None


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Apply a conservative OCR preprocessing baseline."""
    processed = ImageOps.exif_transpose(image).convert("L")
    if processed.width < 1600:
        scale = 1600 / processed.width
        processed = processed.resize(
            (
                1600,
                int(round(processed.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    return ImageOps.autocontrast(processed)


def _run_ocr(
    image: Image.Image,
    *,
    language: str,
    ocr_function: Callable | None,
) -> str:
    """Use injected OCR for tests or pytesseract in normal operation."""
    if ocr_function is not None:
        return str(ocr_function(image, language=language))

    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required for OCR."
        ) from exc

    return pytesseract.image_to_string(
        image,
        lang=language,
        config="--oem 3 --psm 6",
    )


def extract_pages(
    path: Path,
    *,
    config: IngestionConfig,
    ocr_function: Callable | None = None,
) -> list[DocumentPage]:
    """Extract a supported local source into page models."""
    document_id = create_document_id(path)
    source_hash = sha256_file(path)
    suffix = path.suffix.lower()
    handwritten = any(
        marker in path.stem.lower()
        for marker in (
            "handwritten",
            "handwriting",
            "hand-written",
        )
    )

    if suffix == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required for PDF ingestion."
            ) from exc

        output: list[DocumentPage] = []

        with fitz.open(path) as document:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                native_text = (
                    page.get_text("text", sort=True) or ""
                ).strip()
                meaningful = sum(
                    character.isalnum()
                    for character in native_text
                )

                if meaningful >= config.minimum_native_characters:
                    text = native_text
                    method = ExtractionMethod.NATIVE_PDF
                else:
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False,
                    )
                    image = Image.frombytes(
                        "RGB",
                        (pixmap.width, pixmap.height),
                        pixmap.samples,
                    )
                    text = _run_ocr(
                        _preprocess_image(image),
                        language=config.ocr_language,
                        ocr_function=ocr_function,
                    ).strip()
                    method = ExtractionMethod.OCR_PDF

                output.append(
                    DocumentPage(
                        document_id=document_id,
                        filename=path.name,
                        source_path=str(path),
                        source_format=SourceFormat.PDF,
                        page_number=page_index + 1,
                        text=text,
                        extraction_method=method,
                        source_hash=source_hash,
                        metadata={
                            "patient_id": extract_patient_id(text),
                            "is_handwritten": False,
                        },
                    )
                )

        return output

    with Image.open(path) as image:
        text = _run_ocr(
            _preprocess_image(image),
            language=config.ocr_language,
            ocr_function=ocr_function,
        ).strip()

    if handwritten:
        source_format = SourceFormat.HANDWRITTEN_NOTE
        method = ExtractionMethod.OCR_HANDWRITTEN
    else:
        source_format = {
            ".jpeg": SourceFormat.JPEG,
            ".jpg": SourceFormat.JPG,
            ".png": SourceFormat.PNG,
        }[suffix]
        method = ExtractionMethod.OCR_IMAGE

    return [
        DocumentPage(
            document_id=document_id,
            filename=path.name,
            source_path=str(path),
            source_format=source_format,
            page_number=1,
            text=text,
            extraction_method=method,
            source_hash=source_hash,
            metadata={
                "patient_id": extract_patient_id(text),
                "is_handwritten": handwritten,
            },
        )
    ]


def _is_heading(line: str) -> bool:
    candidate = line.strip().rstrip(":").strip()
    if not candidate:
        return False
    if candidate.lower() in KNOWN_HEADINGS:
        return True
    has_letter = any(char.isalpha() for char in candidate)
    return (
        has_letter
        and candidate == candidate.upper()
        and len(candidate) <= 80
    )


def _detect_sections(
    text: str,
) -> list[tuple[str | None, str, int]]:
    """Return section name, section text, and page-relative offset."""
    lines = text.splitlines(keepends=True)
    output: list[tuple[str | None, str, int]] = []

    current_name: str | None = None
    current_parts: list[str] = []
    current_start = 0
    cursor = 0

    def flush() -> None:
        nonlocal current_parts
        section_text = "".join(current_parts)
        if section_text.strip():
            output.append(
                (current_name, section_text, current_start)
            )
        current_parts = []

    for line in lines:
        line_start = cursor
        cursor += len(line)

        if _is_heading(line):
            flush()
            current_name = line.strip().rstrip(":").strip()
            current_start = line_start
            current_parts = [line]
        else:
            if not current_parts:
                current_start = line_start
            current_parts.append(line)

    flush()
    return output


def _preferred_end(text: str, start: int, hard_end: int) -> int:
    """Prefer a natural boundary before the hard size limit."""
    if hard_end >= len(text):
        return len(text)

    floor = start + max(1, (hard_end - start) // 2)
    segment = text[floor:hard_end]
    positions = [
        segment.rfind("\n\n"),
        segment.rfind("\n"),
        segment.rfind(". "),
        segment.rfind("; "),
        segment.rfind(" "),
    ]
    best = max(positions)
    if best < 0:
        return hard_end

    delimiter = segment[best:best + 2]
    extra = 2 if delimiter in {"\n\n", ". ", "; "} else 1
    return floor + best + extra


def chunk_pages(
    pages: list[DocumentPage],
    *,
    config: IngestionConfig,
) -> list[DocumentChunk]:
    """Create deterministic, page-bounded, overlapping chunks."""
    chunks: list[DocumentChunk] = []

    for page in pages:
        for section_name, section_text, base_offset in _detect_sections(
            page.text
        ):
            start = 0

            while start < len(section_text):
                hard_end = min(
                    start + config.max_characters,
                    len(section_text),
                )
                end = _preferred_end(
                    section_text,
                    start,
                    hard_end,
                )

                raw = section_text[start:end]
                left_trim = len(raw) - len(raw.lstrip())
                right_trim = len(raw) - len(raw.rstrip())
                clean_start = start + left_trim
                clean_end = end - right_trim
                clean_text = section_text[
                    clean_start:clean_end
                ]

                if clean_text:
                    absolute_start = base_offset + clean_start
                    absolute_end = base_offset + clean_end
                    metadata = dict(page.metadata)
                    metadata.update(
                        {
                            "source_hash": page.source_hash,
                            "extraction_method": (
                                page.extraction_method.value
                            ),
                            "section": section_name or "",
                        }
                    )

                    chunks.append(
                        DocumentChunk(
                            chunk_id=create_chunk_id(
                                page.document_id,
                                page.page_number,
                                absolute_start,
                                absolute_end,
                                clean_text,
                            ),
                            document_id=page.document_id,
                            filename=page.filename,
                            source_path=page.source_path,
                            source_format=page.source_format,
                            page_number=page.page_number,
                            section=section_name,
                            text=clean_text,
                            start_offset=absolute_start,
                            end_offset=absolute_end,
                            metadata=metadata,
                        )
                    )

                if end >= len(section_text):
                    break

                next_start = max(
                    end - config.overlap_characters,
                    0,
                )
                start = end if next_start <= start else next_start

    return chunks


class LocalIngestionPipeline:
    """Coordinate discovery, extraction, chunking, storage, and registry."""

    def __init__(
        self,
        *,
        chroma_store: ChromaStore,
        registry: DocumentRegistry,
        config: IngestionConfig | None = None,
        ocr_function: Callable | None = None,
    ) -> None:
        self.chroma_store = chroma_store
        self.registry = registry
        self.config = config or IngestionConfig()
        self.ocr_function = ocr_function

    def ingest(self, path: str | Path) -> IngestionReport:
        """Ingest one file or directory and return complete counters."""
        sources = discover_sources(
            path,
            recursive=self.config.recursive,
        )

        # Every accumulator is initialized before the processing loop.
        processed_files = 0
        skipped_files = 0
        failed_files = 0
        page_count = 0
        chunk_count = 0
        indexed_chunk_count = 0
        document_ids: list[str] = []
        warnings: list[str] = []

        for source in sources:
            document_id = create_document_id(source)
            source_hash = sha256_file(source)
            document_ids.append(document_id)

            if self.registry.should_skip(
                document_id=document_id,
                embedding_model=(
                    self.chroma_store.embedding_provider.model_name
                ),
                chunking_signature=self.config.signature,
            ):
                skipped_files += 1
                continue

            try:
                pages = extract_pages(
                    source,
                    config=self.config,
                    ocr_function=self.ocr_function,
                )
                chunks = chunk_pages(
                    pages,
                    config=self.config,
                )

                if not chunks:
                    raise ValueError(
                        "extraction produced no indexable chunks"
                    )

                self.chroma_store.delete_document(document_id)
                indexed = self.chroma_store.upsert_chunks(chunks)

                self.registry.mark_completed(
                    document_id=document_id,
                    source_hash=source_hash,
                    source_path=str(source),
                    filename=source.name,
                    embedding_model=(
                        self.chroma_store.embedding_provider.model_name
                    ),
                    chunking_signature=self.config.signature,
                    chunk_count=len(chunks),
                )

                processed_files += 1
                page_count += len(pages)
                chunk_count += len(chunks)
                indexed_chunk_count += indexed

            except Exception as exc:
                failed_files += 1
                message = f"{source.name}: {exc}"
                warnings.append(message)

                self.registry.mark_failed(
                    document_id=document_id,
                    source_hash=source_hash,
                    source_path=str(source),
                    filename=source.name,
                    embedding_model=(
                        self.chroma_store.embedding_provider.model_name
                    ),
                    chunking_signature=self.config.signature,
                    error_message=str(exc),
                )

        return IngestionReport(
            discovered_files=len(sources),
            processed_files=processed_files,
            skipped_files=skipped_files,
            failed_files=failed_files,
            page_count=page_count,
            chunk_count=chunk_count,
            indexed_chunk_count=indexed_chunk_count,
            document_ids=tuple(document_ids),
            warnings=tuple(warnings),
        )
