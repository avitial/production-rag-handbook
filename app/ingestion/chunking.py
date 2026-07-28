"""Local-source ingestion plus section-aware chunking.

Supported local inputs:
- Native or scanned PDF
- JPEG/JPG/PNG images
- Handwritten-note images, using the same OCR path with handwritten metadata

PDF extraction uses PyMuPDF. OCR uses Tesseract through pytesseract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Callable

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook/app"))

from domain.models import (
    DocumentChunk,
    DocumentMetadata,
    DocumentPage,
    ExtractionMethod,
    LocalSource,
    SourceFormat,
)

SUPPORTED_SUFFIXES = {".pdf", ".jpeg", ".jpg", ".png"}

_KNOWN_HEADINGS = {
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
    "findings",
    "impression",
    "diagnosis",
    "diagnoses",
    "laboratory results",
    "follow-up",
    "follow up",
    "discharge instructions",
}

_PATIENT_ID_PATTERN = re.compile(
    r"Patient\s*ID\s*:\s*([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkingConfig:
    max_characters: int = 800
    overlap_characters: int = 150
    minimum_chunk_characters: int = 1

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("max_characters must be greater than zero")
        if self.overlap_characters < 0:
            raise ValueError("overlap_characters must not be negative")
        if self.overlap_characters >= self.max_characters:
            raise ValueError("overlap_characters must be smaller than max_characters")
        if self.minimum_chunk_characters <= 0:
            raise ValueError("minimum_chunk_characters must be greater than zero")
        if self.minimum_chunk_characters > self.max_characters:
            raise ValueError("minimum_chunk_characters cannot exceed max_characters")


@dataclass(frozen=True)
class TextSection:
    name: str | None
    text: str
    start_offset: int
    end_offset: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_document_id(path: Path) -> str:
    return f"doc-{sha256_file(path)[:16]}"


def infer_source_format(path: Path, *, handwritten: bool = False) -> SourceFormat:
    if handwritten:
        return SourceFormat.HANDWRITTEN_NOTE
    suffix = path.suffix.lower()
    return {
        ".pdf": SourceFormat.PDF,
        ".jpeg": SourceFormat.JPEG,
        ".jpg": SourceFormat.JPG,
        ".png": SourceFormat.PNG,
    }.get(suffix, SourceFormat.UNKNOWN)


def discover_local_sources(
    path: str | Path,
    *,
    recursive: bool = True,
    handwritten_names: bool = True,
) -> list[LocalSource]:
    """Discover supported local files from one file or directory.

    Filenames containing 'handwritten' or 'handwriting' are marked as
    handwritten-note images when handwritten_names is enabled.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"source path does not exist: {source_path}")

    candidates = [source_path] if source_path.is_file() else (
        source_path.rglob("*") if recursive else source_path.glob("*")
    )

    sources: list[LocalSource] = []
    for candidate in sorted(candidates):
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        lowered = candidate.stem.lower()
        is_handwritten = handwritten_names and (
            "handwritten" in lowered or "handwriting" in lowered
        )
        sources.append(
            LocalSource(
                path=candidate,
                source_format=infer_source_format(
                    candidate,
                    handwritten=is_handwritten,
                ),
                is_handwritten=is_handwritten,
            )
        )
    return sources


def _extract_patient_id(text: str) -> str | None:
    match = _PATIENT_ID_PATTERN.search(text)
    return match.group(1) if match else None


def _metadata_for_text(
    text: str,
    *,
    base: DocumentMetadata | None,
    handwritten: bool,
) -> DocumentMetadata:
    original = base or DocumentMetadata()
    data = original.model_dump()
    if not data.get("patient_id"):
        data["patient_id"] = _extract_patient_id(text)
    data["is_handwritten"] = handwritten or bool(data.get("is_handwritten"))
    return DocumentMetadata(**data)


def _ocr_pil_image(image, *, language: str = "eng") -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required for image OCR. "
            "Install it and the external Tesseract executable."
        ) from exc
    return pytesseract.image_to_string(image, lang=language)


def extract_pdf_pages(
    source: LocalSource,
    *,
    metadata: DocumentMetadata | None = None,
    ocr_language: str = "eng",
    native_text_min_characters: int = 40,
    ocr_function: Callable | None = None,
) -> list[DocumentPage]:
    """Extract page text from a native or scanned PDF."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF ingestion") from exc

    if source.path.suffix.lower() != ".pdf":
        raise ValueError("extract_pdf_pages requires a PDF source")

    document_id = create_document_id(source.path)
    source_hash = sha256_file(source.path)
    pages: list[DocumentPage] = []
    ocr = ocr_function or _ocr_pil_image

    with fitz.open(source.path) as pdf:
        for page_index, page in enumerate(pdf):
            native_text = page.get_text("text") or ""
            meaningful = "".join(ch for ch in native_text if ch.isalnum())

            if len(meaningful) >= native_text_min_characters:
                text = native_text
                method = ExtractionMethod.NATIVE_PDF
            else:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                from PIL import Image
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                text = ocr(image, language=ocr_language)
                method = ExtractionMethod.OCR_PDF

            page_metadata = _metadata_for_text(
                text,
                base=metadata,
                handwritten=False,
            )
            pages.append(
                DocumentPage(
                    document_id=document_id,
                    filename=source.path.name,
                    source_path=str(source.path),
                    source_format=SourceFormat.PDF,
                    page_number=page_index + 1,
                    text=text,
                    extraction_method=method,
                    metadata=page_metadata,
                    source_hash=source_hash,
                )
            )
    return pages


def extract_image_page(
    source: LocalSource,
    *,
    metadata: DocumentMetadata | None = None,
    ocr_language: str = "eng",
    ocr_function: Callable | None = None,
) -> DocumentPage:
    """OCR one JPEG/JPG/PNG image as page 1."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image ingestion") from exc

    if source.path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
        raise ValueError("extract_image_page requires JPEG, JPG, or PNG")

    ocr = ocr_function or _ocr_pil_image
    with Image.open(source.path) as image:
        processed = ImageOps.exif_transpose(image).convert("L")
        processed = ImageOps.autocontrast(processed)
        text = ocr(processed, language=ocr_language)

    handwritten = source.is_handwritten or (
        source.source_format == SourceFormat.HANDWRITTEN_NOTE
    )
    method = (
        ExtractionMethod.OCR_HANDWRITTEN
        if handwritten
        else ExtractionMethod.OCR_IMAGE
    )
    return DocumentPage(
        document_id=create_document_id(source.path),
        filename=source.path.name,
        source_path=str(source.path),
        source_format=(
            SourceFormat.HANDWRITTEN_NOTE
            if handwritten
            else infer_source_format(source.path)
        ),
        page_number=1,
        text=text,
        extraction_method=method,
        metadata=_metadata_for_text(
            text,
            base=metadata,
            handwritten=handwritten,
        ),
        source_hash=sha256_file(source.path),
    )


def ingest_local_source(
    source: LocalSource,
    *,
    metadata: DocumentMetadata | None = None,
    ocr_language: str = "eng",
    native_text_min_characters: int = 40,
    ocr_function: Callable | None = None,
) -> list[DocumentPage]:
    """Load one supported local source into page models."""
    suffix = source.path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(
            source,
            metadata=metadata,
            ocr_language=ocr_language,
            native_text_min_characters=native_text_min_characters,
            ocr_function=ocr_function,
        )
    if suffix in {".jpeg", ".jpg", ".png"}:
        return [
            extract_image_page(
                source,
                metadata=metadata,
                ocr_language=ocr_language,
                ocr_function=ocr_function,
            )
        ]
    raise ValueError(f"unsupported source format: {source.path.suffix}")


def ingest_local_path(
    path: str | Path,
    *,
    recursive: bool = True,
    metadata: DocumentMetadata | None = None,
    ocr_language: str = "eng",
    native_text_min_characters: int = 40,
    ocr_function: Callable | None = None,
) -> list[DocumentPage]:
    """Discover and ingest all supported local files under a path."""
    pages: list[DocumentPage] = []
    for source in discover_local_sources(path, recursive=recursive):
        pages.extend(
            ingest_local_source(
                source,
                metadata=metadata,
                ocr_language=ocr_language,
                native_text_min_characters=native_text_min_characters,
                ocr_function=ocr_function,
            )
        )
    return pages


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def is_heading(line: str) -> bool:
    candidate = line.strip().rstrip(":").strip()
    if not candidate or len(candidate) > 80:
        return False

    if candidate.lower() in _KNOWN_HEADINGS:
        return True

    # Generic headings must truly be uppercase. This prevents ordinary values
    # such as "Metformin" or "Shellfish" from becoming section headings.
    has_letter = any(character.isalpha() for character in candidate)
    return (
        has_letter
        and candidate == candidate.upper()
        and bool(re.fullmatch(r"[A-Z0-9 /&()_-]{3,}", candidate))
    )


def detect_sections(text: str) -> list[TextSection]:
    normalized = normalize_newlines(text)
    if not normalized.strip():
        return []

    lines = normalized.splitlines(keepends=True)
    sections: list[TextSection] = []
    current_name: str | None = None
    current_start = 0
    current_parts: list[str] = []
    cursor = 0

    def flush(end_offset: int) -> None:
        nonlocal current_parts
        section_text = "".join(current_parts)
        if section_text.strip():
            sections.append(
                TextSection(
                    name=current_name,
                    text=section_text,
                    start_offset=current_start,
                    end_offset=end_offset,
                )
            )
        current_parts = []

    for line in lines:
        line_start = cursor
        cursor += len(line)
        if is_heading(line):
            flush(line_start)
            current_name = line.strip().rstrip(":").strip()
            current_start = line_start
            current_parts = [line]
        else:
            if not current_parts:
                current_start = line_start
            current_parts.append(line)

    flush(len(normalized))
    return sections


def _preferred_split_position(text: str, start: int, hard_end: int) -> int:
    if hard_end >= len(text):
        return len(text)
    floor = start + max(1, (hard_end - start) // 2)
    segment = text[floor:hard_end]
    options = [
        segment.rfind("\n\n"),
        segment.rfind("\n"),
        segment.rfind(". "),
        segment.rfind("; "),
        segment.rfind(" "),
    ]
    best = max(options)
    if best < 0:
        return hard_end
    delimiter = segment[best:best + 2]
    return floor + best + (2 if delimiter in {"\n\n", ". ", "; "} else 1)


def split_text_with_overlap(
    text: str,
    *,
    base_offset: int,
    config: ChunkingConfig,
) -> list[tuple[str, int, int]]:
    if not text.strip():
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        hard_end = min(start + config.max_characters, len(text))
        end = _preferred_split_position(text, start, hard_end)
        if end <= start:
            end = hard_end

        raw = text[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw) - len(raw.rstrip())
        clean_start = start + left
        clean_end = end - right
        clean = text[clean_start:clean_end]

        if clean and len(clean) >= config.minimum_chunk_characters:
            chunks.append(
                (clean, base_offset + clean_start, base_offset + clean_end)
            )
        if end >= len(text):
            break
        next_start = max(0, end - config.overlap_characters)
        start = end if next_start <= start else next_start
    return chunks


def create_chunk_id(
    *,
    document_id: str,
    page_number: int,
    start_offset: int,
    end_offset: int,
    text: str,
) -> str:
    payload = f"{document_id}|{page_number}|{start_offset}|{end_offset}|{text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}-p{page_number}-c{digest}"


def chunk_page(
    page: DocumentPage,
    *,
    config: ChunkingConfig | None = None,
    section_aware: bool = True,
) -> list[DocumentChunk]:
    config = config or ChunkingConfig()
    text = normalize_newlines(page.text)
    if not text.strip():
        return []

    sections = detect_sections(text) if section_aware else [
        TextSection(None, text, 0, len(text))
    ]

    base_metadata = page.metadata.model_dump(mode="json")
    base_metadata.update(
        {
            "document_id": page.document_id,
            "filename": page.filename,
            "source_path": page.source_path,
            "source_format": page.source_format.value,
            "page_number": page.page_number,
            "extraction_method": page.extraction_method.value,
            "source_hash": page.source_hash,
        }
    )

    chunks: list[DocumentChunk] = []
    for section in sections:
        for chunk_text, start, end in split_text_with_overlap(
            section.text,
            base_offset=section.start_offset,
            config=config,
        ):
            metadata = dict(base_metadata)
            if section.name:
                metadata["section"] = section.name
            chunks.append(
                DocumentChunk(
                    chunk_id=create_chunk_id(
                        document_id=page.document_id,
                        page_number=page.page_number,
                        start_offset=start,
                        end_offset=end,
                        text=chunk_text,
                    ),
                    document_id=page.document_id,
                    filename=page.filename,
                    source_path=page.source_path,
                    source_format=page.source_format,
                    page_number=page.page_number,
                    section=section.name,
                    text=chunk_text,
                    start_offset=start,
                    end_offset=end,
                    metadata=metadata,
                )
            )
    return chunks


def chunk_pages(
    pages: Sequence[DocumentPage] | Iterable[DocumentPage],
    *,
    config: ChunkingConfig | None = None,
    section_aware: bool = True,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for page in pages:
        chunks.extend(
            chunk_page(
                page,
                config=config,
                section_aware=section_aware,
            )
        )
    return chunks


def ingest_and_chunk_local_path(
    path: str | Path,
    *,
    recursive: bool = True,
    metadata: DocumentMetadata | None = None,
    config: ChunkingConfig | None = None,
    section_aware: bool = True,
    ocr_language: str = "eng",
    native_text_min_characters: int = 40,
    ocr_function: Callable | None = None,
) -> list[DocumentChunk]:
    """Convenience function: local path -> extracted pages -> chunks."""
    pages = ingest_local_path(
        path,
        recursive=recursive,
        metadata=metadata,
        ocr_language=ocr_language,
        native_text_min_characters=native_text_min_characters,
        ocr_function=ocr_function,
    )
    return chunk_pages(
        pages,
        config=config,
        section_aware=section_aware,
    )