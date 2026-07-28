"""OCR orchestration for PDFs, images, and handwritten notes.

Supported local sources:

- Scanned PDF pages
- JPEG/JPG/PNG images
- Handwritten-note images

OCR engine:
- Tesseract via pytesseract

Pseudo-code for an image:

    validate image format
    preprocess image
    run Tesseract text extraction
    request word-level confidence data
    average non-negative confidences
    extract deterministic metadata
    return OCRResult

Pseudo-code for a PDF:

    open PDF with PyMuPDF
    for each page:
        try native text
        if native text is sufficient and force_ocr is false:
            return native result
        otherwise:
            render page to image
            preprocess
            OCR
            preserve page number
    return results
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

from PIL import Image

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.domain.models import (
    ExtractionMethod,
    OCRResult,
)
from app.ingestion.image_preprocessor import (
    ImagePreprocessingConfig,
    preprocess_image,
    preprocess_image_file,
)
from app.ingestion.metadata_extractor import extract_metadata


SUPPORTED_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def _load_pytesseract():
    """Import pytesseract lazily so tests can inject a fake OCR engine."""
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required. Install it and the Tesseract executable."
        ) from exc
    return pytesseract


def average_tesseract_confidence(data: dict[str, list]) -> float | None:
    """Average valid word confidences from pytesseract output.

    Tesseract uses -1 for rows that do not represent recognized words.
    """
    values: list[float] = []

    for raw_value in data.get("conf", []):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            values.append(value)

    if not values:
        return None

    return sum(values) / len(values)


def _run_tesseract(
    image: Image.Image,
    *,
    language: str,
    config: str,
    engine=None,
) -> tuple[str, float | None]:
    """Run Tesseract text and confidence extraction."""
    pytesseract = engine or _load_pytesseract()

    text = pytesseract.image_to_string(
        image,
        lang=language,
        config=config,
    )

    output_type = getattr(
        getattr(pytesseract, "Output", object),
        "DICT",
        "dict",
    )
    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=output_type,
    )

    return text, average_tesseract_confidence(data)


def ocr_image(
    path: str | Path,
    *,
    language: str = "eng",
    tesseract_config: str = "--oem 3 --psm 6",
    preprocessing_config: ImagePreprocessingConfig | None = None,
    is_handwritten: bool | None = None,
    engine=None,
) -> OCRResult:
    """OCR a JPEG, JPG, or PNG source.

    Handwritten status can be supplied explicitly. When omitted, filenames
    containing 'handwritten' or 'handwriting' are treated as handwritten.
    """
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"image does not exist: {source}")
    if source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("OCR image must be .jpeg, .jpg, or .png")

    inferred_handwritten = any(
        marker in source.stem.lower()
        for marker in ("handwritten", "handwriting", "hand-written")
    )
    handwritten = (
        inferred_handwritten
        if is_handwritten is None
        else is_handwritten
    )

    started = perf_counter()
    preprocessing = preprocess_image_file(
        source,
        config=preprocessing_config,
    )
    text, confidence = _run_tesseract(
        preprocessing.image,
        language=language,
        config=tesseract_config,
        engine=engine,
    )
    duration_ms = (perf_counter() - started) * 1000

    metadata_result = extract_metadata(
        text,
        is_handwritten=handwritten,
    )

    return OCRResult(
        text=text.replace("\r\n", "\n").replace("\r", "\n").strip(),
        confidence=confidence,
        extraction_method=(
            ExtractionMethod.OCR_HANDWRITTEN
            if handwritten
            else ExtractionMethod.OCR_IMAGE
        ),
        language=language,
        duration_ms=duration_ms,
        source_path=str(source),
        page_number=1,
        metadata={
            "patient_id": metadata_result.metadata.patient_id,
            "document_type": (
                metadata_result.metadata.document_type.value
                if metadata_result.metadata.document_type
                else None
            ),
            "preferred_language": (
                metadata_result.metadata.preferred_language
            ),
            "is_handwritten": handwritten,
            "metadata_confidence": metadata_result.confidence,
            "metadata_warnings": metadata_result.warnings,
            "preprocessing_operations": preprocessing.operations,
            "processed_size": preprocessing.processed_size,
        },
    )


def _meaningful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def ocr_pdf(
    path: str | Path,
    *,
    language: str = "eng",
    tesseract_config: str = "--oem 3 --psm 6",
    preprocessing_config: ImagePreprocessingConfig | None = None,
    minimum_native_characters: int = 40,
    force_ocr: bool = False,
    engine=None,
) -> list[OCRResult]:
    """Extract a PDF with native-text preference and OCR fallback.

    Native pages are returned with ``NATIVE_PDF``. Sparse pages are rendered
    and passed to Tesseract with ``OCR_PDF``.
    """
    source = Path(path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"PDF does not exist: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError("ocr_pdf requires a .pdf file")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF OCR fallback."
        ) from exc

    results: list[OCRResult] = []

    with fitz.open(source) as document:
        for page_index in range(document.page_count):
            page_number = page_index + 1
            page = document.load_page(page_index)
            native_text = page.get_text("text", sort=True) or ""

            if (
                not force_ocr
                and _meaningful_character_count(native_text)
                >= minimum_native_characters
            ):
                metadata_result = extract_metadata(native_text)
                results.append(
                    OCRResult(
                        text=native_text.strip(),
                        confidence=None,
                        extraction_method=ExtractionMethod.NATIVE_PDF,
                        language=language,
                        duration_ms=0,
                        source_path=str(source),
                        page_number=page_number,
                        metadata={
                            "patient_id": (
                                metadata_result.metadata.patient_id
                            ),
                            "document_type": (
                                metadata_result.metadata.document_type.value
                                if metadata_result.metadata.document_type
                                else None
                            ),
                            "metadata_confidence": (
                                metadata_result.confidence
                            ),
                            "ocr_fallback_used": False,
                        },
                    )
                )
                continue

            started = perf_counter()
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
            )
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            preprocessing = preprocess_image(
                image,
                config=preprocessing_config,
            )
            text, confidence = _run_tesseract(
                preprocessing.image,
                language=language,
                config=tesseract_config,
                engine=engine,
            )
            duration_ms = (perf_counter() - started) * 1000
            metadata_result = extract_metadata(text)

            results.append(
                OCRResult(
                    text=text.strip(),
                    confidence=confidence,
                    extraction_method=ExtractionMethod.OCR_PDF,
                    language=language,
                    duration_ms=duration_ms,
                    source_path=str(source),
                    page_number=page_number,
                    metadata={
                        "patient_id": (
                            metadata_result.metadata.patient_id
                        ),
                        "document_type": (
                            metadata_result.metadata.document_type.value
                            if metadata_result.metadata.document_type
                            else None
                        ),
                        "metadata_confidence": (
                            metadata_result.confidence
                        ),
                        "ocr_fallback_used": True,
                        "preprocessing_operations": (
                            preprocessing.operations
                        ),
                    },
                )
            )

    return results


def ingest_ocr_source(
    path: str | Path,
    **kwargs,
) -> list[OCRResult]:
    """Dispatch a local PDF or image to the appropriate OCR workflow."""
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()

    if suffix == ".pdf":
        return ocr_pdf(source, **kwargs)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return [ocr_image(source, **kwargs)]

    raise ValueError(
        "supported OCR sources are PDF, JPEG, JPG, and PNG"
    )