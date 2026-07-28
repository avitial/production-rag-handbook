"""Integration tests for OCR and preprocessing."""

from pathlib import Path
import shutil

import pytest
from PIL import Image

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.domain.models import ExtractionMethod
from app.ingestion.image_preprocessor import (
    ImagePreprocessingConfig,
    preprocess_image_file,
)
from app.ingestion.ocr import (
    average_tesseract_confidence,
    ingest_ocr_source,
    ocr_image,
    ocr_pdf,
)


HANDWRITTEN_SAMPLE = Path(
    "/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200849_handwritten.png"
)


class FakeOutput:
    DICT = "dict"


class FakeTesseract:
    """Deterministic OCR substitute for unit-like integration paths."""

    Output = FakeOutput

    @staticmethod
    def image_to_string(image, *, lang, config):
        assert image.width > 0
        return (
            "Patient ID: SYN-200849\n"
            "Preferred Language: Vietnamese\n"
            "Clinical Notes (SOAP)\n"
            "Medications\nCurrent: Metformin\n"
            "Allergies\nLatex\n"
        )

    @staticmethod
    def image_to_data(
        image,
        *,
        lang,
        config,
        output_type,
    ):
        return {
            "conf": ["-1", "95.0", "90.0", "85.0"],
            "text": ["", "Patient", "ID", "SYN-200849"],
        }


@pytest.fixture()
def handwritten_copy(tmp_path: Path) -> Path:
    if not HANDWRITTEN_SAMPLE.exists():
        pytest.skip("handwritten sample is not mounted")

    target = tmp_path / "SYN-200849_handwritten.png"
    shutil.copy2(HANDWRITTEN_SAMPLE, target)
    return target


def test_preprocessor_upscales_and_grayscales(
    handwritten_copy: Path,
) -> None:
    result = preprocess_image_file(
        handwritten_copy,
        config=ImagePreprocessingConfig(
            minimum_width=1600,
            threshold=None,
        ),
    )

    assert result.original_size == (1000, 1400)
    assert result.processed_size[0] == 1600
    assert result.image.mode == "L"
    assert "grayscale" in result.operations


def test_average_confidence_ignores_negative_rows() -> None:
    confidence = average_tesseract_confidence(
        {"conf": ["-1", "80", "90", "bad"]}
    )

    assert confidence == pytest.approx(85.0)


def test_handwritten_image_ocr_with_fake_engine(
    handwritten_copy: Path,
) -> None:
    result = ocr_image(
        handwritten_copy,
        engine=FakeTesseract,
    )

    assert result.extraction_method == ExtractionMethod.OCR_HANDWRITTEN
    assert result.confidence == pytest.approx(90.0)
    assert "SYN-200849" in result.text
    assert result.metadata["patient_id"] == "SYN-200849"
    assert result.metadata["is_handwritten"] is True


def test_dispatches_png_to_image_ocr(
    handwritten_copy: Path,
) -> None:
    [result] = ingest_ocr_source(
        handwritten_copy,
        engine=FakeTesseract,
    )

    assert result.page_number == 1
    assert result.extraction_method == ExtractionMethod.OCR_HANDWRITTEN


def test_native_pdf_skips_ocr(
    tmp_path: Path,
) -> None:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF is unavailable")

    pdf_path = tmp_path / "native.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Patient ID: SYN-NATIVE-001\nClinical Notes (SOAP)\n"
        "This page contains enough native text for extraction.",
    )
    document.save(pdf_path)
    document.close()

    [result] = ocr_pdf(
        pdf_path,
        engine=FakeTesseract,
        minimum_native_characters=20,
    )

    assert result.extraction_method == ExtractionMethod.NATIVE_PDF
    assert result.metadata["ocr_fallback_used"] is False
    assert "SYN-NATIVE-001" in result.text


def test_scanned_pdf_uses_ocr_fallback(
    handwritten_copy: Path,
    tmp_path: Path,
) -> None:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF is unavailable")

    pdf_path = tmp_path / "scanned.pdf"
    document = fitz.open()
    page = document.new_page(
        width=612,
        height=792,
    )
    page.insert_image(
        page.rect,
        filename=str(handwritten_copy),
    )
    document.save(pdf_path)
    document.close()

    [result] = ocr_pdf(
        pdf_path,
        engine=FakeTesseract,
        minimum_native_characters=40,
    )

    assert result.extraction_method == ExtractionMethod.OCR_PDF
    assert result.metadata["ocr_fallback_used"] is True
    assert result.metadata["patient_id"] == "SYN-200849"


@pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="Tesseract executable is not installed.",
)
def test_real_tesseract_extracts_key_sample_content(
    handwritten_copy: Path,
) -> None:
    result = ocr_image(
        handwritten_copy,
        preprocessing_config=ImagePreprocessingConfig(
            minimum_width=1600,
            threshold=None,
        ),
    )

    # The supplied image is digitally typeset but treated as the handwriting
    # sample for this project. Check robust keywords rather than exact layout.
    lowered = result.text.lower()
    assert "synthetic medical record" in lowered
    assert "syn-200849" in lowered
    assert "metformin" in lowered