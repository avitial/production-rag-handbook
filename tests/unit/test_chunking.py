"""Tests for local PDF/image ingestion and section-aware chunking."""
from pathlib import Path
import shutil

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

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
from ingestion.chunking import (
    ChunkingConfig,
    chunk_page,
    detect_sections,
    discover_local_sources,
    ingest_and_chunk_local_path,
    ingest_local_source,
    split_text_with_overlap,
)


def fake_ocr(_image, *, language="eng") -> str:
    assert language == "eng"
    return (
        "Patient ID: SYN-IMAGE-001\n"
        "HANDWRITTEN NOTE\n"
        "Patient reports mild headache.\n"
        "PLAN\n"
        "Follow up in two weeks."
    )


def make_page(text: str) -> DocumentPage:
    return DocumentPage(
        document_id="doc-test",
        filename="test.pdf",
        source_path="/tmp/test.pdf",
        source_format=SourceFormat.PDF,
        page_number=1,
        text=text,
        extraction_method=ExtractionMethod.NATIVE_PDF,
        metadata=DocumentMetadata(patient_id="SYN-001"),
    )


def test_local_source_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="does not exist"):
        LocalSource(
            path=tmp_path / "missing.pdf",
            source_format=SourceFormat.PDF,
        )


def test_discovery_finds_pdf_and_images(tmp_path: Path) -> None:
    for name in ["a.pdf", "b.jpeg", "c.jpg", "d.png", "ignore.txt"]:
        (tmp_path / name).write_bytes(b"test")

    sources = discover_local_sources(tmp_path)
    assert {source.path.suffix for source in sources} == {
        ".pdf", ".jpeg", ".jpg", ".png"
    }


def test_discovery_marks_handwritten_filename(tmp_path: Path) -> None:
    path = tmp_path / "handwritten_note_01.png"
    path.write_bytes(b"image")
    [source] = discover_local_sources(tmp_path)
    assert source.is_handwritten is True
    assert source.source_format == SourceFormat.HANDWRITTEN_NOTE


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg"])
def test_image_ingestion_uses_ocr_and_extracts_patient_id(
    tmp_path: Path,
    suffix: str,
) -> None:
    image_path = tmp_path / f"note{suffix}"
    image = Image.new("RGB", (300, 120), "white")
    ImageDraw.Draw(image).text((10, 10), "Synthetic note", fill="black")
    image.save(image_path)

    source = LocalSource(
        path=image_path,
        source_format={
            ".png": SourceFormat.PNG,
            ".jpg": SourceFormat.JPG,
            ".jpeg": SourceFormat.JPEG,
        }[suffix],
    )
    [page] = ingest_local_source(source, ocr_function=fake_ocr)

    assert page.page_number == 1
    assert page.extraction_method == ExtractionMethod.OCR_IMAGE
    assert page.metadata.patient_id == "SYN-IMAGE-001"
    assert "mild headache" in page.text


def test_handwritten_image_uses_handwritten_extraction_method(
    tmp_path: Path,
) -> None:
    path = tmp_path / "handwritten_note.png"
    Image.new("RGB", (100, 100), "white").save(path)
    [source] = discover_local_sources(tmp_path)

    [page] = ingest_local_source(source, ocr_function=fake_ocr)
    assert page.extraction_method == ExtractionMethod.OCR_HANDWRITTEN
    assert page.metadata.is_handwritten is True


def test_detect_sections_keeps_heading_with_content() -> None:
    text = (
        "Patient ID: SYN-001\n"
        "Medications\n"
        "Metformin\n"
        "Allergies\n"
        "Shellfish\n"
    )
    sections = detect_sections(text)
    assert [section.name for section in sections] == [
        None, "Medications", "Allergies"
    ]
    assert sections[1].text.startswith("Medications")


def test_split_text_applies_overlap_and_maximum() -> None:
    text = "0123456789ABCDEFGHIJ"
    chunks = split_text_with_overlap(
        text,
        base_offset=0,
        config=ChunkingConfig(max_characters=10, overlap_characters=3),
    )
    assert chunks[0] == ("0123456789", 0, 10)
    assert chunks[1][0].startswith("789")
    assert all(len(item[0]) <= 10 for item in chunks)


def test_chunk_offsets_reconstruct_source_text() -> None:
    page = make_page(
        "PLAN\nContinue therapy and return in six months."
    )
    chunks = chunk_page(
        page,
        config=ChunkingConfig(max_characters=25, overlap_characters=5),
    )
    for chunk in chunks:
        assert page.text[chunk.start_offset:chunk.end_offset] == chunk.text


def test_chunk_model_rejects_inconsistent_offsets() -> None:
    with pytest.raises(ValidationError, match="offset span"):
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            filename="x.pdf",
            source_path="/tmp/x.pdf",
            source_format=SourceFormat.PDF,
            page_number=1,
            text="abc",
            start_offset=0,
            end_offset=10,
        )


def test_ingest_and_chunk_uploaded_sample_pdf() -> None:
    sample = Path("/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200989.pdf")
    if not sample.exists():
        pytest.skip("uploaded sample PDF is not mounted")

    chunks = ingest_and_chunk_local_path(
        sample,
        config=ChunkingConfig(
            max_characters=300,
            overlap_characters=40,
        ),
    )

    assert chunks
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "SYN-200989" in combined
    assert "Metformin" in combined
    assert "Shellfish" in combined
    assert "HbA1c" in combined
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(chunk.metadata["patient_id"] == "SYN-200989" for chunk in chunks)