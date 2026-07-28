"""Integration tests for validation, hashing, and native extraction."""
from pathlib import Path
import shutil
import pytest

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.domain.models import ExtractionMethod,SourceFormat
from app.ingestion.file_validator import FileValidationConfig,FileValidationError,discover_local_files,validate_local_file
from app.ingestion.pdf_extractor import PDFExtractionConfig,count_meaningful_characters,extract_pdf,extract_pdf_pages,normalize_pdf_text,page_needs_ocr
from app.utils.hashing import create_document_id,sha256_file

SAMPLE=Path("/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200989.pdf")

@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    if not SAMPLE.exists(): pytest.skip("SYN-200989.pdf not mounted")
    target=tmp_path/"SYN-200989.pdf"; shutil.copy2(SAMPLE,target); return target

def test_hash_is_deterministic(sample_pdf):
    assert sha256_file(sample_pdf)==sha256_file(sample_pdf)
    assert len(sha256_file(sample_pdf))==64
    assert create_document_id(sample_pdf).startswith("doc-")

def test_validator_accepts_pdf(sample_pdf):
    source=validate_local_file(sample_pdf)
    assert source.source_format==SourceFormat.PDF

def test_validator_rejects_fake_pdf(tmp_path):
    fake=tmp_path/"fake.pdf"; fake.write_text("not pdf")
    with pytest.raises(FileValidationError,match="signature"): validate_local_file(fake)

def test_discovery_supports_pdf_png_jpg_jpeg(sample_pdf,tmp_path):
    (tmp_path/"a.png").write_bytes(b"\x89PNG\r\n\x1a\nX")
    (tmp_path/"b.jpg").write_bytes(b"\xff\xd8\xffX")
    (tmp_path/"c.jpeg").write_bytes(b"\xff\xd8\xffX")
    sources=discover_local_files(tmp_path,config=FileValidationConfig(verify_signature=True))
    assert {s.path.suffix for s in sources}=={".pdf",".png",".jpg",".jpeg"}

def test_extract_uploaded_pdf(sample_pdf):
    result=extract_pdf(sample_pdf,config=PDFExtractionConfig(minimum_meaningful_characters=40))
    assert result.total_pages==1
    page=result.pages[0]; diag=result.diagnostics[0]
    assert page.page_number==1
    assert page.extraction_method==ExtractionMethod.NATIVE_PDF
    assert page.metadata.patient_id=="SYN-200989"
    for expected in ["SYNTHETIC MEDICAL RECORD","SYN-200989","Metformin","Shellfish","HbA1c: 6.8%","Knee X-ray"]:
        assert expected in page.text
    assert diag.needs_ocr is False
    assert diag.character_count==len(page.text)

def test_pages_wrapper(sample_pdf):
    pages=extract_pdf_pages(sample_pdf)
    assert len(pages)==1 and pages[0].filename=="SYN-200989.pdf"

def test_sparse_text_detection():
    assert page_needs_ocr("Title",minimum_meaningful_characters=20)
    assert not page_needs_ocr("Patient ID: SYN-001 with useful content",minimum_meaningful_characters=20)

def test_meaningful_count():
    assert count_meaningful_characters(" A-1 / B! ")==3

def test_normalization():
    assert normalize_pdf_text("Header  \r\n\r\n\r\nBody   \r\n")=="Header\n\nBody"

def test_non_pdf_rejected(tmp_path):
    image=tmp_path/"note.png"; image.write_bytes(b"\x89PNG\r\n\x1a\nX")
    with pytest.raises(ValueError,match="PDF files only"): extract_pdf(validate_local_file(image))