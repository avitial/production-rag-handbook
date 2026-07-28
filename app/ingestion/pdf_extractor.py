"""Native PDF extraction with page-level provenance and OCR-candidate flags.

Pseudo-code:
validate PDF -> hash bytes -> open with PyMuPDF -> for each page:
extract native text -> normalize -> count useful characters -> flag sparse page
-> extract Patient ID -> create DocumentPage -> record diagnostics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.domain.models import DocumentMetadata,DocumentPage,ExtractionMethod,LocalSource,SourceFormat
from app.ingestion.file_validator import validate_local_file
from app.utils.hashing import create_document_id,sha256_file

PATIENT_ID_RE=re.compile(r"\bPatient\s*ID\s*:\s*([A-Za-z0-9_-]+)",re.I)

@dataclass(frozen=True)
class PDFExtractionConfig:
    minimum_meaningful_characters: int=40
    sort_text: bool=True
    preserve_whitespace: bool=True
    def __post_init__(self):
        if self.minimum_meaningful_characters<0: raise ValueError("minimum_meaningful_characters must not be negative")

@dataclass(frozen=True)
class PDFPageDiagnostics:
    page_number: int
    character_count: int
    meaningful_character_count: int
    needs_ocr: bool
    extraction_duration_ms: float

@dataclass(frozen=True)
class PDFExtractionResult:
    document_id: str
    filename: str
    source_path: str
    source_hash: str
    total_pages: int
    pages: list[DocumentPage]=field(default_factory=list)
    diagnostics: list[PDFPageDiagnostics]=field(default_factory=list)
    pdf_metadata: dict[str,Any]=field(default_factory=dict)

def normalize_pdf_text(text: str, *, preserve_whitespace: bool=True) -> str:
    """Normalize line endings, trailing spaces, and repeated blank lines."""
    normalized=text.replace("\r\n","\n").replace("\r","\n")
    lines=[line.rstrip() if preserve_whitespace else " ".join(line.split()) for line in normalized.split("\n")]
    output=[]; previous_blank=False
    for line in lines:
        blank=not line.strip()
        if blank and previous_blank: continue
        output.append(line); previous_blank=blank
    return "\n".join(output).strip()

def count_meaningful_characters(text: str) -> int:
    """Count letters and digits only."""
    return sum(ch.isalnum() for ch in text)

def page_needs_ocr(text: str, *, minimum_meaningful_characters: int=40) -> bool:
    """Heuristic: sparse native text may indicate a scan."""
    if minimum_meaningful_characters<0: raise ValueError("minimum_meaningful_characters must not be negative")
    return count_meaningful_characters(text)<minimum_meaningful_characters

def extract_patient_id(text: str) -> str|None:
    """Extract values such as SYN-200989."""
    match=PATIENT_ID_RE.search(text)
    return match.group(1) if match else None

def _merge_metadata(text: str, metadata: DocumentMetadata|None) -> DocumentMetadata:
    base=metadata or DocumentMetadata()
    data=base.model_dump()
    if not data.get("patient_id"): data["patient_id"]=extract_patient_id(text)
    return DocumentMetadata(**data)

def extract_pdf(source: str|Path|LocalSource, *, config: PDFExtractionConfig|None=None, metadata: DocumentMetadata|None=None) -> PDFExtractionResult:
    """Extract native text from every page of one validated PDF."""
    cfg=config or PDFExtractionConfig()
    local=source if isinstance(source,LocalSource) else validate_local_file(source)
    if local.path.suffix.lower()!=".pdf":
        raise ValueError("extract_pdf accepts PDF files only")
    try: import fitz
    except ImportError as exc: raise RuntimeError("Install PyMuPDF with: pip install pymupdf") from exc
    source_hash=sha256_file(local.path); document_id=create_document_id(local.path)
    pages=[]; diagnostics=[]
    try: pdf=fitz.open(local.path)
    except Exception as exc: raise RuntimeError(f"unable to open PDF: {local.path}") from exc
    with pdf:
        pdf_metadata=dict(pdf.metadata or {})
        for index in range(pdf.page_count):
            page=pdf.load_page(index); started=perf_counter()
            raw=page.get_text("text",sort=cfg.sort_text) or ""
            text=normalize_pdf_text(raw,preserve_whitespace=cfg.preserve_whitespace)
            duration=(perf_counter()-started)*1000
            meaningful=count_meaningful_characters(text)
            needs_ocr=meaningful<cfg.minimum_meaningful_characters
            page_number=index+1
            pages.append(DocumentPage(
                document_id=document_id,filename=local.path.name,source_path=str(local.path),
                source_format=SourceFormat.PDF,page_number=page_number,text=text,
                extraction_method=ExtractionMethod.NATIVE_PDF,metadata=_merge_metadata(text,metadata),
                source_hash=source_hash,character_count=len(text),
                meaningful_character_count=meaningful,extraction_duration_ms=duration))
            diagnostics.append(PDFPageDiagnostics(page_number,len(text),meaningful,needs_ocr,duration))
    return PDFExtractionResult(document_id,local.path.name,str(local.path),source_hash,len(pages),pages,diagnostics,pdf_metadata)

def extract_pdf_pages(source: str|Path|LocalSource, *, config: PDFExtractionConfig|None=None, metadata: DocumentMetadata|None=None) -> list[DocumentPage]:
    """Convenience wrapper returning pages only."""
    return extract_pdf(source,config=config,metadata=metadata).pages