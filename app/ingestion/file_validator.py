"""Validate and discover local PDF/image/handwritten-note files.

Pseudo-code: resolve path -> verify existence -> infer extension -> enforce size -> verify magic bytes -> infer handwritten marker -> return LocalSource.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.domain.models import LocalSource, SourceFormat

SUPPORTED={".pdf":SourceFormat.PDF,".jpeg":SourceFormat.JPEG,".jpg":SourceFormat.JPG,".png":SourceFormat.PNG}
PDF_SIG=b"%PDF-"; PNG_SIG=b"\x89PNG\r\n\x1a\n"; JPEG_SIG=b"\xff\xd8\xff"

class FileValidationError(ValueError): pass

@dataclass(frozen=True)
class FileValidationConfig:
    maximum_file_size_mb: int=100
    verify_signature: bool=True
    handwritten_markers: tuple[str,...]=("handwritten","handwriting","hand-written")
    def __post_init__(self):
        if self.maximum_file_size_mb<=0: raise ValueError("maximum_file_size_mb must be greater than zero")

def infer_source_format(path: str|Path) -> SourceFormat:
    return SUPPORTED.get(Path(path).suffix.lower(), SourceFormat.UNKNOWN)

def signature_matches(path: str|Path, fmt: SourceFormat) -> bool:
    """Basic extension/content check; not malware scanning."""
    with Path(path).open("rb") as handle: sig=handle.read(16)
    if fmt==SourceFormat.PDF: return sig.startswith(PDF_SIG)
    if fmt==SourceFormat.PNG: return sig.startswith(PNG_SIG)
    if fmt in {SourceFormat.JPG,SourceFormat.JPEG}: return sig.startswith(JPEG_SIG)
    return False

def validate_local_file(path: str|Path, *, config: FileValidationConfig|None=None, is_handwritten: bool|None=None) -> LocalSource:
    """Validate one local source."""
    cfg=config or FileValidationConfig()
    p=Path(path).expanduser().resolve()
    if not p.exists(): raise FileValidationError(f"source does not exist: {p}")
    if not p.is_file(): raise FileValidationError(f"source is not a file: {p}")
    fmt=infer_source_format(p)
    if fmt==SourceFormat.UNKNOWN: raise FileValidationError(f"unsupported extension: {p.suffix}")
    size=p.stat().st_size
    if size==0: raise FileValidationError(f"source file is empty: {p}")
    if size>cfg.maximum_file_size_mb*1024*1024: raise FileValidationError(f"source exceeds size limit: {p}")
    if cfg.verify_signature and not signature_matches(p,fmt):
        raise FileValidationError(f"file signature does not match extension: {p}")
    handwritten=(any(m in p.stem.lower() for m in cfg.handwritten_markers) if is_handwritten is None else is_handwritten)
    output_fmt=SourceFormat.HANDWRITTEN_NOTE if handwritten and fmt!=SourceFormat.PDF else fmt
    return LocalSource(path=p,source_format=output_fmt,is_handwritten=handwritten)

def discover_local_files(path: str|Path, *, recursive: bool=True, config: FileValidationConfig|None=None, skip_invalid: bool=False) -> list[LocalSource]:
    """Discover supported local files from a file or directory."""
    root=Path(path).expanduser().resolve()
    if not root.exists(): raise FileValidationError(f"source path does not exist: {root}")
    if root.is_file(): return [validate_local_file(root,config=config)]
    iterator=root.rglob("*") if recursive else root.glob("*")
    found=[]
    for candidate in sorted(iterator):
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED: continue
        try: found.append(validate_local_file(candidate,config=config))
        except FileValidationError:
            if not skip_invalid: raise
    return found