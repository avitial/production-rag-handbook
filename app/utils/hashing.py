"""Streaming hashing for stable document identity.

Pseudo-code:
1. Open file as binary.
2. Read fixed-size blocks.
3. Update SHA-256 for each block.
4. Return the hexadecimal digest.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
DEFAULT_BLOCK_SIZE=1024*1024

def sha256_stream(stream: BinaryIO, *, block_size: int=DEFAULT_BLOCK_SIZE) -> str:
    """Hash an open binary stream without loading the whole file."""
    if block_size<=0: raise ValueError("block_size must be greater than zero")
    digest=sha256()
    while True:
        block=stream.read(block_size)
        if not block: break
        digest.update(block)
    return digest.hexdigest()

def sha256_file(path: str|Path, *, block_size: int=DEFAULT_BLOCK_SIZE) -> str:
    """Validate and hash a local file."""
    p=Path(path).expanduser().resolve()
    if not p.exists(): raise FileNotFoundError(f"file does not exist: {p}")
    if not p.is_file(): raise ValueError(f"path is not a file: {p}")
    with p.open("rb") as handle:
        return sha256_stream(handle, block_size=block_size)

def short_hash(digest: str, *, length: int=16) -> str:
    """Create a readable digest prefix for IDs."""
    value=digest.strip().lower()
    if not value: raise ValueError("digest must not be empty")
    if length<=0 or length>len(value): raise ValueError("invalid length")
    return value[:length]

def create_document_id(path: str|Path) -> str:
    """Create a deterministic content-based ID."""
    return f"doc-{short_hash(sha256_file(path))}"