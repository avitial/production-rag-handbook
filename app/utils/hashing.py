"""Content hashing and deterministic IDs."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the entire file into memory.

    Pseudo-code:
        open source in binary mode
        read one block at a time
        update SHA-256 with every block
        return hexadecimal digest
    """
    if block_size <= 0:
        raise ValueError("block_size must be greater than zero")

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file():
        raise ValueError(f"not a regular file: {source}")

    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def create_document_id(path: str | Path) -> str:
    """Create a stable content-based document ID."""
    return f"doc-{sha256_file(path)[:16]}"


def create_chunk_id(
    document_id: str,
    page_number: int,
    start_offset: int,
    end_offset: int,
    text: str,
) -> str:
    """Create a stable chunk ID from provenance and exact chunk content."""
    payload = (
        f"{document_id}|{page_number}|{start_offset}|{end_offset}|{text}"
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}-p{page_number}-c{digest}"
