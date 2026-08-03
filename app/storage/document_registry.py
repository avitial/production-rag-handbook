"""SQLite ingestion registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class RegistryRecord:
    document_id: str
    source_hash: str
    source_path: str
    filename: str
    status: str
    embedding_model: str
    chunking_signature: str
    chunk_count: int
    updated_at: str
    error_message: str | None


class DocumentRegistry:
    """Track completed/failed documents and indexing configuration."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_hash TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunking_signature TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )

    def get(self, document_id: str) -> RegistryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return RegistryRecord(**dict(row)) if row else None

    def should_skip(
        self,
        *,
        document_id: str,
        embedding_model: str,
        chunking_signature: str,
    ) -> bool:
        record = self.get(document_id)
        return bool(
            record
            and record.status == "completed"
            and record.embedding_model == embedding_model
            and record.chunking_signature == chunking_signature
        )

    def mark_completed(
        self,
        *,
        document_id: str,
        source_hash: str,
        source_path: str,
        filename: str,
        embedding_model: str,
        chunking_signature: str,
        chunk_count: int,
    ) -> None:
        self._upsert(
            document_id=document_id,
            source_hash=source_hash,
            source_path=source_path,
            filename=filename,
            status="completed",
            embedding_model=embedding_model,
            chunking_signature=chunking_signature,
            chunk_count=chunk_count,
            error_message=None,
        )

    def mark_failed(
        self,
        *,
        document_id: str,
        source_hash: str,
        source_path: str,
        filename: str,
        embedding_model: str,
        chunking_signature: str,
        error_message: str,
    ) -> None:
        self._upsert(
            document_id=document_id,
            source_hash=source_hash,
            source_path=source_path,
            filename=filename,
            status="failed",
            embedding_model=embedding_model,
            chunking_signature=chunking_signature,
            chunk_count=0,
            error_message=error_message,
        )

    def _upsert(self, **values) -> None:
        values["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_hash, source_path, filename,
                    status, embedding_model, chunking_signature,
                    chunk_count, updated_at, error_message
                ) VALUES (
                    :document_id, :source_hash, :source_path, :filename,
                    :status, :embedding_model, :chunking_signature,
                    :chunk_count, :updated_at, :error_message
                )
                ON CONFLICT(document_id) DO UPDATE SET
                    source_hash=excluded.source_hash,
                    source_path=excluded.source_path,
                    filename=excluded.filename,
                    status=excluded.status,
                    embedding_model=excluded.embedding_model,
                    chunking_signature=excluded.chunking_signature,
                    chunk_count=excluded.chunk_count,
                    updated_at=excluded.updated_at,
                    error_message=excluded.error_message
                """,
                values,
            )

    def count(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM documents"
                ).fetchone()[0]
            )
