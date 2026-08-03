"""ChromaDB adapter with a persistent local compatibility fallback."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.domain.models import DocumentChunk
from app.embeddings.base import EmbeddingProvider
from app.storage.local_vector_client import LocalPersistentClient


class ChromaStore:
    """Store chunks in real ChromaDB or the bundled local fallback."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "medical_documents",
        persistence_directory: str | Path = "./chroma_data",
        client: Any | None = None,
        backend: str = "auto",
    ) -> None:
        """Initialize storage.

        backend values:
        - auto: real Chroma when installed, otherwise local fallback
        - chroma: require real chromadb
        - local: force bundled fallback
        """
        if not collection_name.strip():
            raise ValueError("collection_name must not be blank")

        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self.persistence_directory = Path(persistence_directory)
        self.persistence_directory.mkdir(parents=True, exist_ok=True)

        if client is None:
            client, resolved_backend = self._create_client(backend)
        else:
            resolved_backend = "injected"

        self.backend_name = resolved_backend
        self.client = client
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "embedding_model": embedding_provider.model_name,
                "distance": "cosine",
            },
        )

    def _create_client(self, backend: str):
        normalized = backend.strip().lower()

        if normalized not in {"auto", "chroma", "local"}:
            raise ValueError(
                "storage backend must be auto, chroma, or local"
            )

        if normalized in {"auto", "chroma"}:
            try:
                import chromadb
                return (
                    chromadb.PersistentClient(
                        path=str(self.persistence_directory)
                    ),
                    "chroma",
                )
            except ImportError:
                if normalized == "chroma":
                    raise RuntimeError(
                        "chromadb is not installed. "
                        "Use --storage-backend local or install chromadb."
                    )

        return (
            LocalPersistentClient(self.persistence_directory),
            "local",
        )

    @staticmethod
    def _clean_metadata(
        chunk: DocumentChunk,
        model_name: str,
    ) -> dict[str, Any]:
        """Create scalar-only storage metadata."""
        values = dict(chunk.metadata)
        values.update(
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "source_path": chunk.source_path,
                "source_format": chunk.source_format.value,
                "page_number": chunk.page_number,
                "section": chunk.section or "",
                "embedding_model": model_name,
            }
        )

        cleaned: dict[str, Any] = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        batch_size: int = 64,
    ) -> int:
        """Embed and persist chunks in bounded batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        indexed = 0
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start:start + batch_size])
            if not batch:
                continue

            texts = [chunk.text for chunk in batch]
            embeddings = self.embedding_provider.embed_documents(
                texts
            )
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    self._clean_metadata(
                        chunk,
                        self.embedding_provider.model_name,
                    )
                    for chunk in batch
                ],
            )
            indexed += len(batch)

        return indexed

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(
            where={"document_id": document_id}
        )

    def count(self) -> int:
        return int(self.collection.count())

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a Day 5 semantic-search smoke query."""
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        embedding = self.embedding_provider.embed_query(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, max(1, self.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)
