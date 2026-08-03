"""A tiny persistent Chroma-compatible fallback.

It implements only the subset used by this Day 5 project:
- get_or_create_collection
- upsert
- delete(where=...)
- count
- query
- get

It persists collection records as JSON. When real ``chromadb`` is installed,
``ChromaStore`` uses it instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math


def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate the Chroma filter subset used by Day 6."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(metadata, item) for item in where["$and"])
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            for operator, target in expected.items():
                if actual is None: return False
                if operator == "$gte" and not actual >= target: return False
                if operator == "$lte" and not actual <= target: return False
                if operator == "$gt" and not actual > target: return False
                if operator == "$lt" and not actual < target: return False
                if operator == "$ne" and not actual != target: return False
        elif actual != expected:
            return False
    return True

def cosine_distance(left: list[float], right: list[float]) -> float:
    """Return 1 - cosine similarity."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - (dot / (left_norm * right_norm))


class LocalPersistentCollection:
    def __init__(
        self,
        path: Path,
        name: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        self.path = path / f"{name}.json"
        self.name = name
        self.metadata = metadata or {}
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.metadata = payload.get("metadata", self.metadata)
        self._records = payload.get("records", {})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "records": self._records,
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def upsert(
        self,
        *,
        ids,
        documents,
        embeddings,
        metadatas,
    ) -> None:
        for item_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            self._records[str(item_id)] = {
                "document": document,
                "embedding": list(embedding),
                "metadata": dict(metadata),
            }
        self._save()

    def delete(self, *, where: dict[str, Any]) -> None:
        matching = [
            item_id
            for item_id, record in self._records.items()
            if _matches_where(record["metadata"], where)
        ]
        for item_id in matching:
            del self._records[item_id]
        self._save()

    def count(self) -> int:
        return len(self._records)

    def get(self, *, include=None) -> dict[str, Any]:
        ids = list(self._records)
        result: dict[str, Any] = {"ids": ids}
        include = include or []
        if "documents" in include:
            result["documents"] = [
                self._records[item_id]["document"]
                for item_id in ids
            ]
        if "metadatas" in include:
            result["metadatas"] = [
                self._records[item_id]["metadata"]
                for item_id in ids
            ]
        if "embeddings" in include:
            result["embeddings"] = [
                self._records[item_id]["embedding"]
                for item_id in ids
            ]
        return result

    def query(
        self,
        *,
        query_embeddings,
        n_results,
        where=None,
        include=None,
    ) -> dict[str, Any]:
        include = include or []
        all_ids: list[list[str]] = []
        all_documents: list[list[str]] = []
        all_metadatas: list[list[dict[str, Any]]] = []
        all_distances: list[list[float]] = []

        for query_embedding in query_embeddings:
            candidates = []
            for item_id, record in self._records.items():
                metadata = record["metadata"]
                if not _matches_where(metadata, where):
                    continue
                distance = cosine_distance(
                    list(query_embedding),
                    record["embedding"],
                )
                candidates.append((distance, item_id, record))

            candidates.sort(key=lambda item: item[0])
            chosen = candidates[:n_results]
            all_ids.append([item[1] for item in chosen])
            all_documents.append([
                item[2]["document"] for item in chosen
            ])
            all_metadatas.append([
                item[2]["metadata"] for item in chosen
            ])
            all_distances.append([item[0] for item in chosen])

        result: dict[str, Any] = {"ids": all_ids}
        if "documents" in include:
            result["documents"] = all_documents
        if "metadatas" in include:
            result["metadatas"] = all_metadatas
        if "distances" in include:
            result["distances"] = all_distances
        return result


class LocalPersistentClient:
    """Factory for local persistent collections."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> LocalPersistentCollection:
        return LocalPersistentCollection(
            self.path,
            name,
            metadata,
        )
