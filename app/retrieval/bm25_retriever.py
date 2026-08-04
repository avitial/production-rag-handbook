"""Dependency-free BM25 keyword retrieval over indexed document chunks.

BM25 is useful for exact identifiers, medication names, dates, dosages,
abbreviations, laboratory values, and uncommon clinical terms.

The implementation follows the standard Okapi BM25 structure:

    score(query, document) =
        sum over query terms of:
        IDF(term) *
        ((frequency * (k1 + 1)) /
         (frequency + k1 * (1 - b + b * document_length / average_length)))

Pseudo-code:

    load chunk documents and metadata from the vector collection
    tokenize every document
    calculate document frequencies and average document length
    for a query:
        tokenize query
        apply metadata filter before scoring
        calculate BM25 score for every eligible document
        sort descending
        return typed RetrievedPassage objects
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.retrieval.filter_builder import build_chroma_where
from app.retrieval.models import (
    RetrievedPassage,
    RetrievalDiagnostics,
    SearchFilters,
    VectorSearchRequest,
    VectorSearchResponse,
)
from app.retrieval.tokenizer import MedicalTokenizer
from app.storage.chroma_store import ChromaStore


@dataclass(frozen=True)
class BM25Config:
    """Okapi BM25 tuning parameters."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between zero and one")


@dataclass(frozen=True)
class BM25Document:
    """Internal indexed chunk representation."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    tokens: tuple[str, ...]
    frequencies: Counter[str]


def _matches_where(
    metadata: dict[str, Any],
    where: dict[str, Any] | None,
) -> bool:
    """Evaluate the Chroma filter subset used by Day 7."""
    if not where:
        return True

    if "$and" in where:
        return all(
            _matches_where(metadata, item)
            for item in where["$and"]
        )

    for key, expected in where.items():
        actual = metadata.get(key)

        if isinstance(expected, dict):
            for operator, target in expected.items():
                if actual is None:
                    return False
                if operator == "$gte" and not actual >= target:
                    return False
                if operator == "$lte" and not actual <= target:
                    return False
                if operator == "$gt" and not actual > target:
                    return False
                if operator == "$lt" and not actual < target:
                    return False
                if operator == "$ne" and not actual != target:
                    return False
        elif actual != expected:
            return False

    return True


class BM25Retriever:
    """Build and query an in-memory BM25 index from stored chunks."""

    def __init__(
        self,
        store: ChromaStore,
        *,
        tokenizer: MedicalTokenizer | None = None,
        config: BM25Config | None = None,
    ) -> None:
        self.store = store
        self.tokenizer = tokenizer or MedicalTokenizer()
        self.config = config or BM25Config()
        self._documents: list[BM25Document] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_document_length = 0.0

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def rebuild(self) -> int:
        """Rebuild BM25 from all chunks currently in the vector collection.

        Pseudo-code:
            collection.get(ids, documents, metadatas)
            tokenize every document
            store term frequencies
            count each term once per document for document frequency
            calculate average document length
        """
        raw = self.store.collection.get(
            include=["documents", "metadatas"]
        )
        ids = list(raw.get("ids", []))
        documents = list(raw.get("documents", []))
        metadatas = list(raw.get("metadatas", []))

        indexed: list[BM25Document] = []
        document_frequency: Counter[str] = Counter()
        total_length = 0

        for chunk_id, text, metadata in zip(
            ids,
            documents,
            metadatas,
        ):
            tokens = tuple(self.tokenizer.tokenize(str(text)))
            frequencies = Counter(tokens)
            total_length += len(tokens)
            document_frequency.update(frequencies.keys())

            indexed.append(
                BM25Document(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    metadata=dict(metadata or {}),
                    tokens=tokens,
                    frequencies=frequencies,
                )
            )

        self._documents = indexed
        self._document_frequency = document_frequency
        self._average_document_length = (
            total_length / len(indexed)
            if indexed
            else 0.0
        )
        return len(indexed)

    def _idf(self, term: str) -> float:
        """Calculate positive Robertson/Sparck Jones IDF."""
        document_count = len(self._documents)
        frequency = self._document_frequency.get(term, 0)

        return math.log(
            1
            + (
                document_count - frequency + 0.5
            )
            / (frequency + 0.5)
        )

    def _score(
        self,
        query_tokens: list[str],
        document: BM25Document,
    ) -> float:
        """Calculate one query-document BM25 score."""
        if not document.tokens or self._average_document_length == 0:
            return 0.0

        score = 0.0
        length = len(document.tokens)

        for term in query_tokens:
            frequency = document.frequencies.get(term, 0)
            if frequency == 0:
                continue

            numerator = frequency * (self.config.k1 + 1)
            denominator = frequency + self.config.k1 * (
                1
                - self.config.b
                + self.config.b
                * length
                / self._average_document_length
            )
            score += self._idf(term) * numerator / denominator

        return score

    def search(
        self,
        request: VectorSearchRequest,
    ) -> VectorSearchResponse:
        """Return BM25-ranked passages using the shared retrieval models."""
        if not self._documents:
            self.rebuild()

        where = build_chroma_where(request.filters)
        query_tokens = self.tokenizer.tokenize(request.query)
        scored: list[tuple[float, BM25Document]] = []

        for document in self._documents:
            if not _matches_where(document.metadata, where):
                continue

            score = self._score(query_tokens, document)

            # Retain only documents with at least one keyword match.
            if score > 0:
                scored.append((score, document))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].chunk_id,
            )
        )

        results: list[RetrievedPassage] = []

        for rank, (score, document) in enumerate(
            scored[:request.top_k],
            start=1,
        ):
            metadata = document.metadata
            results.append(
                RetrievedPassage(
                    chunk_id=document.chunk_id,
                    document_id=str(
                        metadata.get("document_id", "")
                    ),
                    filename=str(metadata.get("filename", "")),
                    source_path=str(
                        metadata.get("source_path", "")
                    ),
                    source_format=str(
                        metadata.get("source_format", "")
                    ),
                    page_number=int(
                        metadata.get("page_number", 1)
                    ),
                    section=(
                        str(metadata["section"])
                        if metadata.get("section")
                        else None
                    ),
                    patient_id=(
                        str(metadata["patient_id"])
                        if metadata.get("patient_id")
                        else None
                    ),
                    text=document.text,
                    rank=rank,
                    distance=0.0,
                    similarity=float(score),
                    metadata={
                        **metadata,
                        "retrieval_method": "bm25",
                        "bm25_score": float(score),
                    },
                )
            )

        return VectorSearchResponse(
            query=request.query,
            filters=request.filters,
            results=tuple(results),
            diagnostics=RetrievalDiagnostics(
                collection_count=len(self._documents),
                requested_top_k=request.top_k,
                returned_count=len(results),
                where_filter=where,
                embedding_model="bm25-keyword-index",
            ),
        )
