"""Embedding provider contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Interface shared by production and offline embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable model identifier stored in index metadata."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of values in each vector."""

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Embed document chunks."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed one retrieval query."""

    def validate_vectors(
        self,
        vectors: Sequence[Sequence[float]],
        expected_count: int,
    ) -> None:
        """Check vector count and dimensional consistency."""
        if len(vectors) != expected_count:
            raise ValueError(
                f"expected {expected_count} vectors, got {len(vectors)}"
            )
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"expected {self.dimensions} dimensions, "
                    f"got {len(vector)}"
                )
