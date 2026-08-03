"""Sentence Transformers provider plus a deterministic offline fallback."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import math
import re
from typing import Any

from app.embeddings.base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Wrap a real Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        model: Any | None = None,
    ) -> None:
        """Load a model or accept an injected compatible model.

        Pseudo-code:
            validate configuration
            import SentenceTransformer only when required
            load selected model
            read output dimension
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Use --embedding-backend hash or install the package."
                ) from exc
            model = SentenceTransformer(model_name)

        self._model = model
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize_embeddings
        self._dimensions = int(
            model.get_sentence_embedding_dimension()
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts and convert the model result to plain lists."""
        cleaned = [text.strip() for text in texts]
        if any(not value for value in cleaned):
            raise ValueError("embedding text must not be blank")
        if not cleaned:
            return []

        vectors = self._model.encode(
            cleaned,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = [
            [float(value) for value in vector]
            for vector in vectors
        ]
        self.validate_vectors(result, len(cleaned))
        return result

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("query must not be blank")
        return self._encode([query])[0]


class DeterministicHashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free embeddings for tests and offline demonstrations.

    This is not a replacement for a trained semantic model. It uses stable
    hashed token and bigram features so the complete pipeline can run without
    downloading model weights.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return f"deterministic-hash-{self.dimensions}d-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
        bigrams = [
            f"{left}_{right}"
            for left, right in zip(words, words[1:])
        ]
        return words + bigrams

    def _embed(self, text: str) -> list[float]:
        tokens = self._tokens(text)
        vector = [0.0] * self.dimensions

        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        if not any(vector):
            vector[0] = 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector]

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        result = [self._embed(text) for text in texts]
        self.validate_vectors(result, len(texts))
        return result

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("query must not be blank")
        return self._embed(query)


def create_embedding_provider(
    backend: str = "auto",
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> EmbeddingProvider:
    """Create a provider with a graceful offline fallback.

    auto:
        use Sentence Transformers when installed and loadable;
        otherwise use deterministic hash embeddings.
    """
    normalized = backend.strip().lower()

    if normalized == "hash":
        return DeterministicHashEmbeddingProvider()

    if normalized == "sentence-transformer":
        return SentenceTransformerEmbeddingProvider(model_name)

    if normalized == "auto":
        try:
            return SentenceTransformerEmbeddingProvider(model_name)
        except Exception:
            return DeterministicHashEmbeddingProvider()

    raise ValueError(
        "embedding backend must be auto, sentence-transformer, or hash"
    )
