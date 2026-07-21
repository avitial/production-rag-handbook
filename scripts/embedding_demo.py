"""Day 1 embedding and semantic-search demonstration."""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PASSAGES = [
    "The patient has a history of hypertension.",
    "The patient was prescribed lisinopril 10 mg once daily.",
    "The patient has no known drug allergies.",
    "Lisinopril is known to not cure hypertension.",
    "A follow-up appointment with cardiology was scheduled in two weeks.",
    "The patient's blood pressure was 148 over 92.",
]

@dataclass(frozen=True)
class SearchResult:
    passage: str
    score: float
    rank: int

def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def encode_texts(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        raise ValueError("texts must contain at least one item")
    embeddings = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)

def cosine_scores(passage_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    if passage_embeddings.ndim != 2:
        raise ValueError("passage_embeddings must be a 2D array")
    query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if passage_embeddings.shape[1] != query_vector.shape[0]:
        raise ValueError("Passage and query embeddings must have the same dimensions")
    return passage_embeddings @ query_vector

def semantic_search(model: SentenceTransformer, query: str, passages: Sequence[str], top_k: int | None = None) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not passages:
        raise ValueError("passages must contain at least one item")
    passage_embeddings = encode_texts(model, passages)
    query_embedding = encode_texts(model, [query])[0]
    scores = cosine_scores(passage_embeddings, query_embedding)
    ranked_indices = np.argsort(scores)[::-1]
    limit = len(passages) if top_k is None else max(0, min(top_k, len(passages)))
    return [SearchResult(passages[i], float(scores[i]), rank) for rank, i in enumerate(ranked_indices[:limit], 1)]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank sample medical passages using semantic similarity.")
    parser.add_argument("--query", default="What medicine treats the patient's high blood pressure?")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=len(DEFAULT_PASSAGES))
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    try:
        results = semantic_search(load_embedding_model(args.model), args.query, DEFAULT_PASSAGES, args.top_k)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Embedding demo failed: {exc}", file=sys.stderr)
        return 1
    print(f"\nQuery: {args.query}\n")
    for result in results:
        print(f"{result.rank}. score={result.score:.4f} | {result.passage}")
    print("\nSimilarity ranks evidence candidates; it is not answer confidence.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
