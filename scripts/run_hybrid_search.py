"""End-to-end Day 7 ingestion and hybrid retrieval demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings.sentence_transformer import (
    create_embedding_provider,
)
from app.ingestion.pipeline import (
    IngestionConfig,
    LocalIngestionPipeline,
)
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import (
    SearchFilters,
    VectorSearchRequest,
)
from app.retrieval.vector_retriever import VectorRetriever
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry


QUESTIONS = [
    "What medication was prescribed at discharge?",
    "What allergies are documented?",
    "When is the follow-up appointment?",
    "What was the primary diagnosis?",
    "Which patient had a cardiology referral?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        default="data/samples",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence-transformer", "hash"],
        default="auto",
    )
    parser.add_argument(
        "--storage-backend",
        choices=["auto", "chroma", "local"],
        default="auto",
    )
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_data",
    )
    parser.add_argument(
        "--registry",
        default="./data/document_registry.sqlite3",
    )
    return parser.parse_args()


def print_results(name, response) -> None:
    print(f"  {name}:")
    if not response.results:
        print("    No matching passages.")
        return

    for item in response.results:
        preview = " ".join(item.text.split())[:115]
        print(
            f"    {item.rank}. score={item.similarity:.4f} "
            f"patient={item.patient_id or 'unknown'} "
            f"{item.citation_label()} | {preview}"
        )


def main() -> int:
    args = parse_args()

    if args.reset:
        shutil.rmtree(args.chroma_dir, ignore_errors=True)
        Path(args.registry).unlink(missing_ok=True)

    provider = create_embedding_provider(
        args.embedding_backend
    )
    store = ChromaStore(
        embedding_provider=provider,
        persistence_directory=args.chroma_dir,
        backend=args.storage_backend,
    )
    registry = DocumentRegistry(args.registry)
    ingestion = LocalIngestionPipeline(
        chroma_store=store,
        registry=registry,
        config=IngestionConfig(),
    )
    report = ingestion.ingest(args.source)

    print("Ingestion")
    print(f"  processed: {report.processed_files}")
    print(f"  skipped:   {report.skipped_files}")
    print(f"  failed:    {report.failed_files}")
    print(f"  indexed:   {report.indexed_chunk_count}")

    if report.failed_files:
        for warning in report.warnings:
            print(f"  warning: {warning}")
        return 2

    vector = VectorRetriever(store)
    bm25 = BM25Retriever(store)
    bm25.rebuild()
    hybrid = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
    )

    filters = SearchFilters(
        patient_id=args.patient_id
    )

    for question in QUESTIONS:
        request = VectorSearchRequest(
            query=question,
            top_k=args.top_k,
            filters=filters,
        )

        print(f"\nQ: {question}")
        print_results("Vector", vector.search(request))
        print_results("BM25", bm25.search(request))
        print_results("Hybrid", hybrid.search(request))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
