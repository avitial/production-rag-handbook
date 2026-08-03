"""End-to-end Day 5 ingestion command."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook/app"))

# When this file is executed as ``python scripts/ingest_samples.py``, Python
# initially places ``scripts/`` rather than the repository root on sys.path.
# Add the project root explicitly so imports such as ``app.ingestion`` resolve.
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
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry


SAMPLE_QUESTIONS = [
    "What medication was prescribed at discharge?",
    "What allergies are documented?",
    "When is the follow-up appointment?",
    "What was the primary diagnosis?",
    "Which patient had a cardiology referral?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest local PDF/image medical samples and run "
            "semantic-search smoke tests."
        )
    )
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
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--storage-backend",
        choices=["auto", "chroma", "local"],
        default="auto",
    )
    parser.add_argument(
        "--collection",
        default="medical_documents",
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_data",
    )
    parser.add_argument(
        "--registry",
        default="./data/document_registry.sqlite3",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--smoke-query",
        action="store_true",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the selected index and registry before ingesting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.reset:
        shutil.rmtree(args.chroma_dir, ignore_errors=True)
        Path(args.registry).unlink(missing_ok=True)

    try:
        provider = create_embedding_provider(
            args.embedding_backend,
            model_name=args.model,
        )
        store = ChromaStore(
            embedding_provider=provider,
            collection_name=args.collection,
            persistence_directory=args.chroma_dir,
            backend=args.storage_backend,
        )
        registry = DocumentRegistry(args.registry)
        pipeline = LocalIngestionPipeline(
            chroma_store=store,
            registry=registry,
            config=IngestionConfig(
                max_characters=args.chunk_size,
                overlap_characters=args.overlap,
            ),
        )
        report = pipeline.ingest(args.source)

    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print("Ingestion complete")
    print(f"Embedding provider: {provider.model_name}")
    print(f"Storage backend:    {store.backend_name}")
    print(f"Discovered files:   {report.discovered_files}")
    print(f"Processed files:    {report.processed_files}")
    print(f"Skipped files:      {report.skipped_files}")
    print(f"Failed files:       {report.failed_files}")
    print(f"Pages:              {report.page_count}")
    print(f"Chunks generated:   {report.chunk_count}")
    print(f"Chunks indexed:     {report.indexed_chunk_count}")
    print(f"Collection count:   {store.count()}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"- {warning}")

    if args.smoke_query and store.count() > 0:
        print("\nSemantic-search smoke tests")

        for question in SAMPLE_QUESTIONS:
            result = store.query(question, top_k=3)
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]

            print(f"\nQ: {question}")

            for rank, (document, metadata, distance) in enumerate(
                zip(documents, metadatas, distances),
                start=1,
            ):
                preview = " ".join(document.split())[:140]
                print(
                    f"  {rank}. distance={distance:.4f} "
                    f"patient={metadata.get('patient_id', 'unknown')} "
                    f"page={metadata.get('page_number', '?')} | "
                    f"{preview}"
                )

    return 0 if report.failed_files == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
