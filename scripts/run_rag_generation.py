"""End-to-end Day 9 ingestion, retrieval, reranking, and generation demo."""

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
from app.generation.context_builder import (
    ContextBuilder,
    ContextBuilderConfig,
)
from app.generation.local_llm_client import (
    DeterministicLocalLLMClient,
)
from app.generation.rag_generator import RAGGenerator
from app.ingestion.pipeline import (
    IngestionConfig,
    LocalIngestionPipeline,
)
from app.reranking.base import RerankingRequest
from app.reranking.cross_encoder import create_reranker
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
    (
        "What medication was prescribed at discharge?",
        None,
    ),
    (
        "What allergies are documented?",
        "SYN-200849",
    ),
    (
        "When is the follow-up appointment?",
        "SYN-200849",
    ),
    (
        "What was the primary diagnosis?",
        "SYN-200989",
    ),
    (
        "Which patient had a cardiology referral?",
        None,
    ),
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
    parser.add_argument(
        "--reranker-backend",
        choices=["auto", "cross-encoder", "deterministic"],
        default="auto",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--context-characters",
        type=int,
        default=4000,
    )
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


def main() -> int:
    args = parse_args()

    if args.reset:
        shutil.rmtree(args.chroma_dir, ignore_errors=True)
        Path(args.registry).unlink(missing_ok=True)

    embedding_provider = create_embedding_provider(
        args.embedding_backend
    )
    store = ChromaStore(
        embedding_provider=embedding_provider,
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
    reranker = create_reranker(
        args.reranker_backend
    )
    generator = RAGGenerator(
        llm_client=DeterministicLocalLLMClient(),
        context_builder=ContextBuilder(
            ContextBuilderConfig(
                maximum_characters=args.context_characters,
                maximum_sources=args.final_k,
            )
        ),
    )

    print(f"Embedding: {embedding_provider.model_name}")
    print(f"Reranker:  {reranker.model_name}")
    print(f"Generator: {generator.llm_client.model_name}")

    for question, patient_id in QUESTIONS:
        retrieved = hybrid.search(
            VectorSearchRequest(
                query=question,
                top_k=args.candidate_k,
                filters=SearchFilters(
                    patient_id=patient_id
                ),
            )
        )
        reranked = reranker.rerank(
            RerankingRequest(
                query=question,
                passages=retrieved.results,
                top_n=args.final_k,
            )
        )
        answer = generator.generate(
            question=question,
            passages=reranked.results,
        )

        print(f"\nQ: {question}")
        print(f"A: {answer.answer}")
        print(
            "  abstained:",
            answer.diagnostics.abstained,
        )
        print(
            "  context sources:",
            answer.diagnostics.context_source_count,
        )

        for citation in answer.citations:
            print(
                f"  [SOURCE {citation.source_number}] "
                f"{citation.citation_label} "
                f"patient={citation.patient_id or 'unknown'}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
