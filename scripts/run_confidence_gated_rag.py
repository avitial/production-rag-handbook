"""Run Day 11 confidence-gated RAG end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.confidence.features import extract_confidence_features
from app.confidence.policy import ConfidencePolicy
from app.embeddings.sentence_transformer import (
    create_embedding_provider,
)
from app.generation.context_builder import ContextBuilder
from app.generation.local_llm_client import (
    DeterministicLocalLLMClient,
)
from app.generation.rag_generator import RAGGenerator
from app.ingestion.pipeline import (
    IngestionConfig,
    LocalIngestionPipeline,
)
from app.observability.logger import StructuredLogger
from app.observability.metrics import MetricsCollector
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
from app.validation.answer_validator import validate_answer
from app.validation.json_validator import validate_rag_response_json


QUESTIONS = [
    ("What medication was prescribed at discharge?", None),
    ("What allergies are documented?", "SYN-200849"),
    ("When is the follow-up appointment?", "SYN-200849"),
    ("What was the primary diagnosis?", "SYN-200989"),
    ("Which patient had a cardiology referral?", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="data/samples")
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
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--chroma-dir", default="./chroma_data")
    parser.add_argument(
        "--registry",
        default="./data/document_registry.sqlite3",
    )
    parser.add_argument(
        "--log-file",
        default="./logs/confidence-events.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.reset:
        shutil.rmtree(args.chroma_dir, ignore_errors=True)
        Path(args.registry).unlink(missing_ok=True)
        Path(args.log_file).unlink(missing_ok=True)

    provider = create_embedding_provider(
        args.embedding_backend
    )
    store = ChromaStore(
        embedding_provider=provider,
        persistence_directory=args.chroma_dir,
        backend=args.storage_backend,
    )
    report = LocalIngestionPipeline(
        chroma_store=store,
        registry=DocumentRegistry(args.registry),
        config=IngestionConfig(),
    ).ingest(args.source)

    print("Ingestion")
    print(f"  processed: {report.processed_files}")
    print(f"  skipped:   {report.skipped_files}")
    print(f"  failed:    {report.failed_files}")
    print(f"  indexed:   {report.indexed_chunk_count}")

    if report.failed_files:
        return 2

    bm25 = BM25Retriever(store)
    bm25.rebuild()
    hybrid = HybridRetriever(
        vector_retriever=VectorRetriever(store),
        bm25_retriever=bm25,
    )
    reranker = create_reranker(args.reranker_backend)
    generator = RAGGenerator(
        llm_client=DeterministicLocalLLMClient(),
        context_builder=ContextBuilder(),
    )
    policy = ConfidencePolicy()
    logger = StructuredLogger(args.log_file)
    metrics = MetricsCollector()

    for question, patient_id in QUESTIONS:
        metrics.increment("requests_total")

        started = time.perf_counter()
        retrieval = hybrid.search(
            VectorSearchRequest(
                query=question,
                top_k=10,
                filters=SearchFilters(
                    patient_id=patient_id
                ),
            )
        )
        metrics.observe_latency(
            "retrieval_ms",
            (time.perf_counter() - started) * 1000,
        )

        reranking = reranker.rerank(
            RerankingRequest(
                query=question,
                passages=retrieval.results,
                top_n=5,
            )
        )
        metrics.observe_latency(
            "reranking_ms",
            reranking.diagnostics.duration_ms,
        )

        answer = generator.generate(
            question=question,
            passages=reranking.results,
        )
        metrics.observe_latency(
            "generation_ms",
            answer.diagnostics.duration_ms,
        )

        answer_validation = validate_answer(
            answer,
            expected_patient_id=patient_id,
        )
        json_validation = validate_rag_response_json(
            {
                "question": answer.question,
                "answer": answer.answer,
                "citations": [
                    {
                        "source_number": item.source_number,
                        "filename": item.filename,
                        "page_number": item.page_number,
                    }
                    for item in answer.citations
                ],
                "abstained": answer.diagnostics.abstained,
                "validation": {
                    "valid": answer_validation.valid,
                },
                "diagnostics": {},
            }
        )

        features = extract_confidence_features(
            retrieval_response=retrieval,
            reranking_response=reranking,
            rag_answer=answer,
            answer_validation=answer_validation,
            json_validation=json_validation,
        )
        assessment = policy.evaluate(features)

        metrics.increment(
            f"answers_{assessment.decision.value}_total"
        )
        metrics.set_gauge(
            "last_context_source_count",
            features.context_source_count,
        )

        logger.log(
            "confidence_decision",
            fields={
                "question": question,
                "patient_id": patient_id,
                "decision": assessment.decision.value,
                "reason_codes": [
                    reason.code
                    for reason in assessment.reasons
                ],
                "features": features.to_dict(),
            },
        )

        print(f"\nQ: {question}")
        print(f"Decision: {assessment.decision.value.upper()}")
        print(f"Answer: {answer.answer}")
        print(
            "Reasons:",
            ", ".join(
                reason.code for reason in assessment.reasons
            ),
        )

    print("\nMetrics")
    print(
        json.dumps(
            metrics.snapshot(),
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nLog file: {Path(args.log_file).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
