"""Run the complete Day 10 validated RAG pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.schemas import (
    CitationSchema,
    RAGAnswerResponse,
    ValidationIssueSchema,
    ValidationSummarySchema,
)
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
from app.validation.json_validator import (
    validate_rag_response_json,
)


QUESTIONS = [
    ("What medication was prescribed at discharge?", None),
    ("What allergies are documented?", "SYN-200849"),
    ("When is the follow-up appointment?", "SYN-200849"),
    ("What was the primary diagnosis?", "SYN-200989"),
    ("Which patient had a cardiology referral?", None),
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
    reranker = create_reranker(
        args.reranker_backend
    )
    generator = RAGGenerator(
        llm_client=DeterministicLocalLLMClient(),
        context_builder=ContextBuilder(),
    )

    overall_valid = True

    for question, patient_id in QUESTIONS:
        retrieved = hybrid.search(
            VectorSearchRequest(
                query=question,
                top_k=10,
                filters=SearchFilters(
                    patient_id=patient_id
                ),
            )
        )
        reranked = reranker.rerank(
            RerankingRequest(
                query=question,
                passages=retrieved.results,
                top_n=5,
            )
        )
        answer = generator.generate(
            question=question,
            passages=reranked.results,
        )
        answer_validation = validate_answer(
            answer,
            expected_patient_id=patient_id,
        )

        response = RAGAnswerResponse(
            question=answer.question,
            answer=answer.answer,
            citations=tuple(
                CitationSchema(
                    source_number=item.source_number,
                    chunk_id=item.chunk_id,
                    filename=item.filename,
                    page_number=item.page_number,
                    section=item.section,
                    patient_id=item.patient_id,
                    citation_label=item.citation_label,
                )
                for item in answer.citations
            ),
            abstained=answer.diagnostics.abstained,
            validation=ValidationSummarySchema(
                valid=answer_validation.valid,
                citation_valid=(
                    answer_validation.citation_result.valid
                ),
                answer_grounded=answer_validation.grounded,
                json_valid=True,
                issues=tuple(
                    ValidationIssueSchema(
                        code=issue.code,
                        message=issue.message,
                        severity=issue.severity,
                        details=issue.details,
                    )
                    for issue in answer_validation.issues
                ),
            ),
            diagnostics={
                "llm_model": answer.diagnostics.llm_model,
                "context_source_count": (
                    answer.diagnostics.context_source_count
                ),
                "duration_ms": answer.diagnostics.duration_ms,
            },
        )

        json_validation = validate_rag_response_json(
            response
        )
        overall_valid = (
            overall_valid
            and answer_validation.valid
            and json_validation.valid
        )

        print(f"\nQ: {question}")
        print(f"A: {answer.answer}")
        print(
            "  citation valid:",
            answer_validation.citation_result.valid,
        )
        print(
            "  answer grounded:",
            answer_validation.grounded,
        )
        print("  JSON valid:", json_validation.valid)

        if json_validation.json_text:
            print(
                json.dumps(
                    json_validation.normalized,
                    indent=2,
                    ensure_ascii=False,
                )
            )

    return 0 if overall_valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
