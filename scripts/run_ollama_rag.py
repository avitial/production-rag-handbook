"""Run ingestion and citation-grounded RAG with a real local Ollama model.

This command intentionally performs a readiness check before ingesting data.

Example:

    python scripts/run_ollama_rag.py data/samples \
      --model gemma3:4b \
      --reset

Pseudo-code:

    create Ollama client
    confirm server and model readiness
    build shared project services
    ingest local PDF/image/handwritten sources
    rebuild hybrid indexes
    retrieve and rerank evidence for each question
    generate through Ollama
    validate citations and answer grounding
    print answer, decision inputs, and citations
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.dependencies import ApiSettings, RuntimeServices
from app.generation.ollama_llm_client import (
    OllamaLLMClient,
    OllamaLLMConfig,
)
from app.reranking.base import RerankingRequest
from app.retrieval.models import (
    SearchFilters,
    VectorSearchRequest,
)
from app.validation.answer_validator import validate_answer


QUESTIONS = [
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
        "--host",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--model",
        default="gemma3:4b",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["hash", "sentence-transformer"],
        default="hash",
    )
    parser.add_argument(
        "--storage-backend",
        choices=["local", "chroma"],
        default="local",
    )
    parser.add_argument(
        "--reranker-backend",
        choices=["deterministic", "cross-encoder"],
        default="deterministic",
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--runtime-dir",
        default="./runtime/ollama-demo",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = Path(args.runtime_dir)

    if args.reset:
        shutil.rmtree(runtime, ignore_errors=True)

    ollama_client = OllamaLLMClient(
        OllamaLLMConfig(
            host=args.host,
            model=args.model,
        )
    )
    status = ollama_client.require_ready()

    print(
        f"Ollama ready: {status.model} at {status.host}"
    )

    services = RuntimeServices(
        ApiSettings(
            embedding_backend=args.embedding_backend,
            storage_backend=args.storage_backend,
            reranker_backend=args.reranker_backend,
            llm_backend="ollama",
            ollama_model=args.model,
            ollama_host=args.host,
            persistence_directory=str(runtime / "vectors"),
            registry_path=str(runtime / "registry.sqlite3"),
            upload_directory=str(runtime / "uploads"),
            log_path=str(runtime / "events.jsonl"),
            collection_name="ollama_rag_demo",
        ),
        llm_client=ollama_client,
    )

    report = services.ingest(args.source)
    print("\nIngestion")
    print(f"  processed: {report.processed_files}")
    print(f"  skipped:   {report.skipped_files}")
    print(f"  failed:    {report.failed_files}")
    print(f"  indexed:   {report.indexed_chunk_count}")

    if report.failed_files:
        for warning in report.warnings:
            print(f"  warning: {warning}")
        return 4

    for question, patient_id in QUESTIONS:
        retrieval = services.hybrid_retriever.search(
            VectorSearchRequest(
                query=question,
                top_k=10,
                filters=SearchFilters(
                    patient_id=patient_id
                ),
            )
        )
        reranking = services.reranker.rerank(
            RerankingRequest(
                query=question,
                passages=retrieval.results,
                top_n=5,
            )
        )
        answer = services.generator.generate(
            question=question,
            passages=reranking.results,
        )
        validation = validate_answer(
            answer,
            expected_patient_id=patient_id,
        )

        print(f"\nQ: {question}")
        print(f"A: {answer.answer}")
        print(f"Model: {answer.llm_response.model_name}")
        print(f"Valid: {validation.valid}")
        for citation in answer.citations:
            print(
                f"  [SOURCE {citation.source_number}] "
                f"{citation.citation_label}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
