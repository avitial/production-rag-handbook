"""Run an offline end-to-end retrieval evaluation.

The script ingests PDF/image/handwritten sources, builds vector and BM25
indexes, runs hybrid retrieval, matches results to references, and writes JSON
and Markdown reports.

Example:

    python evaluation/run_retrieval_evaluation.py \
      data/samples \
      --embedding-backend hash \
      --storage-backend local \
      --reset

Relevance matching uses the synthetic dataset's filename, section, and keyword
annotations. This is transparent and deterministic, though a larger production
evaluation should use manually labeled relevant chunk IDs.
"""

from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings.sentence_transformer import create_embedding_provider
from app.ingestion.pipeline import IngestionConfig, LocalIngestionPipeline
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import SearchFilters, VectorSearchRequest
from app.retrieval.vector_retriever import VectorRetriever
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry
from evaluation.metrics import evaluate_relevance_row, summarize_rows


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL at line {line_number}: {exc}"
            ) from exc
        for field in (
            "question_id", "question", "answerable",
            "expected_decision", "reference_keywords",
            "relevant_filenames", "relevant_sections",
        ):
            if field not in row:
                raise ValueError(
                    f"line {line_number} missing field: {field}"
                )
        rows.append(row)
    return rows


def passage_is_relevant(passage, example: dict[str, Any]) -> bool:
    """Apply transparent weak labels to a retrieved passage.

    For answerable questions:
      filename must match when annotated, and at least one expected keyword
      must appear in passage text/section.

    For unanswerable questions:
      no retrieved passage is labeled relevant.
    """
    if not example["answerable"]:
        return False

    filenames = set(example.get("relevant_filenames", []))
    if filenames and passage.filename not in filenames:
        return False

    haystack = (
        f"{passage.section or ''}\n{passage.text}"
    ).lower()
    keywords = [
        str(value).lower()
        for value in example.get("reference_keywords", [])
    ]
    return any(keyword in haystack for keyword in keywords)


def markdown_report(
    *,
    k: int,
    summary,
    detail_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Day 12 Retrieval Evaluation",
        "",
        f"- Evaluated questions: {summary.question_count}",
        f"- Cutoff: k={k}",
        f"- Mean Hit Rate@{k}: {summary.mean_hit_rate_at_k:.3f}",
        f"- Mean Precision@{k}: {summary.mean_precision_at_k:.3f}",
        f"- Mean Recall@{k}: {summary.mean_recall_at_k:.3f}",
        f"- Mean Reciprocal Rank: {summary.mean_reciprocal_rank:.3f}",
        f"- Mean nDCG@{k}: {summary.mean_ndcg_at_k:.3f}",
        "",
        "## Per-question results",
        "",
        "| ID | Answerable | Hit | P@k | R@k | RR | nDCG |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in detail_rows:
        m = item["metrics"]
        lines.append(
            f"| {item['question_id']} | {item['answerable']} | "
            f"{m['hit_rate_at_k']:.2f} | "
            f"{m['precision_at_k']:.2f} | "
            f"{m['recall_at_k']:.2f} | "
            f"{m['reciprocal_rank']:.2f} | "
            f"{m['ndcg_at_k']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This synthetic baseline uses filename and keyword weak labels. "
        "It verifies the evaluation plumbing and supports parameter comparisons, "
        "but it should be replaced or supplemented with human-labeled relevant "
        "chunk IDs before making production claims.",
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", default="data/samples")
    p.add_argument("--dataset", default="evaluation/dataset.jsonl")
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence-transformer", "hash"],
        default="auto",
    )
    p.add_argument(
        "--storage-backend",
        choices=["auto", "chroma", "local"],
        default="auto",
    )
    p.add_argument("--chroma-dir", default="./evaluation/output/chroma")
    p.add_argument(
        "--registry",
        default="./evaluation/output/registry.sqlite3",
    )
    p.add_argument(
        "--json-output",
        default="./evaluation/output/retrieval-results.json",
    )
    p.add_argument(
        "--markdown-output",
        default="./evaluation/output/retrieval-results.md",
    )
    p.add_argument("--reset", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        print("--k must be greater than zero", file=sys.stderr)
        return 2

    if args.reset:
        shutil.rmtree(args.chroma_dir, ignore_errors=True)
        Path(args.registry).unlink(missing_ok=True)

    examples = load_dataset(Path(args.dataset))
    provider = create_embedding_provider(args.embedding_backend)
    store = ChromaStore(
        embedding_provider=provider,
        persistence_directory=args.chroma_dir,
        backend=args.storage_backend,
        collection_name="day12_evaluation",
    )
    report = LocalIngestionPipeline(
        chroma_store=store,
        registry=DocumentRegistry(args.registry),
        config=IngestionConfig(),
    ).ingest(args.source)

    if report.failed_files:
        print("Ingestion failed:", report.warnings, file=sys.stderr)
        return 3

    bm25 = BM25Retriever(store)
    bm25.rebuild()
    hybrid = HybridRetriever(
        vector_retriever=VectorRetriever(store),
        bm25_retriever=bm25,
    )

    metric_rows = []
    details = []
    for example in examples:
        response = hybrid.search(
            VectorSearchRequest(
                query=example["question"],
                top_k=args.k,
                filters=SearchFilters(
                    patient_id=example.get("patient_id")
                ),
            )
        )
        relevance = [
            passage_is_relevant(item, example)
            for item in response.results
        ]
        # Dataset uses one logical relevant evidence target per question.
        total_relevant = 1 if example["answerable"] else 0
        row = evaluate_relevance_row(
            question_id=example["question_id"],
            relevance=relevance,
            k=args.k,
            total_relevant=total_relevant,
        )
        metric_rows.append(row)
        details.append({
            "question_id": example["question_id"],
            "question": example["question"],
            "answerable": example["answerable"],
            "metrics": row.to_dict(),
            "results": [
                {
                    "rank": item.rank,
                    "chunk_id": item.chunk_id,
                    "filename": item.filename,
                    "page_number": item.page_number,
                    "section": item.section,
                    "patient_id": item.patient_id,
                    "score": item.similarity,
                    "relevant": relevance[index],
                    "text_preview": " ".join(item.text.split())[:180],
                }
                for index, item in enumerate(response.results)
            ],
        })

    summary = summarize_rows(metric_rows)
    payload = {
        "configuration": {
            "k": args.k,
            "embedding_model": provider.model_name,
            "storage_backend": store.backend_name,
            "source": str(Path(args.source).resolve()),
        },
        "ingestion": {
            "processed_files": report.processed_files,
            "skipped_files": report.skipped_files,
            "failed_files": report.failed_files,
            "indexed_chunks": report.indexed_chunk_count,
        },
        "summary": summary.to_dict(),
        "questions": details,
    }

    json_path = Path(args.json_output)
    md_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(
        markdown_report(k=args.k, summary=summary, detail_rows=details),
        encoding="utf-8",
    )

    print("Retrieval evaluation complete")
    print(f"  questions:  {summary.question_count}")
    print(f"  hit@{args.k}:     {summary.mean_hit_rate_at_k:.3f}")
    print(f"  precision@{args.k}: {summary.mean_precision_at_k:.3f}")
    print(f"  recall@{args.k}:    {summary.mean_recall_at_k:.3f}")
    print(f"  MRR:        {summary.mean_reciprocal_rank:.3f}")
    print(f"  nDCG@{args.k}:    {summary.mean_ndcg_at_k:.3f}")
    print(f"  JSON: {json_path.resolve()}")
    print(f"  Markdown: {md_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
