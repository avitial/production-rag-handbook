"""Run repeatable retrieval experiments and compare configurations.

The runner varies:

- Chunk size
- Chunk overlap
- Retrieval mode: vector, BM25, or hybrid
- Candidate counts
- Reciprocal Rank Fusion constant
- Final top-k

Every experiment receives an isolated vector-store directory and registry.
This prevents one configuration from reusing chunks created by another.

Pseudo-code:

    load and validate experiment configuration files
    load the synthetic evaluation dataset

    for each experiment:
        create isolated output directory
        ingest PDF and image/handwritten sources
        construct vector and BM25 retrievers
        choose vector, BM25, or hybrid mode
        run every evaluation question
        create binary relevance labels
        calculate Hit Rate, Precision, Recall, MRR, and nDCG
        measure ingestion and query latency
        save per-experiment JSON details

    write comparison CSV
    select best configurations by MRR and nDCG
    write tuning findings Markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings.sentence_transformer import create_embedding_provider
from app.ingestion.pipeline import IngestionConfig, LocalIngestionPipeline
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import (
    HybridRetrievalConfig,
    HybridRetriever,
)
from app.retrieval.models import SearchFilters, VectorSearchRequest
from app.retrieval.vector_retriever import VectorRetriever
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry
from evaluation.metrics import evaluate_relevance_row, summarize_rows
from evaluation.run_retrieval_evaluation import (
    load_dataset,
    passage_is_relevant,
)


VALID_RETRIEVAL_MODES = {"vector", "bm25", "hybrid"}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    description: str
    chunk_size: int
    chunk_overlap: int
    retrieval_mode: str
    top_k: int
    vector_candidates: int
    bm25_candidates: int
    fusion_constant: int
    embedding_backend: str
    storage_backend: str

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be blank")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller "
                "than chunk_size"
            )
        if self.retrieval_mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(
                "retrieval_mode must be vector, bm25, or hybrid"
            )
        for field_name in (
            "top_k",
            "vector_candidates",
            "bm25_candidates",
            "fusion_constant",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(
                    f"{field_name} must be greater than zero"
                )


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    description: str
    retrieval_mode: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    vector_candidates: int
    bm25_candidates: int
    fusion_constant: int
    embedding_model: str
    storage_backend: str
    question_count: int
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    ingestion_ms: float
    mean_query_ms: float
    indexed_chunks: int
    failed_files: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path) -> ExperimentConfig:
    """Read and validate one JSON experiment configuration."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid config JSON {path}: {exc}") from exc

    required = {
        "experiment_id",
        "description",
        "chunk_size",
        "chunk_overlap",
        "retrieval_mode",
        "top_k",
        "vector_candidates",
        "bm25_candidates",
        "fusion_constant",
        "embedding_backend",
        "storage_backend",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"{path} missing fields: {sorted(missing)}"
        )

    return ExperimentConfig(**payload)


def discover_configs(
    *,
    config_path: Path | None,
    configs_directory: Path,
) -> list[tuple[Path, ExperimentConfig]]:
    """Load one config or every JSON config in a directory."""
    if config_path is not None:
        return [(config_path, load_config(config_path))]

    paths = sorted(configs_directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no JSON experiment configs in {configs_directory}"
        )
    return [(path, load_config(path)) for path in paths]


def build_retriever(
    config: ExperimentConfig,
    store: ChromaStore,
):
    """Construct the selected retrieval mode from shared indexes."""
    vector = VectorRetriever(store)
    bm25 = BM25Retriever(store)
    bm25.rebuild()

    if config.retrieval_mode == "vector":
        return vector

    if config.retrieval_mode == "bm25":
        return bm25

    return HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        config=HybridRetrievalConfig(
            vector_top_k=config.vector_candidates,
            keyword_top_k=config.bm25_candidates,
            fusion_constant=config.fusion_constant,
        ),
    )


def run_one_experiment(
    *,
    config: ExperimentConfig,
    source: Path,
    dataset: list[dict[str, Any]],
    output_root: Path,
    reset: bool,
) -> tuple[ExperimentResult, dict[str, Any]]:
    """Run ingestion and retrieval evaluation for one configuration."""
    experiment_dir = output_root / config.experiment_id
    vector_dir = experiment_dir / "vectors"
    registry_path = experiment_dir / "registry.sqlite3"

    if reset:
        shutil.rmtree(experiment_dir, ignore_errors=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    provider = create_embedding_provider(
        config.embedding_backend
    )
    store = ChromaStore(
        embedding_provider=provider,
        collection_name=f"experiment_{config.experiment_id}",
        persistence_directory=vector_dir,
        backend=config.storage_backend,
    )

    ingestion_started = time.perf_counter()
    ingestion_report = LocalIngestionPipeline(
        chroma_store=store,
        registry=DocumentRegistry(registry_path),
        config=IngestionConfig(
            max_characters=config.chunk_size,
            overlap_characters=config.chunk_overlap,
        ),
    ).ingest(source)
    ingestion_ms = (
        time.perf_counter() - ingestion_started
    ) * 1000

    if ingestion_report.failed_files:
        raise RuntimeError(
            f"{config.experiment_id} ingestion failed: "
            f"{ingestion_report.warnings}"
        )

    retriever = build_retriever(config, store)
    metric_rows = []
    question_details = []
    query_durations = []

    for example in dataset:
        started = time.perf_counter()
        response = retriever.search(
            VectorSearchRequest(
                query=example["question"],
                top_k=config.top_k,
                filters=SearchFilters(
                    patient_id=example.get("patient_id")
                ),
            )
        )
        query_durations.append(
            (time.perf_counter() - started) * 1000
        )

        relevance = [
            passage_is_relevant(item, example)
            for item in response.results
        ]
        total_relevant = 1 if example["answerable"] else 0
        metric = evaluate_relevance_row(
            question_id=example["question_id"],
            relevance=relevance,
            k=config.top_k,
            total_relevant=total_relevant,
        )
        metric_rows.append(metric)

        question_details.append(
            {
                "question_id": example["question_id"],
                "question": example["question"],
                "answerable": example["answerable"],
                "metrics": metric.to_dict(),
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
                        "text_preview": " ".join(
                            item.text.split()
                        )[:180],
                    }
                    for index, item in enumerate(
                        response.results
                    )
                ],
            }
        )

    summary = summarize_rows(metric_rows)
    result = ExperimentResult(
        experiment_id=config.experiment_id,
        description=config.description,
        retrieval_mode=config.retrieval_mode,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k,
        vector_candidates=config.vector_candidates,
        bm25_candidates=config.bm25_candidates,
        fusion_constant=config.fusion_constant,
        embedding_model=provider.model_name,
        storage_backend=store.backend_name,
        question_count=summary.question_count,
        hit_rate_at_k=summary.mean_hit_rate_at_k,
        precision_at_k=summary.mean_precision_at_k,
        recall_at_k=summary.mean_recall_at_k,
        mean_reciprocal_rank=(
            summary.mean_reciprocal_rank
        ),
        ndcg_at_k=summary.mean_ndcg_at_k,
        ingestion_ms=ingestion_ms,
        mean_query_ms=mean(query_durations),
        indexed_chunks=ingestion_report.indexed_chunk_count,
        failed_files=ingestion_report.failed_files,
    )

    detail_payload = {
        "configuration": asdict(config),
        "result": result.to_dict(),
        "ingestion": {
            "discovered_files": (
                ingestion_report.discovered_files
            ),
            "processed_files": ingestion_report.processed_files,
            "skipped_files": ingestion_report.skipped_files,
            "failed_files": ingestion_report.failed_files,
            "page_count": ingestion_report.page_count,
            "chunk_count": ingestion_report.chunk_count,
            "indexed_chunk_count": (
                ingestion_report.indexed_chunk_count
            ),
        },
        "questions": question_details,
    }

    (experiment_dir / "results.json").write_text(
        json.dumps(detail_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result, detail_payload


def write_comparison_csv(
    path: Path,
    results: Iterable[ExperimentResult],
) -> None:
    """Write one row per experiment with stable column order."""
    values = list(results)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "description",
        "retrieval_mode",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "vector_candidates",
        "bm25_candidates",
        "fusion_constant",
        "embedding_model",
        "storage_backend",
        "question_count",
        "hit_rate_at_k",
        "precision_at_k",
        "recall_at_k",
        "mean_reciprocal_rank",
        "ndcg_at_k",
        "ingestion_ms",
        "mean_query_ms",
        "indexed_chunks",
        "failed_files",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for result in values:
            row = result.to_dict()
            for metric in (
                "hit_rate_at_k",
                "precision_at_k",
                "recall_at_k",
                "mean_reciprocal_rank",
                "ndcg_at_k",
                "ingestion_ms",
                "mean_query_ms",
            ):
                row[metric] = f"{row[metric]:.6f}"
            writer.writerow(row)


def _best(
    results: list[ExperimentResult],
    field_name: str,
) -> ExperimentResult:
    return max(
        results,
        key=lambda item: getattr(item, field_name),
    )


def tuning_findings_markdown(
    results: list[ExperimentResult],
) -> str:
    """Create transparent findings derived from actual experiment output."""
    best_mrr = _best(results, "mean_reciprocal_rank")
    best_ndcg = _best(results, "ndcg_at_k")
    best_precision = _best(results, "precision_at_k")
    fastest = min(results, key=lambda item: item.mean_query_ms)

    ordered = sorted(
        results,
        key=lambda item: (
            -item.mean_reciprocal_rank,
            -item.ndcg_at_k,
            -item.hit_rate_at_k,
            item.mean_query_ms,
        ),
    )

    lines = [
        "# Day 13 Tuning Findings",
        "",
        "## Scope",
        "",
        "These findings are generated from the synthetic Day 12 evaluation "
        "dataset using the offline deterministic embedding backend. They "
        "validate the experiment framework and reveal relative behavior in "
        "this small dataset; they are not production-quality semantic-search "
        "benchmarks.",
        "",
        "## Best observed configurations",
        "",
        f"- Best MRR: **{best_mrr.experiment_id}** "
        f"({best_mrr.mean_reciprocal_rank:.3f})",
        f"- Best nDCG: **{best_ndcg.experiment_id}** "
        f"({best_ndcg.ndcg_at_k:.3f})",
        f"- Best Precision@k: **{best_precision.experiment_id}** "
        f"({best_precision.precision_at_k:.3f})",
        f"- Fastest mean query time: **{fastest.experiment_id}** "
        f"({fastest.mean_query_ms:.3f} ms)",
        "",
        "## Comparison",
        "",
        "| Rank | Experiment | Mode | Chunk/Overlap | Hit | Precision | Recall | MRR | nDCG | Query ms | Chunks |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, item in enumerate(ordered, start=1):
        lines.append(
            f"| {rank} | {item.experiment_id} | "
            f"{item.retrieval_mode} | "
            f"{item.chunk_size}/{item.chunk_overlap} | "
            f"{item.hit_rate_at_k:.3f} | "
            f"{item.precision_at_k:.3f} | "
            f"{item.recall_at_k:.3f} | "
            f"{item.mean_reciprocal_rank:.3f} | "
            f"{item.ndcg_at_k:.3f} | "
            f"{item.mean_query_ms:.3f} | "
            f"{item.indexed_chunks} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "1. Prefer configurations that improve MRR and nDCG without a large "
        "latency increase. Hit Rate alone can hide poor ranking.",
        "2. Smaller chunks usually increase the number of indexed chunks and "
        "can isolate exact evidence, but they can lose surrounding context.",
        "3. Larger chunks reduce chunk count but may dilute exact evidence "
        "inside broad passages.",
        "4. BM25 should be watched for exact IDs, medications, codes, and "
        "dates. Vector retrieval is intended to help paraphrases.",
        "5. Hybrid retrieval is most useful when its gains exceed its added "
        "complexity and latency.",
        "",
        "## Recommended next baseline",
        "",
        f"Use **{best_mrr.experiment_id}** as the next offline baseline because "
        "it achieved the best first-relevant-result ranking on this dataset. "
        "Before adopting it for production, rerun the experiment with the real "
        "Sentence Transformer backend and manually labeled relevant chunk IDs.",
        "",
        "## Limitations",
        "",
        "- Only two synthetic records are evaluated.",
        "- Relevance uses filename and keyword weak labels.",
        "- Hash embeddings validate plumbing, not semantic model quality.",
        "- Timing values are local development measurements.",
        "- Ties can be common on a small dataset.",
        "",
        "## Next experiments",
        "",
        "- Run all configs with `sentence-transformer` and real ChromaDB.",
        "- Add 50–100 manually reviewed questions.",
        "- Store manually labeled relevant chunk IDs.",
        "- Compare RRF constants such as 20, 40, 60, and 100.",
        "- Evaluate final confidence-policy decision accuracy.",
        "- Add OCR-noise and scanned-PDF examples.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        default="data/samples",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/dataset.jsonl",
    )
    parser.add_argument(
        "--configs",
        default="configs/experiments",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Run only one experiment config.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/experiment-output",
    )
    parser.add_argument(
        "--comparison-csv",
        default="reports/retrieval-comparison.csv",
    )
    parser.add_argument(
        "--findings",
        default="reports/tuning-findings.md",
    )
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        configs = discover_configs(
            config_path=(
                Path(args.config) if args.config else None
            ),
            configs_directory=Path(args.configs),
        )
        dataset = load_dataset(Path(args.dataset))
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir)
    if args.reset:
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []

    for path, config in configs:
        print(
            f"Running {config.experiment_id} "
            f"from {path.name}..."
        )
        try:
            result, _details = run_one_experiment(
                config=config,
                source=Path(args.source).resolve(),
                dataset=dataset,
                output_root=output_root,
                reset=args.reset,
            )
        except Exception as exc:
            print(
                f"Experiment {config.experiment_id} failed: {exc}",
                file=sys.stderr,
            )
            return 3

        results.append(result)
        print(
            f"  MRR={result.mean_reciprocal_rank:.3f} "
            f"nDCG={result.ndcg_at_k:.3f} "
            f"Hit={result.hit_rate_at_k:.3f} "
            f"query={result.mean_query_ms:.3f} ms"
        )

    write_comparison_csv(
        Path(args.comparison_csv),
        results,
    )
    Path(args.findings).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(args.findings).write_text(
        tuning_findings_markdown(results),
        encoding="utf-8",
    )

    print("\nExperiment comparison complete")
    print(f"  experiments: {len(results)}")
    print(
        f"  CSV: {Path(args.comparison_csv).resolve()}"
    )
    print(
        f"  Findings: {Path(args.findings).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())