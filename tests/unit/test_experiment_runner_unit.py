"""Unit tests for Day 13 experiment configuration and reporting."""

from pathlib import Path
import json

import pytest

from evaluation.run_experiments import (
    ExperimentConfig,
    ExperimentResult,
    load_config,
    tuning_findings_markdown,
)


def test_valid_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "test",
                "description": "test experiment",
                "chunk_size": 500,
                "chunk_overlap": 75,
                "retrieval_mode": "hybrid",
                "top_k": 5,
                "vector_candidates": 15,
                "bm25_candidates": 15,
                "fusion_constant": 60,
                "embedding_backend": "hash",
                "storage_backend": "local",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.experiment_id == "test"
    assert config.retrieval_mode == "hybrid"


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_id="bad",
            description="bad",
            chunk_size=100,
            chunk_overlap=100,
            retrieval_mode="hybrid",
            top_k=5,
            vector_candidates=10,
            bm25_candidates=10,
            fusion_constant=60,
            embedding_backend="hash",
            storage_backend="local",
        )


def test_invalid_retrieval_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_id="bad",
            description="bad",
            chunk_size=500,
            chunk_overlap=50,
            retrieval_mode="unknown",
            top_k=5,
            vector_candidates=10,
            bm25_candidates=10,
            fusion_constant=60,
            embedding_backend="hash",
            storage_backend="local",
        )


def result(
    experiment_id: str,
    *,
    mrr: float,
    ndcg: float,
    precision: float,
    query_ms: float,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        description=experiment_id,
        retrieval_mode="hybrid",
        chunk_size=500,
        chunk_overlap=75,
        top_k=5,
        vector_candidates=15,
        bm25_candidates=15,
        fusion_constant=60,
        embedding_model="hash",
        storage_backend="local",
        question_count=8,
        hit_rate_at_k=0.75,
        precision_at_k=precision,
        recall_at_k=1.0,
        mean_reciprocal_rank=mrr,
        ndcg_at_k=ndcg,
        ingestion_ms=10,
        mean_query_ms=query_ms,
        indexed_chunks=10,
        failed_files=0,
    )


def test_findings_identify_best_experiment() -> None:
    markdown = tuning_findings_markdown(
        [
            result(
                "a",
                mrr=0.4,
                ndcg=0.6,
                precision=0.2,
                query_ms=2,
            ),
            result(
                "b",
                mrr=0.8,
                ndcg=0.9,
                precision=0.3,
                query_ms=3,
            ),
        ]
    )

    assert "Best MRR: **b**" in markdown
    assert "Best nDCG: **b**" in markdown
    assert "Use **b** as the next offline baseline" in markdown