"""Unit tests for dependency-free evaluation metrics."""

import math
import pytest

from evaluation.metrics import (
    decision_accuracy,
    evaluate_relevance_row,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_rows,
)


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k([0, 1, 0], 1) == 0
    assert hit_rate_at_k([0, 1, 0], 2) == 1


def test_precision_at_k_uses_requested_denominator() -> None:
    assert precision_at_k([1, 0, 1], 2) == 0.5
    assert precision_at_k([1], 3) == pytest.approx(1 / 3)


def test_recall_at_k() -> None:
    assert recall_at_k([1, 0, 1], 2, total_relevant=2) == 0.5
    assert recall_at_k([1, 1], 2, total_relevant=2) == 1.0


def test_unanswerable_recall_rewards_no_relevant_results() -> None:
    assert recall_at_k([0, 0], 2, total_relevant=0) == 1.0
    assert recall_at_k([1, 0], 2, total_relevant=0) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0]) == 0


def test_ndcg_is_one_for_ideal_ranking() -> None:
    assert ndcg_at_k([1, 1, 0], 3, total_relevant=2) == 1.0


def test_ndcg_penalizes_late_relevance() -> None:
    value = ndcg_at_k([0, 1, 0], 3, total_relevant=1)
    assert 0 < value < 1


def test_decision_accuracy() -> None:
    assert decision_accuracy(
        ["accept", "abstain", "accept"],
        ["accept", "abstain", "reject"],
    ) == pytest.approx(2 / 3)


def test_summary_means_rows() -> None:
    rows = [
        evaluate_relevance_row(
            question_id="q1",
            relevance=[1, 0],
            k=2,
            total_relevant=1,
        ),
        evaluate_relevance_row(
            question_id="q2",
            relevance=[0, 0],
            k=2,
            total_relevant=1,
        ),
    ]
    summary = summarize_rows(rows)

    assert summary.question_count == 2
    assert summary.mean_hit_rate_at_k == 0.5
    assert summary.mean_reciprocal_rank == 0.5


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        hit_rate_at_k([1], 0)
