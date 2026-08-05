"""Dependency-free retrieval and decision metrics for Day 12.

Metrics:

- Hit Rate@k: whether at least one relevant result appears in top k.
- Precision@k: relevant results divided by k.
- Recall@k: retrieved relevant items divided by all known relevant items.
- Reciprocal Rank: reciprocal of the first relevant rank.
- nDCG@k: ranking quality with binary relevance.
- Mean metrics across an evaluation run.
- Decision accuracy for accept/abstain/reject policies.

Relevance is determined outside these functions. This keeps the metric math
separate from project-specific matching rules.

Pseudo-code:

    receive ranked relevance labels such as [1, 0, 1, 0]
    truncate to k
    calculate each metric
    aggregate rows by arithmetic mean
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from statistics import mean
from typing import Iterable, Sequence, Any


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")


def hit_rate_at_k(relevance: Sequence[int | bool], k: int) -> float:
    _validate_k(k)
    return 1.0 if any(bool(x) for x in relevance[:k]) else 0.0


def precision_at_k(relevance: Sequence[int | bool], k: int) -> float:
    _validate_k(k)
    return sum(bool(x) for x in relevance[:k]) / k


def recall_at_k(
    relevance: Sequence[int | bool],
    k: int,
    *,
    total_relevant: int,
) -> float:
    _validate_k(k)
    if total_relevant < 0:
        raise ValueError("total_relevant must not be negative")
    if total_relevant == 0:
        return 1.0 if not any(bool(x) for x in relevance[:k]) else 0.0
    return min(
        1.0,
        sum(bool(x) for x in relevance[:k]) / total_relevant,
    )


def reciprocal_rank(relevance: Sequence[int | bool]) -> float:
    for rank, value in enumerate(relevance, start=1):
        if bool(value):
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance: Sequence[int | bool], k: int) -> float:
    _validate_k(k)
    return sum(
        float(bool(value)) / math.log2(rank + 1)
        for rank, value in enumerate(relevance[:k], start=1)
    )


def ndcg_at_k(
    relevance: Sequence[int | bool],
    k: int,
    *,
    total_relevant: int | None = None,
) -> float:
    _validate_k(k)
    actual = dcg_at_k(relevance, k)
    if total_relevant is None:
        total_relevant = sum(bool(x) for x in relevance)
    ideal = dcg_at_k(
        [1] * min(total_relevant, k),
        k,
    )
    return actual / ideal if ideal else 1.0


def decision_accuracy(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> float:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    if not expected:
        return 0.0
    return sum(
        left == right
        for left, right in zip(expected, predicted)
    ) / len(expected)


@dataclass(frozen=True)
class RetrievalMetricRow:
    question_id: str
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    retrieved_count: int
    relevant_retrieved_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalMetricSummary:
    question_count: int
    mean_hit_rate_at_k: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_relevance_row(
    *,
    question_id: str,
    relevance: Sequence[int | bool],
    k: int,
    total_relevant: int,
) -> RetrievalMetricRow:
    return RetrievalMetricRow(
        question_id=question_id,
        hit_rate_at_k=hit_rate_at_k(relevance, k),
        precision_at_k=precision_at_k(relevance, k),
        recall_at_k=recall_at_k(
            relevance,
            k,
            total_relevant=total_relevant,
        ),
        reciprocal_rank=reciprocal_rank(relevance[:k]),
        ndcg_at_k=ndcg_at_k(
            relevance,
            k,
            total_relevant=total_relevant,
        ),
        retrieved_count=min(len(relevance), k),
        relevant_retrieved_count=sum(
            bool(x) for x in relevance[:k]
        ),
    )


def summarize_rows(
    rows: Iterable[RetrievalMetricRow],
) -> RetrievalMetricSummary:
    values = list(rows)
    if not values:
        return RetrievalMetricSummary(0, 0, 0, 0, 0, 0)
    return RetrievalMetricSummary(
        question_count=len(values),
        mean_hit_rate_at_k=mean(x.hit_rate_at_k for x in values),
        mean_precision_at_k=mean(x.precision_at_k for x in values),
        mean_recall_at_k=mean(x.recall_at_k for x in values),
        mean_reciprocal_rank=mean(x.reciprocal_rank for x in values),
        mean_ndcg_at_k=mean(x.ndcg_at_k for x in values),
    )
