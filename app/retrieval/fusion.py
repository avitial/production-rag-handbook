"""Rank-fusion utilities for combining BM25 and vector retrieval.

Raw BM25 scores and vector similarities are not directly comparable because
they come from different scoring systems. Reciprocal Rank Fusion (RRF) combines
rank positions rather than raw values.

Formula:

    RRF score(document) =
        sum over result lists of 1 / (fusion_constant + rank)

Pseudo-code:

    for every ranked result list:
        for every result:
            add reciprocal-rank contribution by chunk ID
            retain provenance and per-method ranks
    sort by descending fused score
    return fused candidates
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.retrieval.models import RetrievedPassage


@dataclass(frozen=True)
class FusionCandidate:
    """One chunk with fused score and method-level evidence."""

    passage: RetrievedPassage
    fusion_score: float
    method_ranks: dict[str, int] = field(default_factory=dict)
    method_scores: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    result_sets: dict[str, Iterable[RetrievedPassage]],
    *,
    fusion_constant: int = 60,
    top_n: int | None = None,
) -> tuple[FusionCandidate, ...]:
    """Fuse ranked result lists using Reciprocal Rank Fusion.

    ``fusion_constant`` dampens the impact of rank differences. The commonly
    used baseline value is 60, but it should be evaluated rather than treated
    as universally optimal.
    """
    if fusion_constant <= 0:
        raise ValueError("fusion_constant must be greater than zero")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    candidates: dict[str, dict] = {}

    for method_name, results in result_sets.items():
        if not method_name.strip():
            raise ValueError("method names must not be blank")

        for fallback_rank, passage in enumerate(
            results,
            start=1,
        ):
            rank = passage.rank or fallback_rank
            record = candidates.setdefault(
                passage.chunk_id,
                {
                    "passage": passage,
                    "fusion_score": 0.0,
                    "method_ranks": {},
                    "method_scores": {},
                },
            )
            record["fusion_score"] += 1 / (
                fusion_constant + rank
            )
            record["method_ranks"][method_name] = rank
            record["method_scores"][method_name] = (
                passage.similarity
            )

            # Prefer the passage carrying more metadata when the same chunk
            # appears through several retrieval methods.
            if len(passage.metadata) > len(
                record["passage"].metadata
            ):
                record["passage"] = passage

    fused = [
        FusionCandidate(
            passage=record["passage"],
            fusion_score=record["fusion_score"],
            method_ranks=dict(record["method_ranks"]),
            method_scores=dict(record["method_scores"]),
        )
        for record in candidates.values()
    ]

    fused.sort(
        key=lambda item: (
            -item.fusion_score,
            min(item.method_ranks.values()),
            item.passage.chunk_id,
        )
    )

    if top_n is not None:
        fused = fused[:top_n]

    return tuple(fused)


def as_ranked_passages(
    candidates: Iterable[FusionCandidate],
) -> tuple[RetrievedPassage, ...]:
    """Convert fusion candidates to shared ranked passage models."""
    output: list[RetrievedPassage] = []

    for rank, candidate in enumerate(candidates, start=1):
        metadata = {
            **candidate.passage.metadata,
            "retrieval_method": "hybrid",
            "fusion_score": candidate.fusion_score,
            "method_ranks": str(candidate.method_ranks),
            "method_scores": str(candidate.method_scores),
        }
        output.append(
            replace(
                candidate.passage,
                rank=rank,
                distance=0.0,
                similarity=candidate.fusion_score,
                metadata=metadata,
            )
        )

    return tuple(output)
