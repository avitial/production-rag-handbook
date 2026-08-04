"""Unit tests for Reciprocal Rank Fusion."""
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.retrieval.fusion import (
    as_ranked_passages,
    reciprocal_rank_fusion,
)
from app.retrieval.models import RetrievedPassage


def passage(
    chunk_id: str,
    rank: int,
    score: float,
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id="doc",
        filename="sample.pdf",
        source_path="/sample.pdf",
        source_format="pdf",
        page_number=1,
        section=None,
        patient_id="SYN-001",
        text=f"text for {chunk_id}",
        rank=rank,
        distance=1 - score,
        similarity=score,
    )


def test_duplicate_chunk_is_fused_once() -> None:
    fused = reciprocal_rank_fusion(
        {
            "vector": [
                passage("shared", 1, 0.8),
                passage("vector-only", 2, 0.7),
            ],
            "bm25": [
                passage("shared", 2, 4.2),
                passage("bm25-only", 1, 5.0),
            ],
        },
        fusion_constant=60,
    )

    ids = [item.passage.chunk_id for item in fused]

    assert ids.count("shared") == 1
    assert fused[0].passage.chunk_id == "shared"
    assert fused[0].method_ranks == {
        "vector": 1,
        "bm25": 2,
    }


def test_top_n_limits_results() -> None:
    fused = reciprocal_rank_fusion(
        {
            "vector": [
                passage("a", 1, 0.9),
                passage("b", 2, 0.8),
                passage("c", 3, 0.7),
            ]
        },
        top_n=2,
    )

    assert len(fused) == 2


def test_ranked_conversion_marks_hybrid_metadata() -> None:
    fused = reciprocal_rank_fusion(
        {
            "vector": [passage("a", 1, 0.9)],
            "bm25": [passage("a", 1, 5.0)],
        }
    )

    [result] = as_ranked_passages(fused)

    assert result.rank == 1
    assert result.metadata["retrieval_method"] == "hybrid"
    assert "fusion_score" in result.metadata


def test_invalid_fusion_constant_is_rejected() -> None:
    try:
        reciprocal_rank_fusion(
            {"vector": [passage("a", 1, 0.9)]},
            fusion_constant=0,
        )
    except ValueError as exc:
        assert "greater than zero" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
