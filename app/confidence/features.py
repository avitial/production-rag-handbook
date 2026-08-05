"""Extract deterministic confidence features from a validated RAG answer.

Confidence is represented as a collection of observable signals rather than a
single unexplained score.

Features include:

- Retrieval candidate count
- Top retrieval similarity
- Similarity margin between ranks 1 and 2
- Top reranker score
- Reranker margin
- Citation count
- Citation validity
- Answer grounding validity
- JSON validity
- Context source count
- Whether the generator abstained
- Whether patient filtering was applied

Pseudo-code:

    inspect retrieval response
    inspect reranking response
    inspect generated answer
    inspect answer and JSON validation results
    calculate score margins safely
    return immutable feature object
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.generation.rag_generator import RAGAnswer
from app.reranking.base import RerankingResponse
from app.retrieval.models import VectorSearchResponse
from app.validation.answer_validator import (
    AnswerValidationResult,
)
from app.validation.json_validator import (
    JSONValidationResult,
)


@dataclass(frozen=True)
class ConfidenceFeatures:
    """Observable signals used by the confidence policy."""

    retrieval_candidate_count: int
    top_retrieval_similarity: float | None
    retrieval_similarity_margin: float | None
    reranked_candidate_count: int
    top_reranker_score: float | None
    reranker_score_margin: float | None
    context_source_count: int
    citation_count: int
    citation_valid: bool
    answer_grounded: bool
    json_valid: bool
    abstained: bool
    patient_filter_applied: bool
    invalid_citation_count: int
    validation_issue_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_candidate_count": self.retrieval_candidate_count,
            "top_retrieval_similarity": self.top_retrieval_similarity,
            "retrieval_similarity_margin": (
                self.retrieval_similarity_margin
            ),
            "reranked_candidate_count": self.reranked_candidate_count,
            "top_reranker_score": self.top_reranker_score,
            "reranker_score_margin": self.reranker_score_margin,
            "context_source_count": self.context_source_count,
            "citation_count": self.citation_count,
            "citation_valid": self.citation_valid,
            "answer_grounded": self.answer_grounded,
            "json_valid": self.json_valid,
            "abstained": self.abstained,
            "patient_filter_applied": self.patient_filter_applied,
            "invalid_citation_count": self.invalid_citation_count,
            "validation_issue_count": self.validation_issue_count,
        }


def _margin(values: list[float]) -> float | None:
    """Return rank-1 minus rank-2, or None when fewer than two values exist."""
    if len(values) < 2:
        return None
    return float(values[0] - values[1])


def extract_confidence_features(
    *,
    retrieval_response: VectorSearchResponse,
    reranking_response: RerankingResponse,
    rag_answer: RAGAnswer,
    answer_validation: AnswerValidationResult,
    json_validation: JSONValidationResult,
) -> ConfidenceFeatures:
    """Extract all policy inputs from one completed RAG request."""
    retrieval_scores = [
        float(item.similarity)
        for item in retrieval_response.results
    ]
    reranker_scores = [
        float(item.rerank_score)
        for item in reranking_response.results
    ]

    filters = retrieval_response.filters
    patient_filter_applied = bool(filters.patient_id)

    return ConfidenceFeatures(
        retrieval_candidate_count=len(
            retrieval_response.results
        ),
        top_retrieval_similarity=(
            retrieval_scores[0] if retrieval_scores else None
        ),
        retrieval_similarity_margin=_margin(
            retrieval_scores
        ),
        reranked_candidate_count=len(
            reranking_response.results
        ),
        top_reranker_score=(
            reranker_scores[0] if reranker_scores else None
        ),
        reranker_score_margin=_margin(
            reranker_scores
        ),
        context_source_count=len(
            rag_answer.context.sources
        ),
        citation_count=len(rag_answer.citations),
        citation_valid=(
            answer_validation.citation_result.valid
        ),
        answer_grounded=answer_validation.grounded,
        json_valid=json_validation.valid,
        abstained=answer_validation.abstained,
        patient_filter_applied=patient_filter_applied,
        invalid_citation_count=len(
            answer_validation.citation_result.invalid_source_numbers
        ),
        validation_issue_count=len(
            answer_validation.issues
        ) + len(json_validation.issues),
    )
