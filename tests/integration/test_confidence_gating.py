"""Unit tests for confidence feature extraction and gating."""

from app.confidence.features import ConfidenceFeatures
from app.confidence.policy import (
    ConfidenceDecision,
    ConfidencePolicy,
    ConfidencePolicyConfig,
)


def features(**overrides) -> ConfidenceFeatures:
    values = {
        "retrieval_candidate_count": 5,
        "top_retrieval_similarity": 0.7,
        "retrieval_similarity_margin": 0.1,
        "reranked_candidate_count": 3,
        "top_reranker_score": 1.2,
        "reranker_score_margin": 0.2,
        "context_source_count": 3,
        "citation_count": 1,
        "citation_valid": True,
        "answer_grounded": True,
        "json_valid": True,
        "abstained": False,
        "patient_filter_applied": True,
        "invalid_citation_count": 0,
        "validation_issue_count": 0,
    }
    values.update(overrides)
    return ConfidenceFeatures(**values)


def test_accepts_when_all_requirements_pass() -> None:
    assessment = ConfidencePolicy().evaluate(features())

    assert assessment.decision == ConfidenceDecision.ACCEPT
    assert assessment.allowed is True


def test_generator_abstention_stays_abstained() -> None:
    assessment = ConfidencePolicy().evaluate(
        features(
            abstained=True,
            citation_count=0,
        )
    )

    assert assessment.decision == ConfidenceDecision.ABSTAIN
    assert any(
        reason.code == "generator_abstained"
        for reason in assessment.reasons
    )


def test_invalid_citations_are_rejected() -> None:
    assessment = ConfidencePolicy().evaluate(
        features(
            citation_valid=False,
            invalid_citation_count=1,
            validation_issue_count=1,
        )
    )

    assert assessment.decision == ConfidenceDecision.REJECT
    assert any(
        reason.code == "invalid_citations"
        for reason in assessment.reasons
    )


def test_ungrounded_answer_is_rejected() -> None:
    assessment = ConfidencePolicy().evaluate(
        features(
            answer_grounded=False,
            validation_issue_count=1,
        )
    )

    assert assessment.decision == ConfidenceDecision.REJECT


def test_missing_context_causes_abstention() -> None:
    policy = ConfidencePolicy(
        ConfidencePolicyConfig(
            minimum_context_sources=1
        )
    )
    assessment = policy.evaluate(
        features(context_source_count=0)
    )

    assert assessment.decision == ConfidenceDecision.ABSTAIN
    assert any(
        reason.code == "insufficient_context"
        for reason in assessment.reasons
    )


def test_retrieval_threshold_causes_abstention() -> None:
    policy = ConfidencePolicy(
        ConfidencePolicyConfig(
            minimum_top_retrieval_similarity=0.8
        )
    )
    assessment = policy.evaluate(
        features(top_retrieval_similarity=0.4)
    )

    assert assessment.decision == ConfidenceDecision.ABSTAIN
    assert any(
        reason.code == "retrieval_score_below_threshold"
        for reason in assessment.reasons
    )


def test_reranker_threshold_causes_abstention() -> None:
    policy = ConfidencePolicy(
        ConfidencePolicyConfig(
            minimum_top_reranker_score=2.0
        )
    )
    assessment = policy.evaluate(
        features(top_reranker_score=0.5)
    )

    assert assessment.decision == ConfidenceDecision.ABSTAIN


def test_invalid_json_is_rejected() -> None:
    assessment = ConfidencePolicy().evaluate(
        features(
            json_valid=False,
            validation_issue_count=1,
        )
    )

    assert assessment.decision == ConfidenceDecision.REJECT
