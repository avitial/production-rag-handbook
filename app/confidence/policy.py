"""Confidence gating policy for deciding whether to return or reject an answer.

The policy uses explicit rules instead of treating one similarity value as
answer confidence.

Decision outcomes:

- ACCEPT: return the answer
- ABSTAIN: return a safe evidence-insufficient response
- REJECT: internal validation failed; do not return the generated answer

Pseudo-code:

    if JSON or citation validation failed:
        reject
    if generator already abstained:
        abstain
    if grounding failed:
        reject
    if no context or no citations:
        abstain
    if configured score thresholds are not met:
        abstain
    otherwise:
        accept
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.confidence.features import ConfidenceFeatures


class ConfidenceDecision(StrEnum):
    ACCEPT = "accept"
    ABSTAIN = "abstain"
    REJECT = "reject"


@dataclass(frozen=True)
class ConfidencePolicyConfig:
    """Thresholds and hard requirements for confidence gating."""

    minimum_context_sources: int = 1
    minimum_citations: int = 1
    minimum_top_retrieval_similarity: float | None = None
    minimum_top_reranker_score: float | None = None
    require_valid_citations: bool = True
    require_grounded_answer: bool = True
    require_valid_json: bool = True

    def __post_init__(self) -> None:
        if self.minimum_context_sources < 0:
            raise ValueError(
                "minimum_context_sources must not be negative"
            )
        if self.minimum_citations < 0:
            raise ValueError(
                "minimum_citations must not be negative"
            )


@dataclass(frozen=True)
class ConfidenceReason:
    """One human-readable and machine-readable policy reason."""

    code: str
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Final confidence-gating result."""

    decision: ConfidenceDecision
    reasons: tuple[ConfidenceReason, ...]
    features: ConfidenceFeatures

    @property
    def allowed(self) -> bool:
        return self.decision == ConfidenceDecision.ACCEPT


class ConfidencePolicy:
    """Evaluate extracted confidence features."""

    def __init__(
        self,
        config: ConfidencePolicyConfig | None = None,
    ) -> None:
        self.config = config or ConfidencePolicyConfig()

    def evaluate(
        self,
        features: ConfidenceFeatures,
    ) -> ConfidenceAssessment:
        """Apply hard validation rules before score thresholds."""
        reject_reasons: list[ConfidenceReason] = []
        abstain_reasons: list[ConfidenceReason] = []

        if (
            self.config.require_valid_json
            and not features.json_valid
        ):
            reject_reasons.append(
                ConfidenceReason(
                    code="invalid_json",
                    message="Structured JSON validation failed.",
                    details={},
                )
            )

        if (
            self.config.require_valid_citations
            and not features.citation_valid
        ):
            reject_reasons.append(
                ConfidenceReason(
                    code="invalid_citations",
                    message="Citation validation failed.",
                    details={
                        "invalid_citation_count": (
                            features.invalid_citation_count
                        )
                    },
                )
            )

        if (
            self.config.require_grounded_answer
            and not features.answer_grounded
        ):
            reject_reasons.append(
                ConfidenceReason(
                    code="ungrounded_answer",
                    message="Answer grounding validation failed.",
                    details={},
                )
            )

        if features.validation_issue_count > 0:
            reject_reasons.append(
                ConfidenceReason(
                    code="validation_issues",
                    message=(
                        "One or more validation issues were reported."
                    ),
                    details={
                        "count": features.validation_issue_count
                    },
                )
            )

        if reject_reasons:
            return ConfidenceAssessment(
                decision=ConfidenceDecision.REJECT,
                reasons=tuple(reject_reasons),
                features=features,
            )

        if features.abstained:
            abstain_reasons.append(
                ConfidenceReason(
                    code="generator_abstained",
                    message=(
                        "The generator did not find enough explicit "
                        "evidence."
                    ),
                    details={},
                )
            )

        if (
            features.context_source_count
            < self.config.minimum_context_sources
        ):
            abstain_reasons.append(
                ConfidenceReason(
                    code="insufficient_context",
                    message="Too few source passages were available.",
                    details={
                        "actual": features.context_source_count,
                        "required": (
                            self.config.minimum_context_sources
                        ),
                    },
                )
            )

        if (
            not features.abstained
            and features.citation_count
            < self.config.minimum_citations
        ):
            abstain_reasons.append(
                ConfidenceReason(
                    code="insufficient_citations",
                    message="Too few citations support the answer.",
                    details={
                        "actual": features.citation_count,
                        "required": self.config.minimum_citations,
                    },
                )
            )

        threshold = (
            self.config.minimum_top_retrieval_similarity
        )
        if threshold is not None:
            actual = features.top_retrieval_similarity
            if actual is None or actual < threshold:
                abstain_reasons.append(
                    ConfidenceReason(
                        code="retrieval_score_below_threshold",
                        message=(
                            "Top retrieval score is below the "
                            "configured threshold."
                        ),
                        details={
                            "actual": actual,
                            "required": threshold,
                        },
                    )
                )

        reranker_threshold = (
            self.config.minimum_top_reranker_score
        )
        if reranker_threshold is not None:
            actual = features.top_reranker_score
            if actual is None or actual < reranker_threshold:
                abstain_reasons.append(
                    ConfidenceReason(
                        code="reranker_score_below_threshold",
                        message=(
                            "Top reranker score is below the "
                            "configured threshold."
                        ),
                        details={
                            "actual": actual,
                            "required": reranker_threshold,
                        },
                    )
                )

        if abstain_reasons:
            return ConfidenceAssessment(
                decision=ConfidenceDecision.ABSTAIN,
                reasons=tuple(abstain_reasons),
                features=features,
            )

        return ConfidenceAssessment(
            decision=ConfidenceDecision.ACCEPT,
            reasons=(
                ConfidenceReason(
                    code="all_requirements_met",
                    message=(
                        "Validation and confidence requirements passed."
                    ),
                    details={},
                ),
            ),
            features=features,
        )
