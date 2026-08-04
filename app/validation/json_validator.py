"""Validate and serialize API response JSON.

This module checks both general JSON safety and the expected RAG response
shape.

Pseudo-code:

    convert dataclass or mapping to dictionary
    serialize with json.dumps
    deserialize with json.loads
    verify required top-level fields
    verify field types
    verify citation item structure
    return normalized payload or detailed issues
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping
import json


@dataclass(frozen=True)
class JSONValidationIssue:
    code: str
    message: str
    path: str = "$"


@dataclass(frozen=True)
class JSONValidationResult:
    valid: bool
    normalized: dict[str, Any] | None
    json_text: str | None
    issues: tuple[JSONValidationIssue, ...]


_REQUIRED_RAG_FIELDS = {
    "question",
    "answer",
    "citations",
    "abstained",
    "validation",
    "diagnostics",
}


def to_json_safe(value: Any) -> Any:
    """Recursively convert supported Python values to JSON-safe values."""
    if is_dataclass(value):
        return to_json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "isoformat"):
        return value.isoformat()

    raise TypeError(
        f"value of type {type(value).__name__} is not JSON serializable"
    )


def validate_rag_response_json(
    payload: Any,
) -> JSONValidationResult:
    """Validate JSON serialization and required response structure."""
    issues: list[JSONValidationIssue] = []

    try:
        normalized = to_json_safe(payload)
    except (TypeError, ValueError) as exc:
        return JSONValidationResult(
            valid=False,
            normalized=None,
            json_text=None,
            issues=(
                JSONValidationIssue(
                    code="not_json_serializable",
                    message=str(exc),
                ),
            ),
        )

    if not isinstance(normalized, dict):
        issues.append(
            JSONValidationIssue(
                code="root_not_object",
                message="RAG response JSON root must be an object.",
            )
        )
    else:
        missing = sorted(
            _REQUIRED_RAG_FIELDS.difference(normalized)
        )
        for field_name in missing:
            issues.append(
                JSONValidationIssue(
                    code="missing_field",
                    message=f"Required field is missing: {field_name}",
                    path=f"$.{field_name}",
                )
            )

        if "question" in normalized and not isinstance(
            normalized["question"],
            str,
        ):
            issues.append(
                JSONValidationIssue(
                    code="invalid_type",
                    message="question must be a string",
                    path="$.question",
                )
            )

        if "answer" in normalized and not isinstance(
            normalized["answer"],
            str,
        ):
            issues.append(
                JSONValidationIssue(
                    code="invalid_type",
                    message="answer must be a string",
                    path="$.answer",
                )
            )

        if "abstained" in normalized and not isinstance(
            normalized["abstained"],
            bool,
        ):
            issues.append(
                JSONValidationIssue(
                    code="invalid_type",
                    message="abstained must be a boolean",
                    path="$.abstained",
                )
            )

        citations = normalized.get("citations")
        if citations is not None:
            if not isinstance(citations, list):
                issues.append(
                    JSONValidationIssue(
                        code="invalid_type",
                        message="citations must be an array",
                        path="$.citations",
                    )
                )
            else:
                for index, citation in enumerate(citations):
                    path = f"$.citations[{index}]"
                    if not isinstance(citation, dict):
                        issues.append(
                            JSONValidationIssue(
                                code="invalid_type",
                                message="citation must be an object",
                                path=path,
                            )
                        )
                        continue

                    for required in (
                        "source_number",
                        "filename",
                        "page_number",
                    ):
                        if required not in citation:
                            issues.append(
                                JSONValidationIssue(
                                    code="missing_field",
                                    message=(
                                        f"Citation is missing {required}"
                                    ),
                                    path=f"{path}.{required}",
                                )
                            )

    try:
        json_text = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
        )
        round_trip = json.loads(json_text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(
            JSONValidationIssue(
                code="json_round_trip_failed",
                message=str(exc),
            )
        )
        json_text = None
    else:
        if round_trip != normalized:
            issues.append(
                JSONValidationIssue(
                    code="json_round_trip_mismatch",
                    message=(
                        "Serialized and deserialized payloads differ."
                    ),
                )
            )

    return JSONValidationResult(
        valid=not issues,
        normalized=normalized if isinstance(normalized, dict) else None,
        json_text=json_text,
        issues=tuple(issues),
    )
