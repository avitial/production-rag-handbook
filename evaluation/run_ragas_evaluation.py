"""Prepare or run an optional Ragas evaluation.

Ragas evolves quickly and several metrics require an evaluator LLM and/or
embeddings. This adapter therefore supports:

1. ``--prepare-only``: always offline; validates and exports Ragas-shaped rows.
2. Full evaluation: requires a compatible Ragas installation and configured
   evaluator models.

The stable Ragas concepts used are:
- user_input
- retrieved_contexts
- response
- reference
- EvaluationDataset / SingleTurnSample
- evaluate()

Example offline preparation:

    python evaluation/run_ragas_evaluation.py \
      --prepare-only \
      --input evaluation/output/ragas-input.jsonl

The retrieval runner can be extended to populate actual generated responses
and retrieved contexts. This script never silently invents them.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED_FIELDS.difference(row)
        if missing:
            raise ValueError(
                f"line {number} missing Ragas fields: {sorted(missing)}"
            )
        if not isinstance(row["retrieved_contexts"], list):
            raise ValueError(
                f"line {number}: retrieved_contexts must be a list"
            )
        rows.append(row)
    return rows


def build_dataset(rows):
    """Build a modern Ragas EvaluationDataset when available."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample
    except ImportError as exc:
        raise RuntimeError(
            "Ragas is not installed. Install it or use --prepare-only."
        ) from exc

    return EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=row["user_input"],
                retrieved_contexts=row["retrieved_contexts"],
                response=row["response"],
                reference=row["reference"],
            )
            for row in rows
        ]
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="evaluation/output/ragas-input.jsonl",
    )
    p.add_argument(
        "--output",
        default="evaluation/output/ragas-results.json",
    )
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.input)

    if not path.exists():
        print(
            f"Ragas input does not exist: {path}. "
            "Create rows with user_input, retrieved_contexts, response, "
            "and reference.",
            file=sys.stderr,
        )
        return 2

    rows = load_rows(path)
    if args.prepare_only:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "prepared": True,
                    "sample_count": len(rows),
                    "required_fields": sorted(REQUIRED_FIELDS),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Prepared {len(rows)} Ragas samples.")
        return 0

    try:
        from ragas import evaluate
        # Metric imports are deliberately explicit. These metrics may require
        # evaluator LLM/embedding configuration depending on Ragas version.
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
        )
        dataset = build_dataset(rows)
        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                ResponseRelevancy(),
                LLMContextPrecisionWithoutReference(),
                LLMContextRecall(),
            ],
        )
    except Exception as exc:
        print(
            "Ragas evaluation could not run. Confirm the installed Ragas "
            "version and configure evaluator LLM/embeddings. "
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 3

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # EvaluationResult supports conversion methods in current releases, but
    # fall back to string-safe representation for compatibility.
    if hasattr(result, "to_pandas"):
        payload = result.to_pandas().to_dict(orient="records")
    else:
        payload = {"result": str(result)}
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Ragas results written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
