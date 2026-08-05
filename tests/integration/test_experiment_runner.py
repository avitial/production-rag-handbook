"""End-to-end test for Day 13 experiment execution."""

from pathlib import Path
import csv
import subprocess
import sys


def test_experiment_runner_generates_csv_and_findings(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[2]
    comparison = tmp_path / "comparison.csv"
    findings = tmp_path / "findings.md"
    output = tmp_path / "experiment-output"

    result = subprocess.run(
        [
            sys.executable,
            "evaluation/run_experiments.py",
            "data/samples",
            "--config",
            "configs/experiments/small-chunks-hybrid.json",
            "--output-dir",
            str(output),
            "--comparison-csv",
            str(comparison),
            "--findings",
            str(findings),
            "--reset",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert comparison.exists()
    assert findings.exists()

    with comparison.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "small-chunks-hybrid"
    assert float(rows[0]["mean_reciprocal_rank"]) >= 0
    assert (
        output
        / "small-chunks-hybrid"
        / "results.json"
    ).exists()