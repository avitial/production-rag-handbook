"""End-to-end test for the Day 12 retrieval evaluation runner."""

from pathlib import Path
import json
import subprocess
import sys


def test_retrieval_evaluation_script_runs(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    json_output = tmp_path / "results.json"
    md_output = tmp_path / "results.md"

    result = subprocess.run(
        [
            sys.executable,
            "evaluation/run_retrieval_evaluation.py",
            "data/samples",
            "--embedding-backend", "hash",
            "--storage-backend", "local",
            "--reset",
            "--chroma-dir", str(tmp_path / "chroma"),
            "--registry", str(tmp_path / "registry.sqlite3"),
            "--json-output", str(json_output),
            "--markdown-output", str(md_output),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert json_output.exists()
    assert md_output.exists()
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["summary"]["question_count"] == 8
    assert payload["ingestion"]["failed_files"] == 0
    assert "mean_reciprocal_rank" in payload["summary"]
