"""Structured JSONL logging for RAG requests and confidence decisions.

The logger avoids full-document logging by default and writes one JSON object
per line.

Pseudo-code:

    create event dictionary
    add UTC timestamp and event name
    recursively convert supported values
    append JSON line
    flush output
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json


def _safe(value: Any) -> Any:
    if is_dataclass(value):
        return _safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class StructuredLogger:
    """Append structured events to JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(
        self,
        event_name: str,
        *,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        if not event_name.strip():
            raise ValueError("event_name must not be blank")

        payload = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "event": event_name,
            **_safe(dict(fields or {})),
        }

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")
            handle.flush()

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
