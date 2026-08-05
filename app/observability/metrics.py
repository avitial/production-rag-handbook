"""In-memory counters, gauges, and latency summaries.

This lightweight collector is dependency-free and suitable for tests and local
development. A production service can later export equivalent metrics to
Prometheus, OpenTelemetry, or another monitoring system.

Pseudo-code:

    increment named counters
    set named gauges
    record latency observations
    summarize count/min/max/average
    export a JSON-safe snapshot
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class LatencySummary:
    count: int
    minimum_ms: float | None
    maximum_ms: float | None
    average_ms: float | None


class MetricsCollector:
    """Thread-safe local metrics collector."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._latencies: dict[str, list[float]] = {}

    def increment(
        self,
        name: str,
        amount: int = 1,
    ) -> None:
        if not name.strip():
            raise ValueError("metric name must not be blank")
        if amount < 0:
            raise ValueError("counter increment must not be negative")
        with self._lock:
            self._counters[name] = (
                self._counters.get(name, 0) + amount
            )

    def set_gauge(
        self,
        name: str,
        value: float,
    ) -> None:
        if not name.strip():
            raise ValueError("metric name must not be blank")
        with self._lock:
            self._gauges[name] = float(value)

    def observe_latency(
        self,
        name: str,
        duration_ms: float,
    ) -> None:
        if duration_ms < 0:
            raise ValueError("duration_ms must not be negative")
        with self._lock:
            self._latencies.setdefault(name, []).append(
                float(duration_ms)
            )

    def latency_summary(
        self,
        name: str,
    ) -> LatencySummary:
        with self._lock:
            values = list(self._latencies.get(name, []))

        if not values:
            return LatencySummary(
                count=0,
                minimum_ms=None,
                maximum_ms=None,
                average_ms=None,
            )

        return LatencySummary(
            count=len(values),
            minimum_ms=min(values),
            maximum_ms=max(values),
            average_ms=sum(values) / len(values),
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            latency_names = list(self._latencies)

        latencies = {
            name: self.latency_summary(name).__dict__
            for name in latency_names
        }

        return {
            "counters": counters,
            "gauges": gauges,
            "latencies": latencies,
        }
