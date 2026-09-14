"""Minimal per-batch metrics for the async pipeline (T08a).

The minimal subset the plan requires before G2: run / batch identity, the three
version counters (Vt train, Vr rollout, Vb batch), lag, the consumption
decision, per-batch wait / train time, and the update count. The richer T08b
set (queue peaks, lag distribution, transfer bytes, RSS) extends this record
rather than replacing it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchMetric:
    run_id: str
    batch_id: str
    batch_version: int
    train_version: int
    rollout_version: int
    lag: int
    decision: str
    stepped: bool
    wait_s: float
    train_s: float
    version_after: int


class MetricsCollector:
    """Append-only per-batch metrics; written only by the training writer."""

    def __init__(self) -> None:
        self._records: list[BatchMetric] = []

    def record(self, metric: BatchMetric) -> None:
        self._records.append(metric)

    def records(self) -> tuple[BatchMetric, ...]:
        return tuple(self._records)

    @property
    def updates(self) -> int:
        return sum(1 for record in self._records if record.stepped)

    def sum_by(self, field: str) -> float:
        return sum(float(getattr(record, field)) for record in self._records)
