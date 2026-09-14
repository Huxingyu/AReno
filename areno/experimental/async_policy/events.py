"""Thread-safe event timeline used as the G1 evidence artifact (T05).

The timeline is the difference between "two threads existed" and "two stages
were demonstrably active at the same time". Each stage emits ``start``/``end``
events; :meth:`EventTimeline.overlaps` intersects the resulting intervals, so a
test can assert real concurrency instead of trusting a sleep budget.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    stage: str
    action: str
    timestamp: float
    batch_id: str | None = None
    detail: tuple[tuple[str, Any], ...] = field(default_factory=tuple)


def intervals_from_events(events: Iterable[TimelineEvent], stage: str) -> list[tuple[float, float]]:
    """Pair ``start``/``end`` events for ``stage`` into closed intervals.

    Nested/concurrent starts on the same stage are handled with a stack, so a
    never-ended start is simply omitted rather than corrupting the list.
    """

    open_stack: list[float] = []
    result: list[tuple[float, float]] = []
    for event in events:
        if event.stage != stage:
            continue
        if event.action == "start":
            open_stack.append(event.timestamp)
        elif event.action == "end" and open_stack:
            result.append((open_stack.pop(), event.timestamp))
    return result


def overlaps_events(events: Iterable[TimelineEvent], stage_a: str, stage_b: str) -> bool:
    """Return True if any interval of ``stage_a`` intersects ``stage_b``."""

    events = tuple(events)
    intervals_a = intervals_from_events(events, stage_a)
    intervals_b = intervals_from_events(events, stage_b)
    for start_a, end_a in intervals_a:
        for start_b, end_b in intervals_b:
            if start_a < end_b and start_b < end_a:
                return True
    return False


class EventTimeline:
    """A bounded, append-only record of stage start/end events."""

    def __init__(self, capacity: int = 8192) -> None:
        if int(capacity) < 1:
            raise ValueError(f"timeline capacity must be >= 1, got {capacity!r}")
        self._lock = threading.Lock()
        self._events: deque[TimelineEvent] = deque(maxlen=int(capacity))
        self._dropped = 0

    def record(self, stage: str, action: str, *, batch_id: str | None = None, **detail: Any) -> TimelineEvent:
        event = TimelineEvent(
            stage=stage,
            action=action,
            timestamp=time.monotonic(),
            batch_id=batch_id,
            detail=tuple(sorted(detail.items())),
        )
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self._dropped += 1
            self._events.append(event)
        return event

    def events(self) -> tuple[TimelineEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def intervals(self, stage: str) -> list[tuple[float, float]]:
        return intervals_from_events(self.events(), stage)

    def overlaps(self, stage_a: str, stage_b: str) -> bool:
        return overlaps_events(self.events(), stage_a, stage_b)
