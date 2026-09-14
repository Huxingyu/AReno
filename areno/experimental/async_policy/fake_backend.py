"""Controllable CPU fake backend for the async pipeline tests (T05).

The fakes are deterministic and *event-driven*: instead of sleeping to create
overlap, a test hands in an :class:`threading.Event` per call and the fake
blocks on it. That makes overlap assertions exact and fast, while the queue,
inflight limiter, coordinator, producer thread and train loop under test stay
real.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from areno.api.models import RolloutResult, RolloutSequence


@dataclass(frozen=True, slots=True)
class AsyncPrompt:
    """A minimal prompt item the CPU pipeline can turn into a ready batch."""

    prompt_id: str
    input_tokens: tuple[int, ...]
    prompt: str = "prompt"
    record: dict[str, Any] = field(default_factory=dict)
    solutions: Any = None


@dataclass
class FakeRolloutEngine:
    n_samples: int = 2
    response_len: int = 2
    latency_s: float = 0.0
    started_events: dict[int, threading.Event] = field(default_factory=dict)
    gates: dict[int, threading.Event] = field(default_factory=dict)
    failures: dict[int, BaseException] = field(default_factory=dict)
    calls: int = 0

    def generate(self, prompt: AsyncPrompt, version: int, *, timeout_s: float | None = None) -> RolloutResult:
        call = self.calls
        self.calls += 1
        started = self.started_events.get(call)
        if started is not None:
            started.set()
        gate = self.gates.get(call)
        if gate is not None and not gate.wait(timeout=timeout_s):
            raise TimeoutError(f"fake rollout call {call} gate timed out")
        if self.latency_s:
            time.sleep(self.latency_s)
        failure = self.failures.get(call)
        if failure is not None:
            raise failure
        base = 100 + call * 10
        sequences = [
            RolloutSequence(
                resp_tokens=[base + sample],
                resp_logprobs=[-0.1 * (sample + 1)],
            )
            for sample in range(self.n_samples)
        ]
        return RolloutResult(sequences=sequences)


@dataclass
class FakeTrainEngine:
    latency_s: float = 0.0
    started_events: dict[int, threading.Event] = field(default_factory=dict)
    gates: dict[int, threading.Event] = field(default_factory=dict)
    failures: dict[int, BaseException] = field(default_factory=dict)
    stepped_by_call: dict[int, bool] = field(default_factory=dict)
    calls: int = 0

    def train(self, batch: Any, version: int, *, timeout_s: float | None = None) -> bool:
        call = self.calls
        self.calls += 1
        started = self.started_events.get(call)
        if started is not None:
            started.set()
        gate = self.gates.get(call)
        if gate is not None and not gate.wait(timeout=timeout_s):
            raise TimeoutError(f"fake train call {call} gate timed out")
        if self.latency_s:
            time.sleep(self.latency_s)
        failure = self.failures.get(call)
        if failure is not None:
            raise failure
        return bool(self.stepped_by_call.get(call, True))


@dataclass
class FakeWeightSync:
    latency_s: float = 0.0
    started_events: dict[int, threading.Event] = field(default_factory=dict)
    gates: dict[int, threading.Event] = field(default_factory=dict)
    failures: dict[int, BaseException] = field(default_factory=dict)
    calls: int = 0
    transferred: list[tuple[int, int]] = field(default_factory=list)

    def transfer(self, plan: Any, *, timeout_s: float | None = None) -> None:
        call = self.calls
        self.calls += 1
        started = self.started_events.get(call)
        if started is not None:
            started.set()
        gate = self.gates.get(call)
        if gate is not None and not gate.wait(timeout=timeout_s):
            raise TimeoutError(f"fake weight sync call {call} gate timed out")
        if self.latency_s:
            time.sleep(self.latency_s)
        failure = self.failures.get(call)
        if failure is not None:
            raise failure
        self.transferred.append((plan.source_version, plan.target_version))


RewardFn = Callable[[AsyncPrompt, int], float]
