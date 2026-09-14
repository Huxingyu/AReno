"""Experiment config for the async policy trainer (T01/T05).

These are the three independent capacity concepts from the plan plus the
timeouts and stop semantics. Everything is validated at construction so a
mistyped unit fails loudly instead of silently changing behavior.

* ``queue_capacity``     -- Q: ready batches waiting to be consumed.
* ``max_inflight_rollouts`` -- K: production tasks accepted but not yet enqueued.
* ``weight_sync_interval_updates`` -- C: normal successes between syncs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AsyncPolicyConfig:
    queue_capacity: int = 2
    max_inflight_rollouts: int = 1
    weight_sync_interval_updates: int = 1
    max_policy_lag: int = 1
    max_steps: int | None = None
    poll_interval_s: float = 0.05
    operation_timeout_s: float = 30.0
    shutdown_timeout_s: float = 10.0
    timeline_capacity: int = 8192

    def __post_init__(self) -> None:
        if int(self.queue_capacity) < 1:
            raise ValueError(f"queue_capacity (Q) must be >= 1, got {self.queue_capacity!r}")
        if int(self.max_inflight_rollouts) < 1:
            raise ValueError(f"max_inflight_rollouts (K) must be >= 1, got {self.max_inflight_rollouts!r}")
        if int(self.weight_sync_interval_updates) < 1:
            raise ValueError(
                f"weight_sync_interval_updates (C) must be >= 1, got {self.weight_sync_interval_updates!r}"
            )
        if int(self.max_policy_lag) < 0:
            raise ValueError(f"max_policy_lag must be >= 0, got {self.max_policy_lag!r}")
        if self.max_steps is not None and int(self.max_steps) < 1:
            raise ValueError(f"max_steps must be >= 1 or None, got {self.max_steps!r}")
        if self.poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be > 0, got {self.poll_interval_s!r}")
        for name in ("operation_timeout_s", "shutdown_timeout_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
