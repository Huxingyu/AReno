"""Small, runtime-neutral contracts for the experimental pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from areno.api.models import RolloutResult, TrainSequence


class PipelineClosed(RuntimeError):
    """Admission was cancelled by a stop or failure."""


class BridgeStateError(RuntimeError):
    """A model operation was attempted outside the usable lifecycle."""


class BatchContractError(ValueError):
    """A batch violates version, shape, or payload ownership contracts."""


class ShutdownTimeout(TimeoutError):
    """Shutdown could not prove that every owned operation had ended."""


class DeviceMode(str, Enum):
    ASYNC = "async"
    SYNC = "sync"


class Lifecycle(str, Enum):
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class AsyncPolicyConfig:
    """Bounds for one completion pipeline (all versions count optimizer steps).

    ``queue_capacity`` (Q) counts ready prompt-group batches, default 2.
    ``max_inflight_rollouts`` (K) counts prompt groups from source read
    through generation, CPU scoring and publication, default 1; GPU sessions
    still serialize. ``rollout_batch_groups``, default 1, optionally packs up
    to that many groups into one session and cannot exceed K. Every group
    retains a permit and its own scoring and optimizer-update boundary.
    ``weight_sync_interval_updates`` (C), default 1, counts
    actual optimizer updates since the last weight copy. Lag can force an
    earlier copy: sync is due at ``min(C, max_policy_lag + 1)`` updates.

    ``max_policy_lag``, default 1, is the inclusive training-admission bound
    ``train_version - batch_version``. Older batches are discarded whole.
    It does not select a loss or provide importance-sampling correction. Zero
    lag still permits speculative generation and stale drops; it does not
    promise the same sample stream or trajectory as a serial sync loop.

    ``max_steps`` counts successful optimizer updates in this run (None means
    no limit, zero means no training). Operation/shutdown timeouts and the
    queue poll interval are seconds, defaulting to 60, 60 and 0.05.
    """

    queue_capacity: int = 2
    max_inflight_rollouts: int = 1
    weight_sync_interval_updates: int = 1
    max_policy_lag: int = 1
    max_steps: int | None = None
    operation_timeout_s: float = 60.0
    shutdown_timeout_s: float = 60.0
    poll_interval_s: float = 0.05
    rollout_batch_groups: int = 1

    def __post_init__(self) -> None:
        for name in ("queue_capacity", "max_inflight_rollouts", "weight_sync_interval_updates", "rollout_batch_groups"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.rollout_batch_groups > self.max_inflight_rollouts:
            raise ValueError("rollout_batch_groups cannot exceed max_inflight_rollouts")
        for name in ("max_policy_lag", "max_steps"):
            value = getattr(self, name)
            if name == "max_steps" and value is None:
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("operation_timeout_s", "shutdown_timeout_s", "poll_interval_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class AsyncPrompt:
    prompt_id: str
    input_tokens: tuple[int, ...]
    prompt: str = ""
    record: dict[str, Any] = field(default_factory=dict)
    solutions: Any = None


@dataclass(frozen=True)
class BatchEnvelope:
    run_id: str
    batch_id: str
    epoch: int
    policy_version: int
    prompt_ids: tuple[str, ...]
    group_offsets: tuple[int, ...]
    sequences: tuple[TrainSequence, ...]

    @property
    def trainable(self) -> bool:
        return any(len(row.tokens) > row.prompt_len for row in self.sequences)


@dataclass(frozen=True)
class RolloutSession:
    policy_version: int
    result: RolloutResult


@dataclass(frozen=True)
class SyncPlan:
    """Move the inference copy from source_version (old Vr) to target_version (Vt)."""

    source_version: int
    target_version: int


@dataclass(frozen=True)
class TrainAdmission:
    decision: str
    batch_version: int
    train_version: int
    rollout_version: int
    lag: int


@dataclass(frozen=True)
class TrainResult:
    admission: TrainAdmission
    stepped: bool
    train_version_after: int
    admission_wait_s: float
    train_s: float


@dataclass(frozen=True)
class SyncMetric:
    source_version: int
    target_version: int
    wait_s: float
    transfer_s: float


@dataclass(frozen=True)
class BatchMetric:
    batch_id: str
    batch_version: int
    train_version: int
    rollout_version: int
    train_version_after: int
    lag: int
    decision: str
    stepped: bool
    wait_s: float
    admission_wait_s: float = 0.0
    train_s: float = 0.0


@dataclass(frozen=True)
class PipelineReport:
    exit_reason: str
    state: str
    failure: str | None
    secondary_failures: tuple[str, ...]
    steps: int
    produced_batches: int
    batches_seen: int
    batches_dropped_stale: int
    batches_dropped_empty: int
    batches_cancelled: int
    queue_max_depth: int
    inflight_peak: int
    producer_alive: bool
    inflight_active: int
    active_rollouts: int
    train_active: bool
    sync_active: bool
    train_policy_version: int
    rollout_policy_version: int
    batch_wait_s: float
    batch_metrics: tuple[BatchMetric, ...]
    sync_metrics: tuple[SyncMetric, ...]


class RolloutEngine(Protocol):
    def initialize(self, *, timeout_s: float | None) -> None: ...

    def generate(self, prompt: AsyncPrompt, version: int, *, timeout_s: float | None) -> RolloutResult: ...

    def close(self, *, timeout_s: float | None) -> None: ...


class BatchedRolloutEngine(RolloutEngine, Protocol):
    def generate_batch(
        self, prompts: tuple[AsyncPrompt, ...], version: int, *, timeout_s: float | None,
    ) -> dict[int, RolloutResult]:
        """Return every group by its input position; dictionary order is irrelevant."""
        ...


class TrainEngine(Protocol):
    def initialize(self, *, timeout_s: float | None) -> None: ...

    def train(self, batch: BatchEnvelope, version: int, *, timeout_s: float | None) -> bool:
        """Perform at most one update; return whether optimizer.step actually ran."""
        ...

    def close(self, *, timeout_s: float | None) -> None: ...


class WeightSync(Protocol):
    def transfer(self, plan: SyncPlan, *, timeout_s: float | None) -> None:
        """Return only after copy, receiver acknowledgement and inference preparation."""
        ...
