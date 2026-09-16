"""Bounded queues and one lock protecting model admission and versions."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

from .contracts import (
    AsyncPolicyConfig,
    BatchContractError,
    BridgeStateError,
    DeviceMode,
    PipelineClosed,
    SyncPlan,
    TrainAdmission,
)

T = TypeVar("T")


class Deadline:
    def __init__(self, timeout_s: float | None):
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("timeout must be nonnegative and finite")
        self.end = None if timeout_s is None else time.monotonic() + timeout_s

    def remaining(self) -> float | None:
        return None if self.end is None else max(0.0, self.end - time.monotonic())

    def check(self) -> None:
        if self.remaining() == 0:
            raise TimeoutError("operation deadline expired")

    def wait(self, condition: threading.Condition) -> None:
        self.check()
        condition.wait(self.remaining())


class QueueEmpty(TimeoutError):
    """An open ready queue has no item within this poll budget."""


class QueueFinished(Exception):
    """EOF has drained all published batches."""


class QueueAborted(PipelineClosed):
    pass


class InflightClosed(PipelineClosed):
    pass


class BoundedReadyQueue(Generic[T]):
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("queue capacity must be a positive integer")
        self._capacity = capacity
        self._cond = threading.Condition()
        self._items: deque[T] = deque()
        self._finished = False
        self._aborted = False
        self._max_depth = 0
        self._published = 0

    @property
    def depth(self) -> int:
        with self._cond:
            return len(self._items)

    @property
    def max_depth(self) -> int:
        with self._cond:
            return self._max_depth

    @property
    def published(self) -> int:
        with self._cond:
            return self._published

    def put(self, item: T, *, timeout_s: float | None = None) -> None:
        deadline = Deadline(timeout_s)
        with self._cond:
            while True:
                if self._aborted or self._finished:
                    raise QueueAborted("ready queue is closed for publication")
                if len(self._items) < self._capacity:
                    self._items.append(item)
                    self._published += 1
                    self._max_depth = max(self._max_depth, len(self._items))
                    self._cond.notify_all()
                    return
                deadline.wait(self._cond)

    def get(self, *, timeout_s: float | None = None) -> T:
        deadline = Deadline(timeout_s)
        with self._cond:
            while True:
                # Cancellation wins over buffered data and over a concurrent EOF.
                if self._aborted:
                    raise QueueAborted("ready queue was aborted")
                if self._items:
                    item = self._items.popleft()
                    self._cond.notify_all()
                    return item
                if self._finished:
                    raise QueueFinished()
                if deadline.remaining() == 0:
                    raise QueueEmpty("ready queue poll timed out")
                self._cond.wait(deadline.remaining())

    def finish(self) -> None:
        with self._cond:
            self._finished = True
            self._cond.notify_all()

    def abort(self) -> None:
        with self._cond:
            self._aborted = True
            self._cond.notify_all()

    def discard_pending(self) -> list[T]:
        """Called by the consumer after stop to account for cancelled batches."""
        with self._cond:
            if not self._aborted:
                raise RuntimeError("abort before discarding pending batches")
            items = list(self._items)
            self._items.clear()
            self._cond.notify_all()
            return items


class InflightLimiter:
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("inflight capacity must be a positive integer")
        self._capacity = capacity
        self._cond = threading.Condition()
        self._closed = False
        self._active = 0
        self._peak = 0

    @property
    def active(self) -> int:
        with self._cond:
            return self._active

    @property
    def peak(self) -> int:
        with self._cond:
            return self._peak

    def acquire(self, *, timeout_s: float | None = None) -> None:
        deadline = Deadline(timeout_s)
        with self._cond:
            while True:
                if self._closed:
                    raise InflightClosed("production admission is closed")
                if self._active < self._capacity:
                    self._active += 1
                    self._peak = max(self._peak, self._active)
                    return
                deadline.wait(self._cond)

    def try_acquire(self) -> bool:
        """Reserve another group without waiting for a batch to fill."""
        with self._cond:
            if self._closed:
                raise InflightClosed("production admission is closed")
            if self._active == self._capacity:
                return False
            self._active += 1
            self._peak = max(self._peak, self._active)
            return True

    def release(self) -> None:
        with self._cond:
            if self._active == 0:
                raise RuntimeError("inflight permit released more than once")
            self._active -= 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


@dataclass(frozen=True)
class CoordinatorSnapshot:
    train_policy_version: int
    rollout_policy_version: int
    active_rollouts: int
    train_active: bool
    sync_active: bool
    sync_pending: bool
    closed: bool


class PolicyPipelineCoordinator:
    def __init__(
        self, *, train_policy_version: int = 0, rollout_policy_version: int = 0,
    ):
        if any(type(v) is not int or v < 0 for v in (train_policy_version, rollout_policy_version)):
            raise ValueError("initial versions must be nonnegative integers")
        if rollout_policy_version > train_policy_version:
            raise ValueError("initial rollout version cannot exceed the training version")
        self._cond = threading.Condition()
        self._config = AsyncPolicyConfig()
        self._mode = DeviceMode.ASYNC
        self._attached = False
        self._closed = False
        self._vt = train_policy_version
        self._vr = rollout_policy_version
        self._rollout_active = False
        self._train_active = False
        self._plan: SyncPlan | None = None
        self._sync_waiters = 0
        self._sync_due = False
        self._has_synced = False

    def attach(self, config: AsyncPolicyConfig, mode: DeviceMode) -> None:
        with self._cond:
            if self._attached or self._closed or self._rollout_active or self._train_active or self._plan:
                raise BridgeStateError("a coordinator can belong to only one unused bridge")
            self._config = config
            self._mode = mode
            self._attached = True

    def _check_open(self) -> None:
        if self._closed:
            raise PipelineClosed("model admission is closed")

    def _sync_pending(self) -> bool:
        return self._sync_due or self._sync_waiters > 0 or self._plan is not None

    def snapshot(self) -> CoordinatorSnapshot:
        with self._cond:
            return CoordinatorSnapshot(
                self._vt, self._vr, int(self._rollout_active), self._train_active,
                self._plan is not None, self._sync_pending(), self._closed,
            )

    @property
    def train_policy_version(self) -> int:
        return self.snapshot().train_policy_version

    @property
    def rollout_policy_version(self) -> int:
        return self.snapshot().rollout_policy_version

    @property
    def active_rollouts(self) -> int:
        return self.snapshot().active_rollouts

    @property
    def train_active(self) -> bool:
        return self.snapshot().train_active

    @property
    def sync_active(self) -> bool:
        return self.snapshot().sync_active

    @property
    def sync_pending(self) -> bool:
        return self.snapshot().sync_pending

    def begin_rollout(self, *, timeout_s: float | None = None) -> int:
        deadline = Deadline(timeout_s)
        with self._cond:
            while True:
                self._check_open()
                blocked = self._sync_pending() or self._rollout_active
                blocked |= self._mode == DeviceMode.SYNC and self._train_active
                if not blocked:
                    self._rollout_active = True
                    return self._vr
                deadline.wait(self._cond)

    def end_rollout(self) -> None:
        with self._cond:
            if not self._rollout_active:
                raise BridgeStateError("no rollout lease to release")
            self._rollout_active = False
            self._cond.notify_all()

    def begin_train(
        self, batch_version: int, *, trainable: bool = True, timeout_s: float | None = None,
    ) -> TrainAdmission:
        deadline = Deadline(timeout_s)
        with self._cond:
            while True:
                self._check_open()
                blocked = self._sync_pending() or self._train_active
                blocked |= self._mode == DeviceMode.SYNC and self._rollout_active
                if not blocked:
                    break
                deadline.wait(self._cond)
            if type(batch_version) is not int or batch_version < 0 or batch_version > self._vt:
                raise BatchContractError("batch policy version is invalid or from the future")
            lag = self._vt - batch_version
            decision = "stale" if lag > self._config.max_policy_lag else "trained" if trainable else "empty"
            self._train_active = decision == "trained"
            return TrainAdmission(decision, batch_version, self._vt, self._vr, lag)

    def end_train(self, *, stepped: bool) -> int:
        with self._cond:
            if not self._train_active:
                raise BridgeStateError("no training lease to release")
            if type(stepped) is not bool:
                raise TypeError("stepped must describe one actual optimizer update")
            if stepped:
                self._vt += 1
                lag = self._vt - self._vr
                # Commit the update AND close new admission in the same critical section.
                self._sync_due |= (
                    lag >= self._config.weight_sync_interval_updates or lag > self._config.max_policy_lag
                )
            self._train_active = False
            self._cond.notify_all()
            return self._vt

    def begin_sync(self, *, timeout_s: float | None = None, force: bool = False) -> SyncPlan | None:
        """Coalesce callers; force ensures an actual initial copy even at Vt == Vr."""
        deadline = Deadline(timeout_s)
        with self._cond:
            self._check_open()
            self._sync_waiters += 1
            self._cond.notify_all()
            try:
                while True:
                    self._check_open()
                    if (self._plan is None and not self._train_active
                            and self._vr == self._vt and (not force or self._has_synced)):
                        return None
                    if self._plan is None and not self._train_active and not self._rollout_active:
                        self._plan = SyncPlan(self._vr, self._vt)
                        return self._plan
                    deadline.wait(self._cond)
            finally:
                self._sync_waiters -= 1
                self._cond.notify_all()

    def end_sync(self, plan: SyncPlan, *, succeeded: bool) -> None:
        with self._cond:
            if self._plan is not plan:
                raise BridgeStateError("only the admitted sync owner may complete a transfer")
            if succeeded:
                self._vr = plan.target_version
                self._has_synced = True
                self._sync_due = False
            else:
                # A partial copy is unusable, regardless of the unchanged version number.
                self._closed = True
            self._plan = None
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def wait_idle(self, *, timeout_s: float | None) -> None:
        deadline = Deadline(timeout_s)
        with self._cond:
            while self._rollout_active or self._train_active or self._plan is not None:
                deadline.wait(self._cond)
