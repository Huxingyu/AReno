"""Policy-version coordination for overlapped rollout and training (T04).

Rollout and training may be active together because they run on separate
engines. Policy synchronization is exclusive: once a sync is requested, new
rollout / train admission is closed *before* waiting for active work to drain,
so continuous submission cannot starve synchronization (the gap in the
``feat/async-policy-scheduler`` prototype, where admission stayed open while the
caller waited for ``active_rollouts == 0``).

The coordinator owns only short critical sections over versions and leases. It
never performs CUDA / RPC / blocking-queue work while holding the lock.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


class PipelineClosed(RuntimeError):
    """Raised when new work is submitted after pipeline shutdown."""


@dataclass(frozen=True, slots=True)
class PolicySyncPlan:
    """One immutable train-to-rollout policy synchronization request."""

    source_version: int
    target_version: int


class PolicyPipelineCoordinator:
    """Coordinate rollout, policy updates, and weight synchronization.

    Invariants:

    * ``train_policy_version`` (Vt) advances only on a real optimizer update.
    * ``rollout_policy_version`` (Vr) advances only on a successful transfer.
    * ``Vr <= Vt`` always holds inside a run.
    * Once ``sync_pending`` is set, no new rollout or train lease is admitted.
    """

    def __init__(self, *, train_policy_version: int = 0, rollout_policy_version: int = 0) -> None:
        if train_policy_version < 0 or rollout_policy_version < 0:
            raise ValueError("policy versions must be non-negative")
        if rollout_policy_version > train_policy_version:
            raise ValueError("rollout policy version cannot exceed train policy version")
        self._cond = threading.Condition()
        self._train_policy_version = train_policy_version
        self._rollout_policy_version = rollout_policy_version
        self._active_rollouts = 0
        self._train_active = False
        self._sync_pending = False
        self._sync_active = False
        self._closed = False

    @property
    def train_policy_version(self) -> int:
        with self._cond:
            return self._train_policy_version

    @property
    def rollout_policy_version(self) -> int:
        with self._cond:
            return self._rollout_policy_version

    @property
    def active_rollouts(self) -> int:
        with self._cond:
            return self._active_rollouts

    @property
    def train_active(self) -> bool:
        with self._cond:
            return self._train_active

    @property
    def sync_pending(self) -> bool:
        with self._cond:
            return self._sync_pending

    @property
    def sync_active(self) -> bool:
        with self._cond:
            return self._sync_active

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def begin_rollout(self, *, timeout_s: float | None = None) -> int:
        """Enter a rollout lease and return the immutable policy version used.

        Admission is blocked while a synchronization is pending or active, so a
        rollout either uses a fully-published version or waits.
        """

        with self._cond:
            try:
                self._wait_until(
                    lambda: not self._sync_pending and not self._sync_active,
                    timeout_s,
                    "begin_rollout",
                )
            except TimeoutError as exc:
                raise TimeoutError("begin_rollout timed out; a policy sync is pending or active") from exc
            self._active_rollouts += 1
            return self._rollout_policy_version

    def end_rollout(self) -> None:
        """Release one rollout lease."""

        with self._cond:
            if self._active_rollouts <= 0:
                raise RuntimeError("no active rollout to end")
            self._active_rollouts -= 1
            self._cond.notify_all()

    def begin_train(self, *, timeout_s: float | None = None) -> int:
        """Enter the single-writer training lease.

        The lease may overlap rollout but never a pending / active sync. The
        returned version is the policy state at the start of the update.
        """

        with self._cond:
            try:
                self._wait_until(
                    lambda: not self._sync_pending and not self._sync_active and not self._train_active,
                    timeout_s,
                    "begin_train",
                )
            except TimeoutError as exc:
                raise TimeoutError("begin_train timed out; another lease or a policy sync is active") from exc
            self._train_active = True
            return self._train_policy_version

    def end_train(self, *, stepped: bool, policy_version: int | None = None) -> int:
        """Release the training lease and commit an optimizer version if stepped."""

        with self._cond:
            if not self._train_active:
                raise RuntimeError("no active training lease to end")
            try:
                if not stepped:
                    if policy_version is not None:
                        raise ValueError("policy_version requires stepped=True")
                    return self._train_policy_version
                next_version = self._train_policy_version + 1 if policy_version is None else policy_version
                if next_version <= self._train_policy_version:
                    raise ValueError("a completed optimizer step must advance the policy version")
                self._train_policy_version = next_version
                return next_version
            finally:
                self._train_active = False
                self._cond.notify_all()

    def begin_sync(self, *, timeout_s: float | None = None) -> PolicySyncPlan | None:
        """Acquire the exclusive synchronization lease when versions differ.

        Admission is closed immediately, then the caller waits for active
        rollout / training leases to drain. Returns ``None`` when the two sides
        already share a version (a no-op: only stale queued batches need
        clearing).
        """

        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative or None")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cond:
            self._wait_until(lambda: not self._sync_active, deadline, "begin_sync")
            self._sync_pending = True
            self._cond.notify_all()
            try:
                self._wait_until(
                    lambda: not self._train_active and self._active_rollouts == 0,
                    deadline,
                    "begin_sync",
                )
            except BaseException:
                self._sync_pending = False
                self._cond.notify_all()
                raise
            if self._rollout_policy_version == self._train_policy_version:
                self._sync_pending = False
                self._cond.notify_all()
                return None
            self._sync_pending = False
            self._sync_active = True
            return PolicySyncPlan(
                source_version=self._rollout_policy_version,
                target_version=self._train_policy_version,
            )

    def end_sync(self, plan: PolicySyncPlan, *, succeeded: bool) -> int:
        """Release the synchronization lease and commit only a successful transfer."""

        with self._cond:
            if not self._sync_active:
                raise RuntimeError("no active policy synchronization to end")
            try:
                expected = PolicySyncPlan(
                    source_version=self._rollout_policy_version,
                    target_version=self._train_policy_version,
                )
                if plan != expected:
                    raise ValueError(f"policy synchronization plan changed: expected {expected}, got {plan}")
                if succeeded:
                    self._rollout_policy_version = plan.target_version
                return self._rollout_policy_version
            finally:
                self._sync_active = False
                self._cond.notify_all()

    def close(self) -> None:
        """Reject new work and unblock threads waiting to acquire a lease."""

        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _wait_until(self, ready: Callable[[], bool], timeout_s: float | None, operation: str) -> None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while not ready():
            self._raise_if_closed()
            if deadline is None:
                self._cond.wait()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{operation} timed out")
            self._cond.wait(remaining)
        self._raise_if_closed()

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise PipelineClosed("policy pipeline is closed")
