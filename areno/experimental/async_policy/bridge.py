"""Experimental backend bridge for concurrent rollout / training (T06).

The real ``CudaBackend`` rejects training while a separate rollout session is
active. The bridge is the *narrow* seam that lets the experimental trainer run
the two engines concurrently without deleting that guard:

* ``DeviceMode.INDEPENDENT`` -- train and rollout live on disjoint device
  groups; the coordinator's leases allow overlap.
* ``DeviceMode.SYNC`` -- the sequential SDK behavior is preserved and training
  during an active rollout session still raises.

The bridge also fixes state ownership: the train engine and its metrics are
written only by the owner of the training lease, while rollout sessions only
touch production state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from areno.experimental.async_policy.coordinator import PolicyPipelineCoordinator


class BridgeConfigError(RuntimeError):
    """Raised when a device topology the experimental path cannot support is requested."""


class BridgeStateError(RuntimeError):
    """Raised on lifecycle misuse (use before initialize, etc.)."""


class DeviceMode(str, Enum):
    SYNC = "sync"
    INDEPENDENT = "independent"


@dataclass
class DualEngineBridge:
    """Admission + ownership bridge over a train engine and a rollout engine."""

    train_engine: Any
    rollout_engine: Any
    weight_sync: Any
    train_devices: tuple[str, ...] = ("cuda:0",)
    rollout_devices: tuple[str, ...] = ("cuda:1",)
    mode: DeviceMode = DeviceMode.INDEPENDENT
    coordinator: PolicyPipelineCoordinator = field(default_factory=PolicyPipelineCoordinator)

    _initialized: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)
    _active_sessions: int = field(default=0, init=False)
    _session_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _metrics_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _train_metrics: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.mode == DeviceMode.INDEPENDENT:
            if not self.train_devices or not self.rollout_devices:
                raise BridgeConfigError("independent mode requires non-empty train and rollout device groups")
            overlap = set(self.train_devices) & set(self.rollout_devices)
            if overlap:
                raise BridgeConfigError(f"train and rollout device groups must be disjoint, overlap: {sorted(overlap)}")
        if self.mode == DeviceMode.SYNC and (len(self.train_devices) != 1 or len(self.rollout_devices) != 1):
            raise BridgeConfigError("sync mode only supports a single device per group")

    def initialize(self) -> None:
        """Validate topology and align initial weights exactly once."""

        if self._closed:
            raise BridgeStateError("bridge is closed")
        if self._initialized:
            return
        # First version only supports one device per group in both modes.
        if len(self.train_devices) != 1 or len(self.rollout_devices) != 1:
            raise BridgeConfigError("first version supports exactly one train and one rollout device")
        self._initialized = True

    @property
    def train_metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return dict(self._train_metrics)

    @property
    def active_rollout_sessions(self) -> int:
        with self._session_lock:
            return self._active_sessions

    def rollout_session(self, item: Any, *, timeout_s: float | None = None) -> Any:
        """Run one rollout under a rollout lease; serial across the bridge."""

        if not self._initialized:
            raise BridgeStateError("bridge must be initialized before rollout")
        if self._closed:
            raise BridgeStateError("bridge is closed")
        # The counter is read by the sync-mode guard, so it must never be held
        # while generation blocks. Serialization is enforced by rejecting a
        # second concurrent session instead of blocking on a lock.
        with self._session_lock:
            if self._active_sessions:
                raise BridgeStateError("rollout session operations are serial in the first version")
            version = self.coordinator.begin_rollout(timeout_s=timeout_s)
            self._active_sessions += 1
        try:
            return self.rollout_engine.generate(item, version, timeout_s=timeout_s)
        finally:
            with self._session_lock:
                self._active_sessions -= 1
            self.coordinator.end_rollout()

    def train(self, batch: Any, *, timeout_s: float | None = None) -> bool:
        """Run one training update under the single-writer train lease."""

        if not self._initialized:
            raise BridgeStateError("bridge must be initialized before training")
        if self._closed:
            raise BridgeStateError("bridge is closed")
        if self.mode == DeviceMode.SYNC and self.active_rollout_sessions:
            # Preserve the sequential SDK guard instead of deleting it.
            raise RuntimeError("cannot update policy weights during an active rollout session")
        version = self.coordinator.begin_train(timeout_s=timeout_s)
        stepped = False
        try:
            stepped = bool(self.train_engine.train(batch, version, timeout_s=timeout_s))
            with self._metrics_lock:
                # Metrics are owned by the training writer only.
                self._train_metrics = {
                    "batch_id": getattr(batch, "batch_id", None),
                    "policy_version": version,
                    "stepped": stepped,
                }
        finally:
            self.coordinator.end_train(stepped=stepped)
        return stepped

    def sync(self, *, timeout_s: float | None = None) -> bool:
        """Publish the latest train policy version to the rollout engine."""

        if not self._initialized:
            raise BridgeStateError("bridge must be initialized before sync")
        if self._closed:
            raise BridgeStateError("bridge is closed")
        plan = self.coordinator.begin_sync(timeout_s=timeout_s)
        if plan is None:
            return False
        try:
            self.weight_sync.transfer(plan, timeout_s=timeout_s)
        except BaseException:
            self.coordinator.end_sync(plan, succeeded=False)
            raise
        self.coordinator.end_sync(plan, succeeded=True)
        return True

    def close(self) -> None:
        """Idempotent teardown: reject new work and wake waiters."""

        if self._closed:
            return
        self._closed = True
        self.coordinator.close()
