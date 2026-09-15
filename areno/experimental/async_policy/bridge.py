"""The only path from the pipeline to engine model operations."""

from __future__ import annotations

import threading
import time

from .contracts import (
    AsyncPolicyConfig,
    AsyncPrompt,
    BatchEnvelope,
    DeviceMode,
    Lifecycle,
    PipelineClosed,
    RolloutEngine,
    RolloutSession,
    ShutdownTimeout,
    SyncMetric,
    TrainEngine,
    TrainResult,
    WeightSync,
)
from .coordination import Deadline, PolicyPipelineCoordinator
from .data import snapshot_rollout
from .lifecycle import Supervisor


class DualEngineBridge:
    def __init__(
        self, *, train_engine: TrainEngine, rollout_engine: RolloutEngine, weight_sync: WeightSync,
        config: AsyncPolicyConfig | None = None, coordinator: PolicyPipelineCoordinator | None = None,
        train_devices: tuple[str, ...] = (), rollout_devices: tuple[str, ...] = (),
        mode: DeviceMode = DeviceMode.ASYNC,
    ):
        mode = DeviceMode(mode)
        if mode == DeviceMode.ASYNC and set(train_devices).intersection(rollout_devices):
            raise ValueError("ASYNC mode requires disjoint model devices")
        self.config = config or AsyncPolicyConfig()
        self.coordinator = coordinator or PolicyPipelineCoordinator()
        self.coordinator.attach(self.config, mode)
        self.supervisor = Supervisor()
        self.supervisor.add_stop_hook(self.coordinator.close)
        self._train_engine = train_engine
        self._rollout_engine = rollout_engine
        self._weight_sync = weight_sync
        self._opened: list[TrainEngine | RolloutEngine] = []
        self._close_lock = threading.Lock()
        for engine in (train_engine, rollout_engine):
            bind_stop_event = getattr(engine, "bind_stop_event", None)
            if callable(bind_stop_event):
                bind_stop_event(self.supervisor.stop_event)

    def initialize(self, *, timeout_s: float | None = None) -> SyncMetric | None:
        deadline = Deadline(self.config.operation_timeout_s if timeout_s is None else timeout_s)
        if not self.supervisor.begin_initialization():
            return None
        try:
            for engine in (self._train_engine, self._rollout_engine):
                self.supervisor.check_running()
                deadline.check()
                # Include partially initialized engines in cleanup.
                self._opened.append(engine)
                engine.initialize(timeout_s=deadline.remaining())
            return self._sync(deadline, force=True)
        except BaseException as exc:
            if not isinstance(exc, PipelineClosed) or not self.supervisor.stop_event.is_set():
                self.supervisor.fail(exc)
            raise
        finally:
            self.supervisor.end_initialization()

    def rollout_session(self, prompt: AsyncPrompt, *, timeout_s: float | None = None) -> RolloutSession:
        self.supervisor.check_model_use()
        deadline = Deadline(self.config.operation_timeout_s if timeout_s is None else timeout_s)
        admitted = False
        try:
            version = self.coordinator.begin_rollout(timeout_s=deadline.remaining())
            admitted = True
            deadline.check()
            result = self._rollout_engine.generate(prompt, version, timeout_s=deadline.remaining())
            # Snapshot before releasing the session: a later generate may reuse its buffers.
            return RolloutSession(version, snapshot_rollout(result, version))
        except BaseException as exc:
            if not isinstance(exc, PipelineClosed) or not self.supervisor.stop_event.is_set():
                self.supervisor.fail(exc)
            raise
        finally:
            if admitted:
                self.coordinator.end_rollout()

    def train(self, batch: BatchEnvelope, *, timeout_s: float | None = None) -> TrainResult:
        self.supervisor.check_model_use()
        deadline = Deadline(self.config.operation_timeout_s if timeout_s is None else timeout_s)
        start = time.monotonic()
        admitted = False
        stepped = False
        try:
            admission = self.coordinator.begin_train(
                batch.policy_version, trainable=batch.trainable, timeout_s=deadline.remaining(),
            )
            admitted = admission.decision == "trained"
            wait_s = time.monotonic() - start
            if not admitted:
                return TrainResult(admission, False, admission.train_version, wait_s, 0.0)
            deadline.check()
            train_start = time.monotonic()
            value = self._train_engine.train(batch, admission.train_version, timeout_s=deadline.remaining())
            if type(value) is not bool:
                raise TypeError("train adapter must return bool: whether one optimizer update occurred")
            stepped = value
            train_s = time.monotonic() - train_start
        except BaseException as exc:
            if not isinstance(exc, PipelineClosed) or not self.supervisor.stop_event.is_set():
                self.supervisor.fail(exc)
            raise
        finally:
            if admitted:
                version_after = self.coordinator.end_train(stepped=stepped)
        return TrainResult(admission, stepped, version_after, wait_s, train_s)

    def _sync(self, deadline: Deadline, *, force: bool = False) -> SyncMetric | None:
        plan = None
        succeeded = False
        start = time.monotonic()
        try:
            plan = self.coordinator.begin_sync(timeout_s=deadline.remaining(), force=force)
            if plan is None:
                return None
            wait_s = time.monotonic() - start
            deadline.check()
            transfer_start = time.monotonic()
            self._weight_sync.transfer(plan, timeout_s=deadline.remaining())
            deadline.check()
            succeeded = True
            return SyncMetric(plan.source_version, plan.target_version, wait_s, time.monotonic() - transfer_start)
        except BaseException as exc:
            if not isinstance(exc, PipelineClosed) or not self.supervisor.stop_event.is_set():
                self.supervisor.fail(exc)
            raise
        finally:
            if plan is not None:
                self.coordinator.end_sync(plan, succeeded=succeeded)

    def sync(self, *, timeout_s: float | None = None) -> SyncMetric | None:
        self.supervisor.check_model_use()
        return self._sync(Deadline(self.config.operation_timeout_s if timeout_s is None else timeout_s))

    def close(self, *, timeout_s: float | None = None) -> None:
        deadline = Deadline(self.config.shutdown_timeout_s if timeout_s is None else timeout_s)
        self.supervisor.stop()
        if not self._close_lock.acquire(timeout=deadline.remaining()):
            error = ShutdownTimeout("another bridge close is still running")
            self.supervisor.fail(error)
            raise error
        try:
            if self.supervisor.state == Lifecycle.CLOSED:
                return
            try:
                self.supervisor.wait_initialization(timeout_s=deadline.remaining())
                self.coordinator.wait_idle(timeout_s=deadline.remaining())
            except TimeoutError as exc:
                raise ShutdownTimeout("model initialization or an active model operation did not stop") from exc
            errors = []
            for engine in tuple(reversed(self._opened)):
                try:
                    deadline.check()
                    engine.close(timeout_s=deadline.remaining())
                    self._opened = [opened for opened in self._opened if opened is not engine]
                    deadline.check()
                except BaseException as exc:
                    errors.append(exc)
            if errors:
                first = errors[0]
                if isinstance(first, TimeoutError):
                    raise ShutdownTimeout("engine cleanup exceeded the shutdown deadline") from first
                raise first
            self.supervisor.closed()
        except BaseException as exc:
            self.supervisor.fail(exc)
            raise
        finally:
            self._close_lock.release()
