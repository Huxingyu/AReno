"""One supervisor shared by the pipeline and its model bridge."""

from __future__ import annotations

import threading
from collections.abc import Callable

from .contracts import BridgeStateError, Lifecycle, PipelineClosed
from .coordination import Deadline


class Supervisor:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._state = Lifecycle.NEW
        self._failure: BaseException | None = None
        self._secondary_failures: list[BaseException] = []
        self._exit_reason = "not_started"
        self._initializing = False
        self._stop_hooks: list[Callable[[], None]] = []
        self.stop_event = threading.Event()

    @property
    def state(self) -> Lifecycle:
        with self._cond:
            return self._state

    @property
    def failure(self) -> BaseException | None:
        with self._cond:
            return self._failure

    @property
    def exit_reason(self) -> str:
        with self._cond:
            return self._exit_reason

    @property
    def secondary_failures(self) -> tuple[str, ...]:
        with self._cond:
            return tuple(f"{type(error).__name__}: {error}" for error in self._secondary_failures)

    def add_stop_hook(self, hook: Callable[[], None]) -> None:
        with self._cond:
            self._stop_hooks.append(hook)
            stopped = self.stop_event.is_set()
        if stopped:
            hook()

    def check_running(self) -> None:
        if self.stop_event.is_set():
            raise PipelineClosed("run is stopping")

    def check_model_use(self) -> None:
        with self._cond:
            self.check_running()
            if self._state not in (Lifecycle.READY, Lifecycle.RUNNING, Lifecycle.DRAINING):
                raise BridgeStateError("initialize the bridge before using a model")

    def begin_initialization(self) -> bool:
        with self._cond:
            self.check_running()
            if self._state in (Lifecycle.READY, Lifecycle.RUNNING, Lifecycle.DRAINING):
                return False
            if self._state != Lifecycle.NEW:
                raise BridgeStateError("bridge initialization is already in progress")
            self._initializing = True
            self._state = Lifecycle.INITIALIZING
            self._exit_reason = "initializing"
            return True

    def end_initialization(self) -> None:
        with self._cond:
            self._initializing = False
            if not self.stop_event.is_set():
                self._state = Lifecycle.READY
            self._cond.notify_all()

    def wait_initialization(self, *, timeout_s: float | None) -> None:
        deadline = Deadline(timeout_s)
        with self._cond:
            while self._initializing:
                deadline.wait(self._cond)

    def start_run(self) -> None:
        with self._cond:
            self.check_running()
            if self._state != Lifecycle.READY:
                raise BridgeStateError("a pipeline can run only once")
            self._state = Lifecycle.RUNNING
            self._exit_reason = "running"

    def drain(self) -> None:
        with self._cond:
            if self._state == Lifecycle.RUNNING:
                self._state = Lifecycle.DRAINING

    def stop(self, reason: str = "stopped") -> None:
        with self._cond:
            if not self.stop_event.is_set():
                self._exit_reason = reason
                self._state = Lifecycle.STOPPING
            self.stop_event.set()
            hooks = tuple(self._stop_hooks)
        for hook in hooks:
            hook()

    def fail(self, error: BaseException) -> None:
        with self._cond:
            if self._failure is None:
                self._failure = error
            elif error is not self._failure and all(error is not previous for previous in self._secondary_failures):
                self._secondary_failures.append(error)
            self._state = Lifecycle.FAILED
            self._exit_reason = "failed"
            self.stop_event.set()
        self.stop("failed")

    def closed(self) -> None:
        with self._cond:
            self._state = Lifecycle.CLOSED
            self._cond.notify_all()
