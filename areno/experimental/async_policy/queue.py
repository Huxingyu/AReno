"""Bounded ready queue and inflight permits for the async trainer (T03).

Two independent primitives:

* :class:`BoundedReadyQueue` -- ``Q`` ready batches, with explicit timeout,
  graceful ``close`` (drain then EOF), ``abort``, and error propagation that is
  not blocked by a full queue.
* :class:`InflightLimiter` -- ``K`` production permits covering the whole
  lifecycle of a production task (generation, scoring, materialize, and waiting
  to put), so the producer never creates unbounded futures or threads.

Both are pure CPU synchronization state. Waiting on the ready queue never holds
a GPU lease or the coordinator lock; that ordering is the caller's contract and
is exercised by the CPU pipeline tests.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class QueueClosed(RuntimeError):
    """Raised when the queue is drained and the producer has closed it (EOF)."""


class QueueAborted(RuntimeError):
    """Raised when the queue was aborted and is no longer usable."""


class QueueFull(TimeoutError):
    """Raised when ``put`` cannot make room within the timeout."""


class QueueEmpty(TimeoutError):
    """Raised when ``get`` cannot obtain a batch within the timeout."""


class InflightClosed(RuntimeError):
    """Raised when acquiring an inflight permit after shutdown."""


class InflightLimitExceeded(TimeoutError):
    """Raised when no inflight permit becomes free within the timeout."""


def _check_timeout(timeout_s: float | None) -> None:
    if timeout_s is not None and timeout_s < 0:
        raise ValueError("timeout_s must be non-negative or None")


@dataclass(frozen=True, slots=True)
class QueuedBatch(Generic[T]):
    """A payload plus the sequence and time at which it entered the queue."""

    payload: T
    sequence_number: int
    enqueued_at: float


class BoundedReadyQueue(Generic[T]):
    """A FIFO queue holding at most ``capacity`` ready batches."""

    def __init__(self, capacity: int, *, name: str = "ready-queue") -> None:
        if int(capacity) < 1:
            raise ValueError(f"queue capacity must be >= 1, got {capacity!r}")
        self._capacity = int(capacity)
        self._name = name
        self._cond = threading.Condition()
        self._items: deque[QueuedBatch[T]] = deque()
        self._sequence = 0
        self._closed = False
        self._aborted = False
        self._error: BaseException | None = None
        self._max_depth = 0
        self._total_put = 0
        self._total_get = 0

    def put(self, item: T, *, timeout_s: float | None = None) -> None:
        """Enqueue ``item``, blocking while the queue is full.

        Fails fast if the queue has been closed, aborted, or marked failed, so a
        full queue never hides a control event.
        """

        _check_timeout(timeout_s)
        with self._cond:
            self._raise_state()
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            while len(self._items) >= self._capacity:
                if not self._wait_remaining(deadline):
                    raise QueueFull(f"{self._name} full: put timed out at capacity {self._capacity}")
                # A control event may have arrived while we were waiting.
                self._raise_state()
            self._sequence += 1
            self._items.append(QueuedBatch(payload=item, sequence_number=self._sequence, enqueued_at=time.monotonic()))
            self._total_put += 1
            self._max_depth = max(self._max_depth, len(self._items))
            self._cond.notify_all()

    def get(self, *, timeout_s: float | None = None) -> T:
        """Dequeue the oldest batch, blocking while the queue is empty."""

        _check_timeout(timeout_s)
        with self._cond:
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            while True:
                if self._error is not None:
                    raise self._error
                if self._items:
                    item = self._items.popleft()
                    self._total_get += 1
                    self._cond.notify_all()
                    return item.payload
                if self._aborted:
                    raise QueueAborted(f"{self._name} was aborted")
                if self._closed:
                    raise QueueClosed(f"{self._name} drained and closed")
                if not self._wait_remaining(deadline):
                    raise QueueEmpty(f"{self._name} empty: get timed out")

    def close(self) -> None:
        """Signal that no more batches will be produced; drain remaining items.

        Idempotent. Consumers keep receiving already-queued batches and get
        :class:`QueueClosed` only once the queue is empty.
        """

        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def abort(self, error: BaseException | None = None) -> None:
        """Make the queue permanently unusable and wake every waiter.

        If ``error`` is given it is re-raised to waiters so the original root
        cause propagates instead of a generic abort.
        """

        with self._cond:
            self._aborted = True
            if error is not None:
                self._error = error
            self._cond.notify_all()

    def fail(self, error: BaseException) -> None:
        """Propagate a pipeline failure to every current and future waiter."""

        self.abort(error)

    @property
    def capacity(self) -> int:
        with self._cond:
            return self._capacity

    @property
    def depth(self) -> int:
        with self._cond:
            return len(self._items)

    @property
    def max_depth(self) -> int:
        with self._cond:
            return self._max_depth

    @property
    def total_put(self) -> int:
        with self._cond:
            return self._total_put

    @property
    def total_get(self) -> int:
        with self._cond:
            return self._total_get

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    @property
    def aborted(self) -> bool:
        with self._cond:
            return self._aborted

    @property
    def error(self) -> BaseException | None:
        with self._cond:
            return self._error

    def _raise_state(self) -> None:
        if self._error is not None:
            raise self._error
        if self._aborted:
            raise QueueAborted(f"{self._name} was aborted")
        if self._closed:
            raise QueueClosed(f"{self._name} was closed")

    def _wait_remaining(self, deadline: float | None) -> bool:
        if deadline is None:
            self._cond.wait()
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        self._cond.wait(remaining)
        return True


class InflightLimiter:
    """A closeable counting semaphore bounding in-flight production tasks."""

    def __init__(self, max_inflight: int, *, name: str = "inflight") -> None:
        if int(max_inflight) < 1:
            raise ValueError(f"max_inflight must be >= 1, got {max_inflight!r}")
        self._max = int(max_inflight)
        self._name = name
        self._cond = threading.Condition()
        self._active = 0
        self._peak = 0
        self._closed = False

    def acquire(self, *, timeout_s: float | None = None) -> None:
        _check_timeout(timeout_s)
        with self._cond:
            if self._closed:
                raise InflightClosed(f"{self._name} limiter is closed")
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            while self._active >= self._max:
                if self._closed:
                    raise InflightClosed(f"{self._name} limiter is closed")
                if not self._wait_remaining(deadline):
                    raise InflightLimitExceeded(
                        f"{self._name}: acquire timed out with {self._active}/{self._max} active"
                    )
            self._active += 1
            self._peak = max(self._peak, self._active)

    def release(self) -> None:
        with self._cond:
            if self._active <= 0:
                raise RuntimeError(f"{self._name}: release without an active permit")
            self._active -= 1
            self._cond.notify_all()

    @contextmanager
    def permit(self, *, timeout_s: float | None = None) -> Iterator[None]:
        """Acquire a permit for the duration of the ``with`` block."""

        self.acquire(timeout_s=timeout_s)
        try:
            yield
        finally:
            self.release()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def max_inflight(self) -> int:
        with self._cond:
            return self._max

    @property
    def active(self) -> int:
        with self._cond:
            return self._active

    @property
    def peak(self) -> int:
        with self._cond:
            return self._peak

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def _wait_remaining(self, deadline: float | None) -> bool:
        if deadline is None:
            self._cond.wait()
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        self._cond.wait(remaining)
        return True
