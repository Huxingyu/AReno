"""CPU closed-loop async policy pipeline (T05, G1).

This wires the real primitives -- :class:`BoundedReadyQueue`,
:class:`InflightLimiter`, :class:`PolicyPipelineCoordinator` -- to a genuine
background producer thread and a foreground (main-thread) train loop:

    producer thread                         main thread
    ---------------                         -----------
    acquire inflight permit (K)
    begin_rollout() -> Vb
      fake rollout generate
    end_rollout()
      CPU reward + materialize
    queue.put(envelope)  ---------------->  queue.get()
                                            lag check vs Vt
                                            begin_train()/train/end_train
                                            advance Vt on stepped
                                            periodic / lag-triggered sync

The backend is a fake, but the queue, limiter, coordinator, threads, lifecycle
and event timeline are the real implementation. Overlap is proven from the
timeline, not from sleeps.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from areno.api.models import RolloutResult
from areno.experimental.async_policy.batch import BatchContractError, build_batch_envelope
from areno.experimental.async_policy.config import AsyncPolicyConfig
from areno.experimental.async_policy.coordinator import PolicyPipelineCoordinator
from areno.experimental.async_policy.events import EventTimeline, TimelineEvent, overlaps_events
from areno.experimental.async_policy.fake_backend import AsyncPrompt, RewardFn
from areno.experimental.async_policy.metrics import BatchMetric, MetricsCollector
from areno.experimental.async_policy.queue import (
    BoundedReadyQueue,
    InflightLimiter,
    QueueClosed,
    QueueEmpty,
)

_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class PipelineReport:
    run_id: str
    exit_reason: str
    steps: int
    batches_seen: int
    batches_trained: int
    batches_dropped_stale: int
    syncs: int
    trained_response_tokens: int
    produced_batches: int
    queue_max_depth: int
    inflight_peak: int
    producer_error: BaseException | None
    timeline: tuple[TimelineEvent, ...]
    batch_metrics: tuple[BatchMetric, ...] = ()

    def overlaps(self, stage_a: str, stage_b: str) -> bool:
        return overlaps_events(self.timeline, stage_a, stage_b)


class AsyncPolicyPipeline:
    """Run one async GRPO-style training job on CPU fakes."""

    def __init__(
        self,
        *,
        config: AsyncPolicyConfig,
        data_source: Sequence[AsyncPrompt] | Iterable[AsyncPrompt],
        rollout_engine: Any,
        train_engine: Any,
        weight_sync: Any,
        reward_fn: RewardFn,
        run_id: str = "async-run",
        eos_token_id: int = 2,
        coordinator: PolicyPipelineCoordinator | None = None,
    ) -> None:
        self._config = config
        self._data = data_source
        self._rollout_engine = rollout_engine
        self._train_engine = train_engine
        self._weight_sync = weight_sync
        self._reward_fn = reward_fn
        self._run_id = run_id
        self._eos_token_id = eos_token_id

        self._coordinator = coordinator or PolicyPipelineCoordinator()
        self._queue: BoundedReadyQueue[Any] = BoundedReadyQueue(config.queue_capacity, name=f"{run_id}-ready")
        self._inflight = InflightLimiter(config.max_inflight_rollouts, name=f"{run_id}-inflight")
        self._timeline = EventTimeline(config.timeline_capacity)

        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._producer: threading.Thread | None = None
        self._producer_done = False
        self._producer_error: BaseException | None = None
        self._producer_failed = threading.Event()
        # One rollout engine is shared by at most K production tasks; the plan
        # requires its session operations to stay serial on the first version.
        self._session_lock = threading.Lock()
        self._batch_counter = itertools.count()
        self._produced_lock = threading.Lock()
        self._produced = 0

        self._steps = 0
        self._batches_seen = 0
        self._batches_trained = 0
        self._dropped_stale = 0
        self._syncs = 0
        self._trained_tokens = 0
        self._metrics = MetricsCollector()
        self._sync_requested = False
        self._exit_reason = "not_started"
        self._report: PipelineReport | None = None

    # ---------------------------------------------------------------- control
    def request_stop(self) -> None:
        """Ask the producer and train loop to stop, waking any blocked waiter."""

        self._stop.set()
        self._queue.close()
        self._inflight.close()

    def close(self) -> None:
        """Stop and join the producer. Idempotent; safe to call twice."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self.request_stop()
        self._join_producer()

    def _join_producer(self) -> None:
        producer = self._producer
        if producer is not None and producer.is_alive():
            producer.join(timeout=self._config.shutdown_timeout_s)

    # ------------------------------------------------------------------- run
    def run(self) -> PipelineReport:
        with self._state_lock:
            if self._started:
                raise RuntimeError("pipeline already started")
            if self._closed:
                raise RuntimeError("pipeline is closed")
            self._started = True

        self._timeline.record("run", "start", exit_reason=None)
        failure: BaseException | None = None
        try:
            self._producer = threading.Thread(
                target=self._producer_main,
                name=f"{self._run_id}-producer",
                daemon=True,
            )
            self._producer.start()
            self._train_loop()
        except BaseException as exc:  # noqa: BLE001 - propagate the root cause
            failure = exc
        finally:
            self.request_stop()
            self._join_producer()
        self._timeline.record("run", "end", exit_reason=self._exit_reason)
        self._report = self._build_report()

        if failure is not None:
            raise failure
        return self._report

    def report(self) -> PipelineReport:
        if self._report is None:
            raise RuntimeError("pipeline has not finished a run yet")
        return self._report

    @property
    def producer_alive(self) -> bool:
        return self._producer is not None and self._producer.is_alive()

    @property
    def inflight_active(self) -> int:
        return self._inflight.active

    # ------------------------------------------------------------- producer
    def _producer_main(self) -> None:
        source = iter(self._data)
        executor = ThreadPoolExecutor(
            max_workers=self._config.max_inflight_rollouts,
            thread_name_prefix=f"{self._run_id}-produce",
        )
        try:
            while not self._stop.is_set() and not self._producer_failed.is_set():
                self._inflight.acquire(timeout_s=self._config.operation_timeout_s)
                if self._stop.is_set() or self._producer_failed.is_set():
                    self._inflight.release()
                    break
                item = next(source, _SENTINEL)
                if item is _SENTINEL:
                    self._inflight.release()
                    break
                # The task owns the permit until it enqueues or fails; the
                # bounded executor caps how many futures exist at once.
                executor.submit(self._production_task, item)
        except BaseException as exc:  # noqa: BLE001 - route to the consumer
            if not self._stop.is_set():
                self._producer_error = exc
                self._producer_failed.set()
                self._queue.fail(exc)
        finally:
            executor.shutdown(wait=True)
            self._producer_done = True
            self._queue.close()

    def _production_task(self, item: AsyncPrompt) -> None:
        try:
            self._produce_one(item)
        except BaseException as exc:  # noqa: BLE001 - route to the consumer
            if not self._stop.is_set():
                self._producer_error = exc
                self._producer_failed.set()
                self._queue.fail(exc)
        finally:
            self._inflight.release()

    def _produce_one(self, item: AsyncPrompt) -> None:
        # Rollout session operations stay serial across production tasks.
        with self._session_lock:
            self._timeline.record("rollout", "start", batch_id=item.prompt_id)
            version = self._coordinator.begin_rollout(timeout_s=self._config.operation_timeout_s)
            try:
                rollout: RolloutResult = self._rollout_engine.generate(
                    item, version, timeout_s=self._config.operation_timeout_s
                )
            finally:
                self._coordinator.end_rollout()
                self._timeline.record("rollout", "end", batch_id=item.prompt_id, version=version)

        self._timeline.record("materialize", "start", batch_id=item.prompt_id)
        try:
            rewards = [float(self._reward_fn(item, sample)) for sample in range(len(rollout.sequences))]
            batch = build_batch_envelope(
                run_id=self._run_id,
                batch_id=f"{self._run_id}:{next(self._batch_counter)}",
                epoch=0,
                policy_version=version,
                prompt_items=[item],
                rollout_results=[rollout],
                rewards=rewards,
                eos_token_id=self._eos_token_id,
                produced_at=None,
            )
        finally:
            self._timeline.record("materialize", "end", batch_id=item.prompt_id)

        with self._produced_lock:
            self._produced += 1
        if self._stop.is_set():
            return
        self._queue.put(batch)
        self._timeline.record("enqueue", "done", batch_id=batch.batch_id, version=version)

    # ------------------------------------------------------------ train loop
    def _train_loop(self) -> None:
        updates_since_sync = 0
        exit_reason = "stopped"
        while True:
            if self._stop.is_set():
                exit_reason = "stopped"
                break
            if self._config.max_steps is not None and self._steps >= self._config.max_steps:
                exit_reason = "max_steps"
                break
            if self._sync_requested or updates_since_sync >= self._config.weight_sync_interval_updates:
                self._do_sync()
                updates_since_sync = 0
                self._sync_requested = False

            wait_start = time.monotonic()
            try:
                batch = self._queue.get(timeout_s=self._config.poll_interval_s)
            except QueueEmpty:
                continue
            except QueueClosed:
                # A closed queue means either "producer finished" (data
                # exhausted) or "we are shutting down". Distinguish them so a
                # requested stop is reported as such instead of looking like a
                # clean end of data.
                exit_reason = "stopped" if self._stop.is_set() else "data_exhausted"
                break
            wait_s = time.monotonic() - wait_start

            self._batches_seen += 1
            if self._consume_batch(batch, wait_s=wait_s):
                updates_since_sync += 1

        self._exit_reason = exit_reason

    def _consume_batch(self, batch: Any, *, wait_s: float) -> bool:
        vt = self._coordinator.train_policy_version
        vrr = self._coordinator.rollout_policy_version
        vb = batch.policy_version
        if vb > vt:
            raise BatchContractError(f"batch {batch.batch_id} version {vb} is ahead of current {vt}")
        if vt - vb > self._config.max_policy_lag:
            self._record_metric(batch, vt, vrr, decision="dropped_stale", stepped=False, wait_s=wait_s, train_s=0.0)
            self._dropped_stale += 1
            self._sync_requested = True
            self._timeline.record("drop", "stale", batch_id=batch.batch_id, version=vb, current=vt)
            return False

        admitted_version = self._coordinator.begin_train(timeout_s=self._config.operation_timeout_s)
        stepped = False
        train_start = time.monotonic()
        try:
            if self._coordinator.train_policy_version - vb > self._config.max_policy_lag:
                self._record_metric(
                    batch, admitted_version, vrr, decision="dropped_stale", stepped=False, wait_s=wait_s, train_s=0.0
                )
                self._dropped_stale += 1
                self._sync_requested = True
                self._timeline.record("drop", "stale", batch_id=batch.batch_id, version=vb, current=vt)
                return False
            self._timeline.record("train", "start", batch_id=batch.batch_id, version=vb)
            try:
                stepped = bool(
                    self._train_engine.train(batch, admitted_version, timeout_s=self._config.operation_timeout_s)
                )
            finally:
                self._timeline.record("train", "end", batch_id=batch.batch_id)
        finally:
            self._coordinator.end_train(stepped=stepped)

        train_s = time.monotonic() - train_start
        if stepped:
            self._steps += 1
            self._batches_trained += 1
            self._trained_tokens += batch.response_token_count
        self._record_metric(
            batch, admitted_version, vrr, decision="trained", stepped=stepped, wait_s=wait_s, train_s=train_s
        )
        return stepped

    def _record_metric(
        self,
        batch: Any,
        train_version: int,
        rollout_version: int,
        *,
        decision: str,
        stepped: bool,
        wait_s: float,
        train_s: float,
    ) -> None:
        self._metrics.record(
            BatchMetric(
                run_id=self._run_id,
                batch_id=batch.batch_id,
                batch_version=batch.policy_version,
                train_version=train_version,
                rollout_version=rollout_version,
                lag=train_version - batch.policy_version,
                decision=decision,
                stepped=stepped,
                wait_s=wait_s,
                train_s=train_s,
                version_after=self._coordinator.train_policy_version,
            )
        )

    def _do_sync(self) -> None:
        plan = self._coordinator.begin_sync(timeout_s=self._config.operation_timeout_s)
        if plan is None:
            return
        self._timeline.record("sync", "start", source=plan.source_version, target=plan.target_version)
        try:
            self._weight_sync.transfer(plan, timeout_s=self._config.operation_timeout_s)
        except BaseException:
            self._coordinator.end_sync(plan, succeeded=False)
            self._timeline.record("sync", "end", ok=False, target=plan.target_version)
            raise
        self._coordinator.end_sync(plan, succeeded=True)
        self._syncs += 1
        self._timeline.record("sync", "end", ok=True, target=plan.target_version)

    # ---------------------------------------------------------------- report
    def _build_report(self) -> PipelineReport:
        return PipelineReport(
            run_id=self._run_id,
            exit_reason=self._exit_reason,
            steps=self._steps,
            batches_seen=self._batches_seen,
            batches_trained=self._batches_trained,
            batches_dropped_stale=self._dropped_stale,
            syncs=self._syncs,
            trained_response_tokens=self._trained_tokens,
            produced_batches=self._produced,
            queue_max_depth=self._queue.max_depth,
            inflight_peak=self._inflight.peak,
            producer_error=self._producer_error,
            timeline=self._timeline.events(),
            batch_metrics=self._metrics.records(),
        )
