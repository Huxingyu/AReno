"""Bounded producer tasks and a single training consumer, all through the bridge."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .bridge import DualEngineBridge
from .contracts import (
    AsyncPolicyConfig,
    AsyncPrompt,
    BatchEnvelope,
    BatchMetric,
    BridgeStateError,
    Lifecycle,
    PipelineClosed,
    PipelineReport,
    RolloutEngine,
    RolloutSession,
    ShutdownTimeout,
    SyncMetric,
    TrainEngine,
    WeightSync,
)
from .coordination import (
    BoundedReadyQueue,
    Deadline,
    InflightLimiter,
    LagBudgetLimiter,
    PolicyPipelineCoordinator,
    QueueEmpty,
    QueueFinished,
)
from .data import build_batch_envelope, score_prompt_group, snapshot_prompt

if TYPE_CHECKING:
    from areno.api.rewards import RewardRecord


@dataclass(frozen=True)
class _ReadyBatch:
    batch: BatchEnvelope
    lag_budget_token: int


class AsyncPolicyPipeline:
    def __init__(
        self, *, config: AsyncPolicyConfig, data_source: Iterable[AsyncPrompt],
        rollout_engine: RolloutEngine, train_engine: TrainEngine, weight_sync: WeightSync,
        reward_fn: Callable[[RewardRecord], float], coordinator: PolicyPipelineCoordinator | None = None,
        decode: Callable[[list[int]], str] | None = None, eos_token_id: int = 0,
    ):
        self._config = config
        self._source = data_source
        self._reward_fn = reward_fn
        self._decode = decode
        self._eos_token_id = eos_token_id
        self._run_id = uuid.uuid4().hex
        self._bridge = DualEngineBridge(
            config=config, rollout_engine=rollout_engine, train_engine=train_engine,
            weight_sync=weight_sync, coordinator=coordinator,
        )
        self._coordinator = self._bridge.coordinator
        self._supervisor = self._bridge.supervisor
        self._queue: BoundedReadyQueue[_ReadyBatch] = BoundedReadyQueue(config.queue_capacity)
        self._inflight = InflightLimiter(config.max_inflight_rollouts)
        # Preserve the established speculative single-group scheduler. The
        # extra budget is needed only when one session creates multiple update
        # boundaries at once and can otherwise refill the queue with work that
        # is guaranteed to expire.
        lag_budget = config.max_policy_lag if config.rollout_batch_groups > 1 else None
        self._lag_budget = LagBudgetLimiter(self._coordinator, lag_budget)
        self._supervisor.add_stop_hook(self._queue.abort)
        self._supervisor.add_stop_hook(self._inflight.close)
        self._supervisor.add_stop_hook(self._lag_budget.close)
        self._producer: threading.Thread | None = None
        self._close_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._batch_metrics: list[BatchMetric] = []
        self._sync_metrics: list[SyncMetric] = []
        self._batches_seen = 0
        self._batch_wait_s = 0.0
        self._initial_version = self._coordinator.train_policy_version

    @property
    def stop_event(self) -> threading.Event:
        return self._supervisor.stop_event

    @property
    def producer_alive(self) -> bool:
        return self._producer is not None and self._producer.is_alive()

    @property
    def inflight_active(self) -> int:
        return self._inflight.active

    def request_stop(self) -> None:
        self._supervisor.stop("stopped")

    def abort(self) -> None:
        self._supervisor.stop("aborted")

    def _publish_group(
        self, prompt: AsyncPrompt, session: RolloutSession, batch_id: str, lag_budget_token: int,
    ) -> None:
        self._lag_budget.bind(lag_budget_token, session.policy_version)
        rewards = score_prompt_group(
            prompt, session.result, self._reward_fn, decode=self._decode,
            check_running=self._supervisor.check_running,
        )
        batch = build_batch_envelope(
            run_id=self._run_id, batch_id=batch_id, epoch=0, policy_version=session.policy_version,
            prompt_items=[prompt], rollout_results=[session.result], rewards=rewards,
            eos_token_id=self._eos_token_id,
        )
        self._supervisor.check_running()
        self._queue.put(_ReadyBatch(batch, lag_budget_token))

    def _production_task(self, prompt: AsyncPrompt, batch_id: str, lag_budget_token: int) -> None:
        published = False
        try:
            self._supervisor.check_running()
            self._publish_group(prompt, self._bridge.rollout_session(prompt), batch_id, lag_budget_token)
            published = True
        except BaseException as exc:
            if not isinstance(exc, PipelineClosed) or not self.stop_event.is_set():
                self._supervisor.fail(exc)
        finally:
            if not published:
                self._lag_budget.release(lag_budget_token)
            self._inflight.release()

    def _production_batch_task(self, groups: tuple[tuple[AsyncPrompt, str, int], ...]) -> None:
        unpublished = {lag_budget_token for _, _, lag_budget_token in groups}
        try:
            self._supervisor.check_running()
            sessions = self._bridge.rollout_sessions(tuple(prompt for prompt, _, _ in groups))
            for (prompt, batch_id, lag_budget_token), session in zip(groups, sessions, strict=True):
                self._publish_group(prompt, session, batch_id, lag_budget_token)
                unpublished.remove(lag_budget_token)
        except BaseException as exc:
            if not isinstance(exc, PipelineClosed) or not self.stop_event.is_set():
                self._supervisor.fail(exc)
        finally:
            for lag_budget_token in unpublished:
                self._lag_budget.release(lag_budget_token)
            for _ in groups:
                self._inflight.release()

    def _produce(self) -> None:
        executor = None
        iterator = None
        exhausted = False
        try:
            executor = ThreadPoolExecutor(
                max_workers=self._config.max_inflight_rollouts, thread_name_prefix="areno-async-work",
            )
            index = 0
            while not self.stop_event.is_set():
                self._inflight.acquire()
                permits = 1
                lag_budget_tokens = []
                submitted = False
                try:
                    lag_budget_tokens.append(self._lag_budget.reserve())
                    self._supervisor.check_running()
                    if iterator is None:
                        iterator = iter(self._source)
                    groups = []
                    for group_index in range(self._config.rollout_batch_groups):
                        if group_index:
                            if not self._inflight.try_acquire():
                                break
                            permits += 1
                            lag_budget_token = self._lag_budget.try_reserve()
                            if lag_budget_token is None:
                                self._inflight.release()
                                permits -= 1
                                break
                            lag_budget_tokens.append(lag_budget_token)
                        try:
                            prompt = next(iterator)
                        except StopIteration:
                            exhausted = True
                            self._lag_budget.release(lag_budget_tokens.pop())
                            self._inflight.release()
                            permits -= 1
                            break
                        groups.append((snapshot_prompt(prompt), f"{self._run_id}:{index}", lag_budget_tokens[-1]))
                        index += 1
                    if not groups:
                        break
                    self._supervisor.check_running()
                    # At most K futures exist: a permit spans read, submit, work and put.
                    if self._config.rollout_batch_groups == 1:
                        executor.submit(self._production_task, *groups[0])
                    else:
                        executor.submit(self._production_batch_task, tuple(groups))
                    submitted = True
                finally:
                    if not submitted:
                        for lag_budget_token in lag_budget_tokens:
                            self._lag_budget.release(lag_budget_token)
                        for _ in range(permits):
                            self._inflight.release()
                if exhausted:
                    break
        except BaseException as exc:
            if not isinstance(exc, PipelineClosed) or not self.stop_event.is_set():
                self._supervisor.fail(exc)
        finally:
            try:
                if executor is not None:
                    # Cancelled queued tasks still execute their release finally block.
                    executor.shutdown(wait=True)
                if iterator is not None:
                    close_source = getattr(iterator, "close", None)
                    if close_source is not None:
                        close_source()
            except BaseException as exc:
                self._supervisor.fail(exc)
            if exhausted and not self.stop_event.is_set():
                self._supervisor.drain()
                self._queue.finish()

    def _record_sync(self, metric: SyncMetric | None) -> None:
        if metric is not None:
            with self._metrics_lock:
                self._sync_metrics.append(metric)
            self._lag_budget.notify_version_change()

    def _record_terminal_batch(self, batch: BatchEnvelope, decision: str, wait_s: float = 0.0) -> None:
        state = self._coordinator.snapshot()
        with self._metrics_lock:
            self._batch_metrics.append(BatchMetric(
                batch.batch_id, batch.policy_version, state.train_policy_version, state.rollout_policy_version,
                state.train_policy_version, state.train_policy_version - batch.policy_version,
                decision, False, wait_s,
            ))

    def _consume(self) -> None:
        if self._config.max_steps == 0:
            self._supervisor.stop("max_steps")
            return
        wait_s = 0.0
        while not self.stop_event.is_set():
            if self._coordinator.sync_pending:
                self._record_sync(self._bridge.sync())
            start = time.monotonic()
            try:
                try:
                    ready = self._queue.get(timeout_s=self._config.poll_interval_s)
                finally:
                    elapsed = time.monotonic() - start
                    wait_s += elapsed
                    with self._metrics_lock:
                        self._batch_wait_s += elapsed
            except QueueEmpty:
                continue
            except QueueFinished:
                self._record_sync(self._bridge.sync())
                self._supervisor.stop("data_exhausted")
                break
            batch = ready.batch
            with self._metrics_lock:
                self._batches_seen += 1
            try:
                try:
                    result = self._bridge.train(batch)
                except BaseException:
                    decision = "failed" if self._supervisor.failure is not None else "cancelled"
                    self._record_terminal_batch(batch, decision, wait_s)
                    wait_s = 0.0
                    raise
            finally:
                self._lag_budget.release(ready.lag_budget_token)
            admission = result.admission
            with self._metrics_lock:
                self._batch_metrics.append(BatchMetric(
                    batch.batch_id, batch.policy_version, admission.train_version, admission.rollout_version,
                    result.train_version_after, admission.lag, admission.decision, result.stepped,
                    wait_s, result.admission_wait_s, result.train_s,
                ))
            wait_s = 0.0
            steps = result.train_version_after - self._initial_version
            if self._config.max_steps is not None and steps >= self._config.max_steps:
                self._supervisor.stop("max_steps")

    def run(self) -> PipelineReport:
        if self._supervisor.state != Lifecycle.NEW:
            raise BridgeStateError("a pipeline is single-use")
        try:
            self._record_sync(self._bridge.initialize())
            self._supervisor.start_run()
            if self._config.max_steps != 0:
                # Serialize thread publication/start with close, including start failures.
                with self._close_lock:
                    self._supervisor.check_running()
                    self._producer = threading.Thread(target=self._produce, name="areno-async-producer")
                    self._producer.start()
            self._consume()
        except BaseException as exc:
            if not isinstance(exc, PipelineClosed) or not self.stop_event.is_set():
                self._supervisor.fail(exc)
        finally:
            try:
                self.close()
            except BaseException as exc:
                self._supervisor.fail(exc)
            for ready in self._queue.discard_pending():
                self._record_terminal_batch(ready.batch, "cancelled")
                self._lag_budget.release(ready.lag_budget_token)
        if self._supervisor.failure is not None:
            raise self._supervisor.failure
        return self.report()

    def close(self, *, timeout_s: float | None = None) -> None:
        deadline = Deadline(self._config.shutdown_timeout_s if timeout_s is None else timeout_s)
        self.request_stop()
        if not self._close_lock.acquire(timeout=deadline.remaining()):
            error = ShutdownTimeout("another pipeline close is still running")
            self._supervisor.fail(error)
            raise error
        try:
            if self._producer is not None and self._producer.ident is not None:
                self._producer.join(timeout=deadline.remaining())
                if self._producer.is_alive():
                    raise ShutdownTimeout("producer or accepted work is still running after the shutdown deadline")
            self._bridge.close(timeout_s=deadline.remaining())
        except BaseException as exc:
            self._supervisor.fail(exc)
            raise
        finally:
            self._close_lock.release()

    def report(self) -> PipelineReport:
        state = self._coordinator.snapshot()
        with self._metrics_lock:
            batches = tuple(self._batch_metrics)
            return PipelineReport(
                exit_reason=self._supervisor.exit_reason, state=self._supervisor.state.value,
                failure=str(self._supervisor.failure) if self._supervisor.failure is not None else None,
                secondary_failures=self._supervisor.secondary_failures,
                steps=state.train_policy_version - self._initial_version,
                produced_batches=self._queue.published, batches_seen=self._batches_seen,
                batches_dropped_stale=sum(row.decision == "stale" for row in batches),
                batches_dropped_empty=sum(row.decision == "empty" for row in batches),
                batches_cancelled=sum(row.decision == "cancelled" for row in batches),
                queue_max_depth=self._queue.max_depth, inflight_peak=self._inflight.peak,
                producer_alive=self.producer_alive, inflight_active=self.inflight_active,
                active_rollouts=state.active_rollouts, train_active=state.train_active, sync_active=state.sync_active,
                train_policy_version=state.train_policy_version, rollout_policy_version=state.rollout_policy_version,
                batch_wait_s=self._batch_wait_s, batch_metrics=batches, sync_metrics=tuple(self._sync_metrics),
            )
