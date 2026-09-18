"""Additional black-box contracts and seeded stress; intentionally not auto-collected."""

from __future__ import annotations

import gc
import itertools
import json
import os
import random
import threading
import time
import tracemalloc
import weakref
from dataclasses import asdict
from pathlib import Path

import pytest

from areno.experimental.async_policy import (
    AsyncPolicyConfig,
    AsyncPolicyPipeline,
    AsyncPrompt,
    BoundedReadyQueue,
    InflightClosed,
    InflightLimiter,
    QueueAborted,
    QueueEmpty,
)
from tests.async_policy_validation.fakes import FakeRolloutEngine, FakeTrainEngine, FakeWeightSync, sample_reward


def config(**kwargs):
    values = dict(queue_capacity=2, max_inflight_rollouts=2, max_policy_lag=100,
                  operation_timeout_s=3, shutdown_timeout_s=3, poll_interval_s=0.005)
    values.update(kwargs)
    return AsyncPolicyConfig(**values)


def prompts(count=12):
    return [AsyncPrompt(prompt_id=str(i), input_tokens=(1, 2, i + 3)) for i in range(count)]


def pipeline(*, data=None, cfg=None, rollout=None, train=None, sync=None, reward=None):
    return AsyncPolicyPipeline(config=cfg or config(), data_source=prompts() if data is None else data,
                               rollout_engine=rollout or FakeRolloutEngine(),
                               train_engine=train or FakeTrainEngine(), weight_sync=sync or FakeWeightSync(),
                               reward_fn=reward or sample_reward)


def save(name, value):
    directory = os.environ.get("ARENO_AUDIT_ARTIFACT_DIR")
    if directory:
        (Path(directory) / f"{name}.json").write_text(json.dumps(value, indent=2, default=str) + "\n")


def assert_drained(pipe):
    assert not pipe.producer_alive
    assert pipe.inflight_active == 0, f"leaked permits: {pipe.inflight_active}"
    assert pipe._lag_budget.active == 0, f"leaked lag budget: {pipe._lag_budget.active}"
    assert pipe._coordinator.active_rollouts == 0
    assert not pipe._coordinator.train_active
    assert not pipe._coordinator.sync_active


def test_queue_abort_rejects_buffered_batches():
    queue = BoundedReadyQueue(1)
    queue.put("batch")
    queue.abort()
    with pytest.raises(QueueAborted):
        queue.get(timeout_s=0)


def test_inflight_close_and_release_rejects_waiter():
    limiter = InflightLimiter(1)
    limiter.acquire()
    waiting = threading.Event()
    original = limiter._cond.wait

    def observed_wait(timeout=None):
        waiting.set()
        return original(timeout)

    limiter._cond.wait = observed_wait
    errors = []
    acquired = []

    def acquire():
        try:
            limiter.acquire(timeout_s=1)
            acquired.append(True)
            limiter.release()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=acquire)
    thread.start()
    assert waiting.wait(1)
    with limiter._cond:
        limiter.close()
        limiter.release()
    thread.join(2)
    assert not thread.is_alive()
    assert not acquired, "waiter acquired a permit after close()"
    assert len(errors) == 1 and isinstance(errors[0], InflightClosed)


def test_data_source_iter_error_reaches_consumer():
    class BrokenSource:
        def __iter__(self):
            raise ValueError("source initialization failed")

    pipe = pipeline(data=BrokenSource())
    done = threading.Event()
    errors = []

    def run():
        try:
            pipe.run()
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        completed = done.wait(0.4)
    finally:
        pipe.request_stop()
        thread.join(2)
        pipe.close()
    assert completed, "producer died while the consumer kept polling an open queue"
    assert len(errors) == 1 and str(errors[0]) == "source initialization failed"


def test_data_source_next_error_releases_permit():
    def source():
        raise ValueError("source read failed")
        yield  # pragma: no cover - makes this a generator

    pipe = pipeline(data=source())
    with pytest.raises(ValueError, match="source read failed"):
        pipe.run()
    assert_drained(pipe)


def test_train_failure_report_records_started_run():
    pipe = pipeline(train=FakeTrainEngine(failures={0: ValueError("train failed")}))
    with pytest.raises(ValueError, match="train failed"):
        pipe.run()
    assert_drained(pipe)
    assert pipe.report().exit_reason != "not_started", asdict(pipe.report())


@pytest.mark.parametrize("stage", ["rollout", "reward", "train", "sync"])
def test_stage_failure_propagates_and_releases_resources(stage):
    boom = ValueError(f"injected {stage} failure")
    rollout, train, sync = FakeRolloutEngine(), FakeTrainEngine(), FakeWeightSync()
    reward = sample_reward
    if stage == "reward":
        def reward(record):
            raise boom
    else:
        # Call 0 now performs mandatory initial alignment; inject the runtime sync.
        {"rollout": rollout, "train": train, "sync": sync}[stage].failures[1 if stage == "sync" else 0] = boom
    pipe = pipeline(rollout=rollout, train=train, sync=sync, reward=reward)
    with pytest.raises(ValueError) as caught:
        pipe.run()
    assert caught.value is boom
    assert_drained(pipe)


def test_full_queue_stop_releases_all_producers():
    started, gate = threading.Event(), threading.Event()
    pipe = pipeline(cfg=config(queue_capacity=1, max_inflight_rollouts=3),
                    train=FakeTrainEngine(started_events={0: started}, gates={0: gate}))
    errors = []

    def run():
        try:
            pipe.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(2)
        deadline = time.monotonic() + 2
        while pipe._queue.depth != 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert pipe._queue.depth == 1
        pipe.request_stop()
    finally:
        gate.set()
        pipe.request_stop()
        thread.join(4)
    assert not thread.is_alive()
    assert not errors
    assert_drained(pipe)


def test_empty_data_and_repeated_lifecycle():
    before = {thread.ident for thread in threading.enumerate()}
    for _ in range(30):
        pipe = pipeline(data=[])
        report = pipe.run()
        assert report.steps == 0 and report.exit_reason == "data_exhausted"
        pipe.close()
        pipe.close()
        assert_drained(pipe)
    remaining = [t.name for t in threading.enumerate() if t.ident not in before]
    assert not remaining, remaining


def test_repeated_nonempty_runs_release_objects_and_threads():
    warmup = pipeline(data=prompts(2))
    warmup.run()
    warmup.close()
    del warmup
    gc.collect()
    before = {thread.ident for thread in threading.enumerate()}
    references = []
    tracemalloc.start()
    for index in range(50):
        pipe = pipeline(data=prompts(8), cfg=config(queue_capacity=1 + index % 3,
                        max_inflight_rollouts=1 + index % 4))
        pipe.run()
        assert_drained(pipe)
        pipe.close()
        references.append(weakref.ref(pipe))
        del pipe
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    remaining = [thread.name for thread in threading.enumerate() if thread.ident not in before]
    alive = sum(reference() is not None for reference in references)
    save("lifecycle-soak", dict(runs=50, prompt_groups=400, live_pipelines=alive,
                               remaining_threads=remaining, traced_current_bytes=current, traced_peak_bytes=peak))
    assert alive == 0 and not remaining


def test_batch_wait_metric_includes_empty_polls():
    gate, polled = threading.Event(), threading.Event()
    pipe = pipeline(data=prompts(1), rollout=FakeRolloutEngine(gates={0: gate}))
    original_get = pipe._queue.get
    empty_waits = []

    def observed_get(*, timeout_s=None):
        start = time.monotonic()
        try:
            return original_get(timeout_s=timeout_s)
        except QueueEmpty:
            empty_waits.append(time.monotonic() - start)
            if len(empty_waits) == 5:
                polled.set()
            raise

    pipe._queue.get = observed_get
    errors = []

    def run():
        try:
            pipe.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert polled.wait(2)
    finally:
        gate.set()
        thread.join(4)
        pipe.close()
    assert not errors and not thread.is_alive()
    measured = pipe.report().batch_metrics[0].wait_s
    observed = sum(empty_waits[:5])
    save("wait-time", dict(reported_wait_s=measured, observed_empty_wait_s=observed))
    assert measured >= observed * 0.9, f"reported {measured}s after waiting at least {observed}s"


@pytest.mark.parametrize("seed", [0, 11, 29, 47])
def test_seeded_capacity_and_lag_matrix(seed):
    records = []
    # 3 * 3 * 2 * 3 = 54 configs, each with 16 prompt groups and variable latency.
    for q, k, cadence, lag in itertools.product((1, 2, 4), (1, 2, 4), (1, 4), (0, 1, 4)):
        rollout_rng, train_rng, reward_rng = [random.Random(seed + offset) for offset in (0, 100, 200)]

        class Rollout(FakeRolloutEngine):
            def generate(self, prompt, version, *, timeout_s=None):
                time.sleep(rollout_rng.uniform(0, 0.001))
                result = super().generate(prompt, version, timeout_s=timeout_s)
                return result

        class Train(FakeTrainEngine):
            def __init__(self):
                super().__init__()
                self.batch_ids = []

            def train(self, batch, version, *, timeout_s=None):
                time.sleep(train_rng.uniform(0, 0.001))
                assert len(batch.group_offsets) == 2 and len(batch.sequences) == 2
                self.batch_ids.append(batch.batch_id)
                self.calls += 1
                return self.calls % 3 != 1

        delays = {str(i): reward_rng.uniform(0, 0.002) for i in range(16)}

        def reward(record):
            time.sleep(delays[record.metadata["prompt_id"]])
            return float(record.metadata["sample_index"])

        train = Train()
        pipe = pipeline(data=prompts(16), cfg=config(queue_capacity=q, max_inflight_rollouts=k,
                        weight_sync_interval_updates=cadence, max_policy_lag=lag),
                        rollout=Rollout(), train=train, reward=reward)
        report = pipe.run()
        assert_drained(pipe)
        assert report.exit_reason == "data_exhausted"
        assert report.produced_batches == report.batches_seen == 16
        assert report.queue_max_depth <= q and report.inflight_peak <= k
        assert len(set(train.batch_ids)) == len(train.batch_ids)
        assert report.steps == sum(metric.stepped for metric in report.batch_metrics)
        assert report.steps == pipe._coordinator.train_policy_version
        assert len(train.batch_ids) + report.batches_dropped_stale == 16
        assert all(0 <= metric.lag <= lag for metric in report.batch_metrics if metric.decision == "trained")
        records.append({"seed": seed, "Q": q, "K": k, "C": cadence, "lag": lag,
                        "steps": report.steps, "dropped": report.batches_dropped_stale,
                        "queue_peak": report.queue_max_depth, "inflight_peak": report.inflight_peak})
    save(f"matrix-seed-{seed}", records)
