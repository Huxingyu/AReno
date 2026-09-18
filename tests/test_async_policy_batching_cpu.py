"""Whole-group batching preserves admission, ownership and optimizer boundaries."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from areno.api.models import RolloutResult, RolloutSequence, SamplingParams
from areno.experimental.async_policy import (
    AsyncPolicyConfig,
    BatchContractError,
    DeviceMode,
    DualEngineBridge,
    PolicyPipelineCoordinator,
)
from areno.experimental.async_policy.coordination import InflightClosed, InflightLimiter, LagBudgetLimiter
from tests.async_policy_validation.fakes import FakeRolloutEngine, FakeTrainEngine, FakeWeightSync, ready_batch
from tests.async_policy_validation.protocol_cases import assert_drained, config, pipeline, prompts
from tests.test_async_policy_core_cpu import background, wait_for


class BatchedRollout(FakeRolloutEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.groups = []
        self.versions = []

    def generate_batch(self, items, version, *, timeout_s=None):
        try:
            self.enter(timeout_s)
            self.groups.append(items)
            self.versions.append(version)
            # Intentionally return insertion order opposite to input order.
            return {index: RolloutResult(adapter_version=version, sequences=[
                RolloutSequence(resp_tokens=[item.input_tokens[-1], sample], resp_logprobs=[-0.1, -0.2])
                for sample in (0, 1)
            ]) for index, item in reversed(list(enumerate(items)))}
        finally:
            self.active -= 1


@pytest.mark.parametrize("groups", [0, -1, True, 1.5, 3])
def test_batch_group_limit_requires_positive_integer_within_k(groups):
    with pytest.raises(ValueError, match="rollout_batch_groups"):
        AsyncPolicyConfig(max_inflight_rollouts=2, rollout_batch_groups=groups)


def test_nonblocking_permit_does_not_wait_for_a_full_batch():
    limiter = InflightLimiter(2)
    assert limiter.try_acquire() and limiter.try_acquire()
    assert not limiter.try_acquire() and limiter.active == limiter.peak == 2
    limiter.close()
    limiter.release()
    with pytest.raises(InflightClosed):
        limiter.try_acquire()
    limiter.release()
    assert limiter.active == 0


def test_lag_budget_tracks_bound_versions_and_rejects_duplicate_ownership():
    coordinator = PolicyPipelineCoordinator()
    coordinator.attach(config(max_policy_lag=1), DeviceMode.ASYNC)
    limiter = LagBudgetLimiter(coordinator, 1)
    first = limiter.reserve()
    second = limiter.try_reserve()
    assert second is not None and limiter.try_reserve() is None
    limiter.bind(first, 0)
    with pytest.raises(RuntimeError, match="bound more than once"):
        limiter.bind(first, 0)
    limiter.release(first)
    with pytest.raises(RuntimeError, match="released more than once"):
        limiter.release(first)
    limiter.release(second)


def test_unbound_lag_budget_follows_a_new_rollout_version():
    coordinator = PolicyPipelineCoordinator()
    coordinator.attach(config(max_policy_lag=1, weight_sync_interval_updates=1), DeviceMode.ASYNC)
    limiter = LagBudgetLimiter(coordinator, 1)
    first = limiter.reserve()
    admission = coordinator.begin_train(0)
    assert admission.decision == "trained"
    coordinator.end_train(stepped=True)
    assert limiter.try_reserve() is None
    plan = coordinator.begin_sync()
    assert plan is not None
    coordinator.end_sync(plan, succeeded=True)
    limiter.notify_version_change()
    second = limiter.try_reserve()
    assert second is not None
    limiter.bind(first, 1)
    limiter.release(first)
    limiter.release(second)


def test_lag_budget_close_wakes_a_waiter():
    coordinator = PolicyPipelineCoordinator()
    coordinator.attach(config(max_policy_lag=0), DeviceMode.ASYNC)
    limiter = LagBudgetLimiter(coordinator, 0)
    token = limiter.reserve()
    waiting = threading.Event()
    original = limiter._cond.wait

    def observed_wait(timeout=None):
        waiting.set()
        return original(timeout)

    limiter._cond.wait = observed_wait
    errors = []

    def reserve():
        try:
            limiter.reserve(timeout_s=1)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=reserve)
    thread.start()
    assert waiting.wait(1)
    limiter.close()
    thread.join(2)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], InflightClosed)
    limiter.release(token)


def test_batching_requires_adapter_support_before_initialization():
    rollout = FakeRolloutEngine()
    with pytest.raises(TypeError, match="generate_batch"):
        pipeline(rollout=rollout, cfg=config(rollout_batch_groups=2))
    assert rollout.initializations == 0
    # Legacy adapters and the default one-group path remain usable.
    single = pipeline(data=prompts(1), rollout=rollout, cfg=config(max_inflight_rollouts=1))
    assert single.run().steps == 1 and rollout.calls == 1
    assert_drained(single)


@pytest.mark.parametrize("count", [0, 1, 5])
def test_partial_tail_and_out_of_order_groups_keep_independent_advantages(count):
    rollout, train = BatchedRollout(), FakeTrainEngine()

    def reward(record):
        return float(10 * int(record.metadata["prompt_id"]) + record.metadata["sample_index"])

    pipe = pipeline(data=prompts(count), rollout=rollout, train=train, reward=reward,
                    cfg=config(queue_capacity=1, rollout_batch_groups=2))
    report = pipe.run()
    assert_drained(pipe)
    assert report.steps == report.produced_batches == count
    assert report.queue_max_depth <= 1 and report.inflight_peak <= 2
    assert sum(map(len, rollout.groups)) == count
    assert all(1 <= len(group) <= 2 for group in rollout.groups)
    if count:
        assert len(rollout.groups[-1]) == 1
    if count > 1:
        assert len(rollout.groups[0]) == 2
    assert train.versions == list(range(count))
    assert [batch.prompt_ids for batch in train.batches] == [(str(index),) for index in range(count)]
    for index, batch in enumerate(train.batches):
        assert batch.group_offsets == (0, 2)
        assert [row.scalar_advantage for row in batch.sequences] == pytest.approx([-1, 1])
        assert [row.tokens[-2:] for row in batch.sequences] == [[index + 3, 0], [index + 3, 1]]
    if count > 1:
        assert train.batches[0].policy_version == train.batches[1].policy_version


@pytest.mark.parametrize("cadence", [1, 2])
def test_lag_budget_prevents_batched_overproduction(cadence):
    class SlowTrain(FakeTrainEngine):
        def train(self, batch, version, *, timeout_s=None):
            time.sleep(0.005)
            return super().train(batch, version, timeout_s=timeout_s)

    train = SlowTrain()
    pipe = pipeline(data=prompts(20), rollout=BatchedRollout(), train=train,
                    cfg=config(queue_capacity=2, max_inflight_rollouts=2, rollout_batch_groups=2,
                               weight_sync_interval_updates=cadence, max_policy_lag=1))
    report = pipe.run()
    assert report.steps == train.calls == report.produced_batches == 20
    assert report.batches_dropped_stale == 0
    assert all(metric.decision == "trained" for metric in report.batch_metrics)
    assert_drained(pipe)


@pytest.mark.parametrize(("lag", "largest_session"), [(0, 1), (1, 2)])
def test_lag_budget_submits_partial_sessions_when_a_full_session_cannot_fit(lag, largest_session):
    rollout = BatchedRollout()
    pipe = pipeline(data=prompts(7), rollout=rollout,
                    cfg=config(max_inflight_rollouts=3, rollout_batch_groups=3,
                               weight_sync_interval_updates=1, max_policy_lag=lag))
    report = pipe.run()
    assert report.steps == report.produced_batches == 7
    assert report.batches_dropped_stale == 0
    assert max(map(len, rollout.groups)) == largest_session
    assert_drained(pipe)


@pytest.mark.parametrize("bad", ["missing", "extra", "bool_key", "version", "logprobs"])
def test_invalid_second_group_publishes_nothing_and_returns_all_permits(bad):
    class InvalidBatch(BatchedRollout):
        def generate_batch(self, items, version, *, timeout_s=None):
            results = super().generate_batch(items, version, timeout_s=timeout_s)
            if bad == "missing":
                del results[1]
            elif bad == "extra":
                results[2] = results[1]
            elif bad == "bool_key":
                results = {False: results[0], 1: results[1]}
            elif bad == "version":
                results[1].adapter_version = version + 1
            else:
                results[1].sequences[0].resp_logprobs.pop()
            return results

    train = FakeTrainEngine()
    pipe = pipeline(data=prompts(2), train=train, rollout=InvalidBatch(), cfg=config(rollout_batch_groups=2))
    with pytest.raises(BatchContractError):
        pipe.run()
    assert_drained(pipe)
    assert train.calls == pipe.report().produced_batches == 0


def test_source_failure_while_assembling_a_batch_releases_every_permit():
    def source():
        yield prompts(1)[0]
        raise ValueError("second source item failed")

    rollout = BatchedRollout()
    pipe = pipeline(data=source(), rollout=rollout, cfg=config(rollout_batch_groups=2))
    with pytest.raises(ValueError, match="second source"):
        pipe.run()
    assert_drained(pipe)
    assert rollout.calls == 0


def test_source_snapshots_each_group_before_reading_the_next_one():
    source_record = {"value": 0}

    def source():
        for index, prompt in enumerate(prompts(2)):
            source_record["value"] = index
            yield replace(prompt, record=source_record)

    rollout = BatchedRollout()
    pipe = pipeline(data=source(), rollout=rollout, cfg=config(rollout_batch_groups=2))
    pipe.run()
    assert [prompt.record["value"] for prompt in rollout.groups[0]] == [0, 1]
    assert_drained(pipe)


@pytest.mark.parametrize("stage", ["rollout", "reward", "train", "sync"])
def test_batched_failure_does_not_leave_work_or_permits(stage):
    boom = ValueError("injected batch failure")
    rollout, train, sync = BatchedRollout(), FakeTrainEngine(), FakeWeightSync()
    reward = None
    if stage == "reward":
        def reward(record):
            if record.metadata["prompt_id"] == "1":
                raise boom
            return float(record.metadata["sample_index"])
    else:
        {"rollout": rollout, "train": train, "sync": sync}[stage].failures[1 if stage == "sync" else 0] = boom
    pipe = pipeline(data=prompts(6), rollout=rollout, train=train, sync=sync, reward=reward,
                    cfg=config(rollout_batch_groups=2))
    with pytest.raises(ValueError) as caught:
        pipe.run()
    assert caught.value is boom
    assert_drained(pipe)


def test_cancel_during_generation_releases_all_groups_without_scoring():
    entered, release = threading.Event(), threading.Event()
    rollout = BatchedRollout(started_events={0: entered}, gates={0: release})
    scored = []
    pipe = pipeline(data=prompts(4), rollout=rollout, reward=lambda record: scored.append(record),
                    cfg=config(rollout_batch_groups=2))
    thread, errors = background(pipe.run)
    try:
        assert entered.wait(2)
        assert pipe.inflight_active == 2
        pipe.request_stop()
    finally:
        release.set()
        thread.join(4)
        pipe.close()
    assert not errors and not thread.is_alive() and not scored
    assert_drained(pipe)


def test_sync_has_priority_between_batch_sessions():
    generating, generated = threading.Event(), threading.Event()
    syncing, synced = threading.Event(), threading.Event()
    rollout = BatchedRollout(started_events={0: generating}, gates={0: generated})
    sync = FakeWeightSync(started_events={1: syncing}, gates={1: synced})
    bridge = DualEngineBridge(train_engine=FakeTrainEngine(), rollout_engine=rollout, weight_sync=sync,
                              config=config(rollout_batch_groups=2))
    bridge.initialize()
    threads, errors = [], []
    try:
        first, first_errors = background(lambda: bridge.rollout_sessions(tuple(prompts(2))))
        threads.append(first)
        errors.append(first_errors)
        assert generating.wait(2)
        assert bridge.coordinator.active_rollouts == 1
        bridge.train(ready_batch())
        second, second_errors = background(lambda: bridge.rollout_sessions(tuple(prompts(2))))
        threads.append(second)
        errors.append(second_errors)
        copy, copy_errors = background(bridge.sync)
        threads.append(copy)
        errors.append(copy_errors)
        generated.set()
        assert syncing.wait(2)
        assert rollout.calls == 1, "the next session overtook pending synchronization"
        synced.set()
    finally:
        generated.set()
        synced.set()
        for thread in threads:
            thread.join(4)
        bridge.close()
    assert not any(errors) and all(not thread.is_alive() for thread in threads)
    assert rollout.versions == [0, 1]


def test_cancel_with_full_ready_queue_releases_batched_publishers():
    entered, release = threading.Event(), threading.Event()
    train = FakeTrainEngine(started_events={0: entered}, gates={0: release})
    pipe = pipeline(data=prompts(10), rollout=BatchedRollout(), train=train,
                    cfg=config(queue_capacity=1, max_inflight_rollouts=4, rollout_batch_groups=2))
    thread, errors = background(pipe.run)
    try:
        assert entered.wait(2)
        wait_for(lambda: pipe._queue.depth == 1)
        pipe.request_stop()
    finally:
        release.set()
        thread.join(4)
        pipe.close()
    assert not errors and not thread.is_alive()
    assert_drained(pipe)


def test_batch_results_are_owned_before_an_adapter_reuses_buffers():
    class ReusingBatch(BatchedRollout):
        result = RolloutResult(sequences=[RolloutSequence(resp_tokens=[7], resp_logprobs=[-0.5])])

        def generate_batch(self, items, version, *, timeout_s=None):
            self.result.sequences[0].resp_tokens[0] += 1
            return dict.fromkeys(range(len(items)), self.result)

    bridge = DualEngineBridge(train_engine=FakeTrainEngine(), rollout_engine=ReusingBatch(),
                              weight_sync=FakeWeightSync(), config=config(rollout_batch_groups=2))
    bridge.initialize()
    try:
        first = bridge.rollout_sessions(tuple(prompts(2)))
        second = bridge.rollout_sessions(tuple(prompts(2)))
        assert first[0].result.sequences[0].resp_tokens == [8]
        assert second[0].result.sequences[0].resp_tokens == [9]
        first[0].result.sequences[0].resp_tokens[0] = 0
        assert first[1].result.sequences[0].resp_tokens == [8]
    finally:
        bridge.close()


@pytest.mark.parametrize("fault", [None, "incomplete", "wrong_rows", "version"])
def test_native_batch_rows_restore_group_order_and_bound_capacity(monkeypatch, tmp_path, fault):
    import torch

    from areno.api.backend.cuda import generation
    from areno.api.config import CudaConfig
    from areno.engine.protocol import Op
    from areno.engine.worker import _build_rollout_from_tensor_row_ids
    from areno.experimental.async_policy.native import NativeCudaEnginePair

    pair = NativeCudaEnginePair(model_path=str(tmp_path), config=CudaConfig(
        devices=[0], rollout_devices=[1], max_running_prompts=4,
    ), sampling_params=SamplingParams(max_prompt_len=128, max_new_tokens=4), tokenizer=None, n_samples=3)
    pair._initialized = True
    pair._clusters = {"rollout": SimpleNamespace(config=SimpleNamespace(runtime=SimpleNamespace(kv_block_size=16)))}
    monkeypatch.setattr(generation, "rollout_options", lambda *args: {
        "eos_token_id": [2], "max_prompt_len": 128, "sampling_params": SimpleNamespace(seed=None),
    })
    calls = []

    def call(role, op, payload, deadline):
        calls.append(op)
        if op != Op.INFER_ROLLOUT:
            return [None]
        assert payload.prompt_indices_by_dp == [list(range(6))]
        assert payload.max_running_seqs == 4
        assert payload.num_blocks == 4 * payload.max_blocks_per_seq
        assert payload.max_prefill_tokens == 4 * 128
        tokens = torch.zeros((6, 4), dtype=torch.long)
        logprobs = torch.zeros((6, 4))
        # Model completions arrive in arbitrary order but fill their state rows.
        for row in (4, 1, 3, 5, 0, 2):
            tokens[row, :2] = torch.tensor([100 + row, 200 + row])
            logprobs[row, :2] = torch.tensor([-row / 10, -row / 20])
        result = _build_rollout_from_tensor_row_ids(payload.prompts_by_dp[0], tokens, logprobs,
                    torch.full((6,), 2), ["length"] * 6, list(range(6)), ())
        result.adapter_version = 4 if fault == "version" else 3
        if fault == "incomplete":
            result.response_ids.pop()
        if fault == "wrong_rows":
            result.prompt_ids.reverse()
        return [result]

    pair._call = call
    if fault:
        with pytest.raises(RuntimeError):
            pair.generate_batch(tuple(prompts(2)), 3, timeout_s=1)
    else:
        groups = pair.generate_batch(tuple(prompts(2)), 3, timeout_s=1)
        assert [row.resp_tokens[0] for row in groups[0].sequences] == [100, 101, 102]
        assert [row.resp_tokens[0] for row in groups[1].sequences] == [103, 104, 105]
        assert groups[1].sequences[2].resp_logprobs == pytest.approx([-0.5, -0.25])
    assert calls == [Op.ROLLOUT_SESSION_BEGIN, Op.INFER_ROLLOUT, Op.ROLLOUT_SESSION_END]
