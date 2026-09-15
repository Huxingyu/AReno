"""Additional contracts for the rewrite, using events to control competing operations."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import replace

import pytest

from areno.api.models import RolloutResult, RolloutSequence
from areno.experimental.async_policy import (
    AsyncPolicyConfig,
    AsyncPrompt,
    BatchContractError,
    DeviceMode,
    DualEngineBridge,
    PipelineClosed,
    PolicyPipelineCoordinator,
    QueueEmpty,
    build_batch_envelope,
)
from tests.async_policy_validation.fakes import FakeRolloutEngine, FakeTrainEngine, FakeWeightSync, ready_batch
from tests.async_policy_validation.protocol_cases import assert_drained, config, pipeline, prompts


def background(fn):
    errors = []

    def run():
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, errors


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline, "condition did not become true"
        time.sleep(0.001)


@pytest.mark.parametrize("field,value", [
    ("queue_capacity", 0), ("queue_capacity", True), ("max_inflight_rollouts", -1),
    ("max_inflight_rollouts", 1.5), ("weight_sync_interval_updates", 0),
    ("max_policy_lag", -1), ("max_policy_lag", False), ("max_steps", -1),
    ("max_steps", 0.5), ("operation_timeout_s", float("nan")),
    ("operation_timeout_s", float("inf")), ("shutdown_timeout_s", 0),
    ("poll_interval_s", -0.1), ("poll_interval_s", True),
])
def test_invalid_config_is_rejected_before_work(field, value):
    with pytest.raises(ValueError, match=field):
        AsyncPolicyConfig(**{field: value})


def test_import_and_plugin_discovery_do_not_load_torch_or_register_a_fake_algorithm():
    subprocess.run([sys.executable, "-c", """
import sys
from areno.api.algorithms import list_algorithms
before = set(list_algorithms(include_experimental=False))
from areno.experimental.async_policy import AsyncPolicyPipeline
assert set(list_algorithms()) == before
assert 'torch' not in sys.modules
from areno import Trainer
assert 'areno.api.backend.cuda' not in sys.modules
"""], check=True)


def test_pipeline_uses_the_bridge_for_all_model_operations(monkeypatch):
    pipe = pipeline(data=prompts(3))
    calls = {name: 0 for name in ("initialize", "rollout_session", "train", "sync")}
    for name in calls:
        original = getattr(pipe._bridge, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(pipe._bridge, name, counted)
    report = pipe.run()
    assert calls["initialize"] == 1 and calls["rollout_session"] == 3
    assert calls["train"] == 3 and calls["sync"] >= 1
    assert report.steps == 3
    assert_drained(pipe)


def test_zero_gradient_optimizer_steps_still_advance_versions():
    import torch

    class ZeroGradientTrain(FakeTrainEngine):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, device="cpu"))
            self.optimizer = torch.optim.SGD([self.weight], lr=0.1)
            self.actual_updates = 0

        def train(self, batch, version, *, timeout_s=None):
            assert version == self.actual_updates
            assert super().train(batch, version, timeout_s=timeout_s)
            self.optimizer.zero_grad(set_to_none=True)
            (self.weight * 0).sum().backward()
            self.optimizer.step()
            self.actual_updates += 1
            return True

    train = ZeroGradientTrain()
    pipe = pipeline(data=prompts(4), train=train)
    report = pipe.run()
    assert train.weight.item() == 1
    assert report.steps == train.actual_updates == report.train_policy_version == 4
    assert all(metric.stepped for metric in report.batch_metrics)
    assert_drained(pipe)


@pytest.mark.parametrize("empty_kind", ["no_samples", "no_response_tokens"])
def test_empty_batches_never_call_optimizer(empty_kind):
    class EmptyRollout(FakeRolloutEngine):
        def generate(self, prompt, version, *, timeout_s=None):
            return RolloutResult(sequences=[] if empty_kind == "no_samples" else [RolloutSequence()])

    train = FakeTrainEngine()
    pipe = pipeline(data=prompts(3), rollout=EmptyRollout(), train=train)
    report = pipe.run()
    assert report.steps == train.calls == 0
    assert report.batches_dropped_empty == report.produced_batches == report.batches_seen == 3
    assert report.train_policy_version == report.rollout_policy_version == 0
    assert_drained(pipe)


def test_max_steps_counts_updates_and_zero_limit_never_reads_source():
    def forbidden_source():
        raise AssertionError("max_steps=0 read the source")
        yield

    zero = pipeline(data=forbidden_source(), cfg=config(max_steps=0))
    assert zero.run().steps == 0
    assert_drained(zero)
    train = FakeTrainEngine(skip_calls=(0, 2))
    pipe = pipeline(data=prompts(20), train=train, cfg=config(max_steps=2))
    report = pipe.run()
    assert report.exit_reason == "max_steps" and report.steps == 2
    assert train.calls == 4 and sum(row.stepped for row in report.batch_metrics) == 2
    assert len(report.batch_metrics) == report.produced_batches
    assert_drained(pipe)


def test_future_batch_fails_before_model_use():
    train = FakeTrainEngine()
    bridge = DualEngineBridge(train_engine=train, rollout_engine=FakeRolloutEngine(), weight_sync=FakeWeightSync())
    bridge.initialize()
    try:
        with pytest.raises(BatchContractError, match="future"):
            bridge.train(replace(ready_batch(), policy_version=1))
        assert train.calls == 0
        with pytest.raises(PipelineClosed):
            bridge.train(ready_batch())
    finally:
        bridge.close()


def test_lag_is_checked_after_waiting_for_train_and_sync():
    coordinator = PolicyPipelineCoordinator()
    coordinator.attach(config(max_policy_lag=0, weight_sync_interval_updates=4), DeviceMode.ASYNC)
    coordinator.begin_train(0)
    waiting = threading.Event()
    original = coordinator._cond.wait

    def observed_wait(timeout=None):
        waiting.set()
        return original(timeout)

    coordinator._cond.wait = observed_wait
    admissions = []
    thread, errors = background(lambda: admissions.append(coordinator.begin_train(0, timeout_s=2)))
    try:
        assert waiting.wait(1)
        coordinator.end_train(stepped=True)
        # The cadence is 4, but lag=0 forces alignment before further generation.
        assert coordinator.sync_pending
        plan = coordinator.begin_sync(timeout_s=1)
        assert plan.target_version == 1
        coordinator.end_sync(plan, succeeded=True)
        thread.join(2)
        assert not errors and not thread.is_alive()
        assert len(admissions) == 1 and admissions[0].decision == "stale"
        assert admissions[0].lag == 1 and not coordinator.train_active
    finally:
        coordinator.close()
        thread.join(2)


def test_committed_update_closes_rollout_admission_before_sync_starts():
    coordinator = PolicyPipelineCoordinator()
    coordinator.begin_train(0)
    coordinator.end_train(stepped=True)
    entered = threading.Event()
    finished = threading.Event()
    versions = []

    def generate():
        entered.set()
        try:
            versions.append(coordinator.begin_rollout(timeout_s=2))
        finally:
            finished.set()

    thread, errors = background(generate)
    try:
        assert entered.wait(1)
        assert not finished.wait(0.03), "rollout slipped through after Vt advanced"
        plan = coordinator.begin_sync(timeout_s=1)
        assert not versions
        coordinator.end_sync(plan, succeeded=True)
        thread.join(2)
        assert not errors and versions == [1]
    finally:
        coordinator.close()
        thread.join(2)
        if coordinator.active_rollouts:
            coordinator.end_rollout()


def test_concurrent_bridge_sync_calls_share_one_copy():
    copying, release = threading.Event(), threading.Event()
    sync = FakeWeightSync(started_events={1: copying}, gates={1: release})
    bridge = DualEngineBridge(train_engine=FakeTrainEngine(), rollout_engine=FakeRolloutEngine(), weight_sync=sync)
    bridge.initialize()
    bridge.train(ready_batch())
    results = []
    first, errors1 = background(lambda: results.append(bridge.sync(timeout_s=2)))
    second = None
    try:
        assert copying.wait(1)
        second, errors2 = background(lambda: results.append(bridge.sync(timeout_s=2)))
        wait_for(lambda: bridge.coordinator._sync_waiters == 1)
        assert bridge.coordinator.sync_active
        assert bridge.coordinator.rollout_policy_version == 0
        release.set()
        first.join(2)
        second.join(2)
        assert not errors1 + errors2
        assert not first.is_alive() and not second.is_alive()
        assert sum(result is not None for result in results) == 1
        assert sync.transfers == [(0, 0), (0, 1)]
    finally:
        release.set()
        first.join(2)
        if second is not None:
            second.join(2)
        bridge.close()


def test_sync_mode_blocks_training_while_rollout_uses_the_shared_device():
    generating, release = threading.Event(), threading.Event()
    rollout = FakeRolloutEngine(started_events={0: generating}, gates={0: release})
    train = FakeTrainEngine()
    bridge = DualEngineBridge(train_engine=train, rollout_engine=rollout, weight_sync=FakeWeightSync(),
                              mode=DeviceMode.SYNC, train_devices=("cpu",), rollout_devices=("cpu",))
    bridge.initialize()
    thread, errors = background(lambda: bridge.rollout_session(prompts(1)[0], timeout_s=2))
    try:
        assert generating.wait(1)
        with pytest.raises(TimeoutError):
            bridge.train(ready_batch(), timeout_s=0.02)
        assert train.calls == 0
    finally:
        release.set()
        thread.join(2)
        bridge.close()
    assert not errors and not thread.is_alive()


@pytest.mark.parametrize("operation", ["rollout", "train", "sync"])
def test_closing_model_admission_wakes_every_kind_of_waiter(operation):
    coordinator = PolicyPipelineCoordinator(train_policy_version=1)
    coordinator.attach(config(), DeviceMode.SYNC)
    coordinator.begin_rollout()
    waiting = threading.Event()
    original_wait = coordinator._cond.wait

    def observed_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    coordinator._cond.wait = observed_wait
    calls = {
        "rollout": lambda: coordinator.begin_rollout(timeout_s=10),
        "train": lambda: coordinator.begin_train(0, timeout_s=10),
        "sync": lambda: coordinator.begin_sync(timeout_s=10),
    }
    thread, errors = background(calls[operation])
    try:
        assert waiting.wait(1)
        coordinator.close()
        thread.join(0.5)
        assert not thread.is_alive(), "close did not wake a model admission waiter"
        assert len(errors) == 1 and isinstance(errors[0], PipelineClosed)
    finally:
        coordinator.close()
        coordinator.end_rollout()
        thread.join(2)


def test_k_covers_source_reads_and_blocked_put_without_holding_model_leases():
    training, release = threading.Event(), threading.Event()
    train = FakeTrainEngine(started_events={0: training}, gates={0: release})

    class Source:
        reads = 0

        def __iter__(self):
            assert pipe.inflight_active > 0
            return self

        def __next__(self):
            assert pipe.inflight_active > 0
            self.reads += 1
            return AsyncPrompt(prompt_id=str(self.reads), input_tokens=(1, 2))

    source = Source()
    pipe = pipeline(data=source, train=train, cfg=config(queue_capacity=1, max_inflight_rollouts=2,
                    weight_sync_interval_updates=100, max_policy_lag=100))
    thread, errors = background(pipe.run)
    try:
        assert training.wait(1)
        wait_for(lambda: source.reads == 4 and pipe._queue.depth == 1
                 and pipe._bridge._rollout_engine.calls == 4 and pipe._coordinator.active_rollouts == 0)
        # One consumer batch + Q queued batches + K accepted tasks; no fifth read.
        assert pipe.inflight_active == 2
        time.sleep(0.02)
        assert source.reads == 4
        pipe.request_stop()
    finally:
        release.set()
        pipe.request_stop()
        thread.join(4)
        pipe.close()
    assert not errors and not thread.is_alive()
    assert_drained(pipe)


@pytest.mark.parametrize("where", ["executor_init", "submit"])
def test_executor_failure_releases_source_permit_and_closes_engines(monkeypatch, where):
    import areno.experimental.async_policy.pipeline as module

    boom = ValueError(where)
    original = module.ThreadPoolExecutor

    class BrokenExecutor(original):
        def __init__(self, *args, **kwargs):
            if where == "executor_init":
                raise boom
            super().__init__(*args, **kwargs)

        def submit(self, *args, **kwargs):
            raise boom

    monkeypatch.setattr(module, "ThreadPoolExecutor", BrokenExecutor)
    train, rollout = FakeTrainEngine(), FakeRolloutEngine()
    pipe = pipeline(train=train, rollout=rollout)
    with pytest.raises(ValueError) as caught:
        pipe.run()
    assert caught.value is boom
    assert_drained(pipe)
    assert train.closed and rollout.closed


def test_producer_start_failure_still_closes_initialized_engines(monkeypatch):
    original = threading.Thread.start
    boom = RuntimeError("cannot start producer")

    def fail_start(thread):
        if thread.name == "areno-async-producer":
            raise boom
        original(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    train, rollout = FakeTrainEngine(), FakeRolloutEngine()
    pipe = pipeline(train=train, rollout=rollout)
    with pytest.raises(RuntimeError) as caught:
        pipe.run()
    assert caught.value is boom
    assert_drained(pipe)
    assert train.closed and rollout.closed


def test_partial_initialization_closes_every_attempted_engine():
    boom = ValueError("partial initialization")

    class Rollout(FakeRolloutEngine):
        def initialize(self, *, timeout_s=None):
            super().initialize(timeout_s=timeout_s)
            raise boom

    train, rollout = FakeTrainEngine(), Rollout()
    pipe = pipeline(train=train, rollout=rollout)
    with pytest.raises(ValueError) as caught:
        pipe.run()
    assert caught.value is boom
    assert train.closed and rollout.closed and rollout.calls == 0
    assert pipe.report().exit_reason == "failed"
    assert_drained(pipe)


def test_invalid_initialization_timeout_does_not_leave_initialization_active():
    bridge = DualEngineBridge(train_engine=FakeTrainEngine(), rollout_engine=FakeRolloutEngine(), weight_sync=FakeWeightSync())
    with pytest.raises(ValueError):
        bridge.initialize(timeout_s=-1)
    bridge.initialize()
    bridge.close(timeout_s=0.2)


def test_initial_transfer_failure_prevents_source_and_model_use():
    boom = ValueError("initial transfer failed")
    rollout, train = FakeRolloutEngine(), FakeTrainEngine()
    pipe = pipeline(rollout=rollout, train=train, sync=FakeWeightSync(failures={0: boom}))
    with pytest.raises(ValueError) as caught:
        pipe.run()
    assert caught.value is boom
    assert rollout.calls == train.calls == pipe.report().produced_batches == 0
    assert rollout.closed and train.closed
    assert_drained(pipe)


def test_initialization_failure_uses_one_cleanup_attempt_and_preserves_root_error():
    boom = ValueError("initialization failed before rollout initialization")

    class Train(FakeTrainEngine):
        close_calls = 0
        refuse_close = True

        def initialize(self, *, timeout_s=None):
            raise boom

        def close(self, *, timeout_s=None):
            self.close_calls += 1
            if self.refuse_close:
                raise TimeoutError("cleanup deadline expired")
            super().close(timeout_s=timeout_s)

    train = Train()
    pipe = pipeline(data=[], train=train)
    try:
        with pytest.raises(ValueError) as caught:
            pipe.run()
        assert caught.value is boom
        assert train.close_calls == 1, "run silently retried cleanup with a fresh shutdown budget"
        assert pipe.report().secondary_failures
    finally:
        train.refuse_close = False
        pipe.close()
    assert_drained(pipe)


def test_first_failure_survives_shutdown_timeout_with_live_resources_reported():
    started, release = threading.Event(), threading.Event()
    boom = ValueError("first train failure")
    pipe = pipeline(data=prompts(2), cfg=config(max_inflight_rollouts=1, shutdown_timeout_s=0.02),
                    rollout=FakeRolloutEngine(started_events={1: started}, gates={1: release}),
                    train=FakeTrainEngine(gates={0: started}, failures={0: boom}))
    try:
        with pytest.raises(ValueError) as caught:
            pipe.run()
        assert caught.value is boom
        report = pipe.report()
        assert report.exit_reason == "failed" and report.producer_alive and report.inflight_active == 1
        assert any("ShutdownTimeout" in failure for failure in report.secondary_failures)
    finally:
        release.set()
        pipe.close(timeout_s=2)
    assert_drained(pipe)


def test_cooperative_reward_stop_releases_all_work_without_model_lease():
    scoring = threading.Event()

    def reward(record):
        scoring.set()
        assert pipe.stop_event.wait(2)
        return 0.0

    pipe = pipeline(data=prompts(2), reward=reward)
    thread, errors = background(pipe.run)
    try:
        assert scoring.wait(1)
        assert not pipe._coordinator.train_active
        wait_for(lambda: pipe._coordinator.active_rollouts == 0)
        pipe.abort()
    finally:
        pipe.abort()
        thread.join(3)
        pipe.close()
    assert not errors and not thread.is_alive()
    assert pipe.report().exit_reason == "aborted" and pipe.report().steps == 0
    assert_drained(pipe)


def test_source_finalizer_failure_is_reported_and_does_not_leak_work():
    boom = ValueError("source close failed")

    def source():
        try:
            yield from prompts(3)
        finally:
            raise boom

    pipe = pipeline(data=source())
    with pytest.raises(ValueError) as caught:
        pipe.run()
    assert caught.value is boom
    assert_drained(pipe)


def test_cleanup_failure_does_not_skip_other_engine_and_can_be_retried():
    boom = ValueError("rollout close failed")

    class Rollout(FakeRolloutEngine):
        close_calls = 0

        def close(self, *, timeout_s=None):
            self.close_calls += 1
            if self.close_calls == 1:
                raise boom
            super().close(timeout_s=timeout_s)

    train, rollout = FakeTrainEngine(), Rollout()
    pipe = pipeline(data=[], train=train, rollout=rollout)
    with pytest.raises(ValueError) as caught:
        pipe.run()
    assert caught.value is boom
    assert train.closed and not rollout.closed
    pipe.close()
    assert rollout.closed and pipe.report().state == "closed"
    assert pipe.report().exit_reason == "failed"
    assert_drained(pipe)


def test_rollout_buffer_and_reward_mutation_cannot_change_published_rows():
    import torch

    second_generated = threading.Event()
    shared_feature = torch.tensor([5.0])

    class ReusingRollout(FakeRolloutEngine):
        route = torch.zeros((3, 1, 1), dtype=torch.int16)
        row = RolloutSequence(resp_tokens=[3, 5], resp_logprobs=[-0.5, -0.4])

        def generate(self, prompt, version, *, timeout_s=None):
            self.route.fill_(int(prompt.prompt_id) + 1)
            self.row.resp_tokens[0] = 3 + int(prompt.prompt_id)
            self.row.routed_experts = self.route
            if prompt.prompt_id == "1":
                second_generated.set()
            return RolloutResult(sequences=[self.row])

    def source():
        for index in range(2):
            shared_feature.fill_(5 + index)
            yield AsyncPrompt(prompt_id=str(index), input_tokens=(1, 2), record={"features": {"x": shared_feature}})

    def reward(record):
        if record.metadata["prompt_id"] == "0":
            assert second_generated.wait(2)
            assert record.tokens[-2] == 3
        record.source_record["features"]["x"].fill_(999)
        return 0.0

    train = FakeTrainEngine()
    pipe = pipeline(data=source(), train=train, rollout=ReusingRollout(), reward=reward)
    pipe.run()
    by_id = {batch.prompt_ids[0]: batch for batch in train.batches}
    for index in range(2):
        row = by_id[str(index)].sequences[0]
        assert row.tokens[-2] == 3 + index
        assert row.routed_experts.unique().tolist() == [index + 1]
        assert row.features["x"].tolist() == [5 + index]
    assert_drained(pipe)


@pytest.mark.parametrize("bad_payload", ["leaf_graph", "non_cpu_route", "opaque"])
def test_all_payload_paths_enforce_cpu_snapshot_contract(bad_payload):
    import torch

    result = FakeRolloutEngine().generate(prompts(1)[0], 0)
    if bad_payload == "leaf_graph":
        result.sequences[0].routed_experts = torch.ones(1, requires_grad=True)
    elif bad_payload == "non_cpu_route":
        result.sequences[0].routed_experts = torch.empty(1, device="meta")
    else:
        result.sequences[0].routed_experts = object()
    with pytest.raises(BatchContractError):
        build_batch_envelope(run_id="x", batch_id="b", epoch=0, policy_version=0,
                             prompt_items=prompts(1), rollout_results=[result], rewards=[0, 1], eos_token_id=2)


def test_failed_run_wait_metric_includes_empty_polls():
    gate, polled = threading.Event(), threading.Event()
    boom = ValueError("rollout fails after consumer waits")
    pipe = pipeline(data=prompts(1), rollout=FakeRolloutEngine(gates={0: gate}, failures={0: boom}))
    original_get = pipe._queue.get
    waits = []

    def observed_get(*, timeout_s=None):
        start = time.monotonic()
        try:
            return original_get(timeout_s=timeout_s)
        except QueueEmpty:
            waits.append(time.monotonic() - start)
            if len(waits) == 5:
                polled.set()
            raise

    pipe._queue.get = observed_get
    thread, errors = background(pipe.run)
    try:
        assert polled.wait(1)
    finally:
        gate.set()
        thread.join(3)
        pipe.close()
    assert errors == [boom] and not thread.is_alive()
    assert pipe.report().batch_wait_s >= sum(waits[:5]) * 0.9
    assert_drained(pipe)
