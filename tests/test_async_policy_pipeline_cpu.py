from __future__ import annotations

import threading

import pytest

from areno.experimental.async_policy import (
    AsyncPolicyConfig,
    AsyncPolicyPipeline,
    AsyncPrompt,
    FakeRolloutEngine,
    FakeTrainEngine,
    FakeWeightSync,
    PolicyPipelineCoordinator,
)


def _prompts(count: int) -> list[AsyncPrompt]:
    return [AsyncPrompt(prompt_id=f"p{i}", input_tokens=(i + 1, i + 2)) for i in range(count)]


def _reward(prompt: AsyncPrompt, sample: int) -> float:
    return float(sample)


def _config(**overrides) -> AsyncPolicyConfig:
    base = dict(
        queue_capacity=1,
        max_inflight_rollouts=1,
        weight_sync_interval_updates=1,
        max_policy_lag=1,
        max_steps=None,
        poll_interval_s=0.01,
        operation_timeout_s=5.0,
        shutdown_timeout_s=5.0,
    )
    base.update(overrides)
    return AsyncPolicyConfig(**base)


def _pipeline(config, data, rollout, train, sync, *, coordinator=None, reward=_reward):
    return AsyncPolicyPipeline(
        config=config,
        data_source=data,
        rollout_engine=rollout,
        train_engine=train,
        weight_sync=sync,
        reward_fn=reward,
        coordinator=coordinator,
    )


def test_closed_loop_proves_real_overlap() -> None:
    # Q=1, K=1, C=1. Hold rollout call 1 and train call 0 on gates so the two
    # stages are simultaneously active; the timeline must show the overlap.
    rollout = FakeRolloutEngine()
    rollout.started_events = {1: threading.Event()}
    rollout.gates = {1: threading.Event()}
    train = FakeTrainEngine()
    train.started_events = {0: threading.Event()}
    train.gates = {0: threading.Event()}
    sync = FakeWeightSync()

    pipe = _pipeline(_config(max_steps=2, max_policy_lag=2), _prompts(2), rollout, train, sync)
    run_thread = threading.Thread(target=pipe.run)
    run_thread.start()

    try:
        assert rollout.started_events[1].wait(timeout=5.0), "second rollout never started"
        assert train.started_events[0].wait(timeout=5.0), "first train never started"
        # Both gated calls are now executing at the same time.
    finally:
        train.gates[0].set()
        rollout.gates[1].set()

    run_thread.join(timeout=10.0)
    assert not run_thread.is_alive()

    report = pipe.report()
    assert report.steps == 2
    assert report.exit_reason == "max_steps"
    assert report.syncs >= 1
    assert report.overlaps("rollout", "train"), "rollout and train never overlapped"
    assert not pipe.producer_alive


def test_data_exhaustion_drains_and_exits() -> None:
    pipe = _pipeline(
        _config(max_policy_lag=5, weight_sync_interval_updates=1),
        _prompts(3),
        FakeRolloutEngine(),
        FakeTrainEngine(),
        FakeWeightSync(),
    )
    report = pipe.run()

    assert report.exit_reason == "data_exhausted"
    assert report.steps == 3
    assert report.batches_seen == 3
    assert report.produced_batches == 3
    assert not pipe.producer_alive


def test_max_steps_stops_early() -> None:
    pipe = _pipeline(
        _config(max_policy_lag=5, max_steps=2),
        _prompts(10),
        FakeRolloutEngine(),
        FakeTrainEngine(),
        FakeWeightSync(),
    )
    report = pipe.run()

    assert report.exit_reason == "max_steps"
    assert report.steps == 2
    assert not pipe.producer_alive


def test_reward_error_propagates() -> None:
    def bad_reward(prompt: AsyncPrompt, sample: int) -> float:
        if prompt.prompt_id == "p1":
            raise RuntimeError("reward exploded")
        return float(sample)

    pipe = _pipeline(
        _config(max_policy_lag=5),
        _prompts(4),
        FakeRolloutEngine(),
        FakeTrainEngine(),
        FakeWeightSync(),
        reward=bad_reward,
    )
    with pytest.raises(RuntimeError, match="reward exploded"):
        pipe.run()
    assert not pipe.producer_alive


def test_rollout_failure_propagates() -> None:
    rollout = FakeRolloutEngine()
    rollout.failures = {1: RuntimeError("rollout engine died")}
    pipe = _pipeline(_config(max_policy_lag=5), _prompts(4), rollout, FakeTrainEngine(), FakeWeightSync())

    with pytest.raises(RuntimeError, match="rollout engine died"):
        pipe.run()
    assert not pipe.producer_alive


def test_sync_failure_does_not_advance_rollout_version() -> None:
    coordinator = PolicyPipelineCoordinator()
    sync = FakeWeightSync()
    sync.failures = {0: RuntimeError("nccl transfer failed")}
    pipe = _pipeline(
        _config(max_policy_lag=5),
        _prompts(3),
        FakeRolloutEngine(),
        FakeTrainEngine(),
        sync,
        coordinator=coordinator,
    )

    with pytest.raises(RuntimeError, match="nccl transfer failed"):
        pipe.run()
    assert coordinator.rollout_policy_version == 0
    assert not pipe.producer_alive


def test_request_stop_is_clean() -> None:
    train = FakeTrainEngine()
    train.started_events = {0: threading.Event()}
    pipe = _pipeline(_config(max_policy_lag=5), _prompts(20), FakeRolloutEngine(), train, FakeWeightSync())

    run_thread = threading.Thread(target=pipe.run)
    run_thread.start()
    try:
        assert train.started_events[0].wait(timeout=5.0)
        pipe.request_stop()
    finally:
        pipe.request_stop()
    run_thread.join(timeout=10.0)

    assert not run_thread.is_alive()
    assert pipe.report().exit_reason == "stopped"
    assert not pipe.producer_alive


def test_double_close_is_idempotent_and_run_after_close_fails() -> None:
    pipe = _pipeline(_config(), _prompts(1), FakeRolloutEngine(), FakeTrainEngine(), FakeWeightSync())
    pipe.close()
    pipe.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipe.run()


def test_k_greater_than_one_runs_multiple_production_tasks() -> None:
    # K=2 must actually admit two production tasks at once. Hold both rewards
    # on gates so both tasks are in flight simultaneously and the inflight
    # limiter reports an active count of two.
    reward_started = {0: threading.Event(), 1: threading.Event()}
    reward_gate = {0: threading.Event(), 1: threading.Event()}

    def gated_reward(prompt: AsyncPrompt, sample: int) -> float:
        index = int(prompt.prompt_id[1:])
        started = reward_started.get(index)
        if started is not None:
            started.set()
            reward_gate[index].wait(timeout=5.0)
        return float(sample)

    pipe = _pipeline(
        _config(
            queue_capacity=2,
            max_inflight_rollouts=2,
            max_policy_lag=5,
            max_steps=2,
        ),
        _prompts(4),
        FakeRolloutEngine(),
        FakeTrainEngine(),
        FakeWeightSync(),
        reward=gated_reward,
    )
    run_thread = threading.Thread(target=pipe.run)
    run_thread.start()
    try:
        assert reward_started[0].wait(timeout=5.0)
        assert reward_started[1].wait(timeout=5.0)
        assert pipe.inflight_active == 2
    finally:
        reward_gate[0].set()
        reward_gate[1].set()
        run_thread.join(timeout=10.0)

    assert not run_thread.is_alive()
    report = pipe.report()
    assert report.inflight_peak == 2
    assert report.steps == 2
    assert not pipe.producer_alive


def test_non_stepped_update_does_not_advance_version_or_sync() -> None:
    # A batch that performs no optimizer update (zero gradient, an explicitly
    # skipped update) must not advance Vt and must not trigger a sync.
    coordinator = PolicyPipelineCoordinator()
    train = FakeTrainEngine()
    train.stepped_by_call = {0: False, 1: False}
    pipe = _pipeline(
        _config(max_policy_lag=5, weight_sync_interval_updates=1, max_steps=1),
        _prompts(2),
        FakeRolloutEngine(),
        train,
        FakeWeightSync(),
        coordinator=coordinator,
    )
    report = pipe.run()

    assert report.steps == 0
    assert report.batches_seen == 2
    assert report.batches_trained == 0
    assert report.syncs == 0
    assert coordinator.train_policy_version == 0
    assert coordinator.rollout_policy_version == 0
    assert {metric.decision for metric in report.batch_metrics} == {"trained"}
    assert all(metric.stepped is False for metric in report.batch_metrics)


def test_batch_metrics_capture_versions_lag_and_decision() -> None:
    pipe = _pipeline(
        _config(max_policy_lag=5, weight_sync_interval_updates=1),
        _prompts(3),
        FakeRolloutEngine(),
        FakeTrainEngine(),
        FakeWeightSync(),
    )
    report = pipe.run()

    assert len(report.batch_metrics) == 3
    for metric in report.batch_metrics:
        assert metric.run_id == "async-run"
        assert metric.decision == "trained"
        assert metric.lag == metric.train_version - metric.batch_version
        assert metric.rollout_version <= metric.train_version
        assert metric.version_after == metric.train_version + 1
        assert metric.train_s >= 0.0
        assert metric.wait_s >= 0.0


def test_stale_batch_is_dropped_and_sync_recovers_progress() -> None:
    # C=100 delays the normal sync, so batch 1 is produced against Vr=0 while
    # training has already advanced to Vt=1. With max_policy_lag=0 it must be
    # dropped whole, a sync must be requested by the consumer, and the run must
    # still finish a real update.
    coordinator = PolicyPipelineCoordinator()
    rollout = FakeRolloutEngine()
    rollout.gates = {1: threading.Event()}
    rollout.started_events = {1: threading.Event()}
    pipe = _pipeline(
        _config(
            queue_capacity=2,
            weight_sync_interval_updates=100,
            max_policy_lag=0,
            max_steps=None,
        ),
        _prompts(3),
        rollout,
        FakeTrainEngine(),
        FakeWeightSync(),
        coordinator=coordinator,
    )

    def release_after_train() -> None:
        # Let batch 0 train and advance Vt, then release the stale rollout.
        import time

        deadline = time.monotonic() + 5.0
        while coordinator.train_policy_version == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        rollout.gates[1].set()

    releaser = threading.Thread(target=release_after_train)
    releaser.start()
    report = pipe.run()
    releaser.join(timeout=5.0)

    assert report.steps >= 1
    assert report.batches_dropped_stale >= 1
    assert report.syncs >= 1
    assert coordinator.rollout_policy_version >= 1
    assert not pipe.producer_alive
