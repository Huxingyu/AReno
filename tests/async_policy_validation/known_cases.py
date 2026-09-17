from __future__ import annotations

import threading

import pytest

from areno.experimental.async_policy import (
    AsyncPolicyConfig,
    AsyncPolicyPipeline,
    AsyncPrompt,
    DeviceMode,
    DualEngineBridge,
    FakeRolloutEngine,
    FakeTrainEngine,
    FakeWeightSync,
    PolicyPipelineCoordinator,
)


def test_begin_sync_honors_timeout() -> None:
    coordinator = PolicyPipelineCoordinator(train_policy_version=1)
    coordinator.begin_rollout()
    done = threading.Event()
    errors = []

    def acquire_sync() -> None:
        try:
            coordinator.begin_sync(timeout_s=0.02)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=acquire_sync)
    worker.start()
    try:
        completed_in_budget = done.wait(timeout=0.2)
    finally:
        coordinator.close()
        coordinator.end_rollout()
        worker.join(timeout=1)

    assert completed_in_budget, "20 ms begin_sync timeout was still blocked after 200 ms"
    assert isinstance(errors[0], TimeoutError)


def test_only_one_sync_lease_is_admitted() -> None:
    coordinator = PolicyPipelineCoordinator(train_policy_version=1)
    coordinator.begin_rollout()
    both_waiting = threading.Event()
    wait_count = 0
    original_wait = coordinator._cond.wait

    def observed_wait(timeout=None):
        nonlocal wait_count
        wait_count += 1
        if wait_count == 2:
            both_waiting.set()
        return original_wait(timeout)

    coordinator._cond.wait = observed_wait
    plans = []
    errors = []

    def acquire_sync() -> None:
        try:
            plans.append(coordinator.begin_sync(timeout_s=0.1))
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=acquire_sync) for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        assert both_waiting.wait(timeout=1)
        coordinator.end_rollout()
        for worker in workers:
            worker.join(timeout=0.2)
        admitted_before_any_end_sync = list(plans)
    finally:
        coordinator.close()
        for worker in workers:
            worker.join(timeout=1)
        if coordinator.sync_active and plans:
            coordinator.end_sync(plans[0], succeeded=False)

    assert len(admitted_before_any_end_sync) == 1, admitted_before_any_end_sync


def test_max_steps_does_not_report_success_with_live_producer() -> None:
    second_rollout_started = threading.Event()
    release_rollout = threading.Event()
    rollout = FakeRolloutEngine(
        started_events={1: second_rollout_started},
        gates={1: release_rollout},
    )
    train = FakeTrainEngine(gates={0: second_rollout_started})
    pipeline = AsyncPolicyPipeline(
        config=AsyncPolicyConfig(
            queue_capacity=1,
            max_steps=1,
            operation_timeout_s=2,
            shutdown_timeout_s=0.02,
            poll_interval_s=0.005,
        ),
        data_source=[AsyncPrompt(prompt_id=str(i), input_tokens=(1, 2)) for i in range(2)],
        rollout_engine=rollout,
        train_engine=train,
        weight_sync=FakeWeightSync(),
        reward_fn=lambda prompt, sample: float(sample),
    )
    try:
        report = pipeline.run()
        state_at_return = {
            "exit_reason": report.exit_reason,
            "producer_alive": pipeline.producer_alive,
            "inflight_active": pipeline.inflight_active,
        }
    finally:
        release_rollout.set()
        pipeline.close()
        pipeline._producer.join(timeout=2)

    assert not state_at_return["producer_alive"], state_at_return
    assert state_at_return["inflight_active"] == 0


def test_sync_mode_rejects_rollout_during_active_training() -> None:
    train_started = threading.Event()
    release_train = threading.Event()
    bridge = DualEngineBridge(
        train_engine=FakeTrainEngine(started_events={0: train_started}, gates={0: release_train}),
        rollout_engine=FakeRolloutEngine(),
        weight_sync=FakeWeightSync(),
        train_devices=("cuda:0",),
        rollout_devices=("cuda:0",),
        mode=DeviceMode.SYNC,
    )
    bridge.initialize()
    errors = []

    def train() -> None:
        try:
            bridge.train(object(), timeout_s=2)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=train)
    worker.start()
    try:
        assert train_started.wait(timeout=1)
        with pytest.raises((RuntimeError, TimeoutError)):
            bridge.rollout_session(AsyncPrompt(prompt_id="p", input_tokens=(1, 2)), timeout_s=0.02)
    finally:
        release_train.set()
        worker.join(timeout=2)
        bridge.close()
    assert not errors
