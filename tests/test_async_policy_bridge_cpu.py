from __future__ import annotations

import threading

import pytest

from areno.experimental.async_policy import (
    BridgeConfigError,
    BridgeStateError,
    DeviceMode,
    DualEngineBridge,
    FakeRolloutEngine,
    FakeTrainEngine,
    FakeWeightSync,
    PolicyPipelineCoordinator,
)
from areno.experimental.async_policy.fake_backend import AsyncPrompt


def _prompt() -> AsyncPrompt:
    return AsyncPrompt(prompt_id="p0", input_tokens=(1, 2))


def test_independent_mode_allows_concurrent_rollout_and_training() -> None:
    rollout = FakeRolloutEngine()
    rollout.started_events = {0: threading.Event()}
    rollout.gates = {0: threading.Event()}
    train = FakeTrainEngine()
    train.started_events = {0: threading.Event()}

    bridge = DualEngineBridge(
        train_engine=train,
        rollout_engine=rollout,
        weight_sync=FakeWeightSync(),
        mode=DeviceMode.INDEPENDENT,
    )
    bridge.initialize()

    session_result: list[object] = []
    session_thread = threading.Thread(target=lambda: session_result.append(bridge.rollout_session(_prompt())))
    session_thread.start()
    assert rollout.started_events[0].wait(timeout=5.0)

    # Both leases are held at once: rollout session is active and training runs.
    stepped = bridge.train(object())
    assert stepped is True
    assert bridge.active_rollout_sessions == 1
    assert bridge.train_metrics["stepped"] is True

    rollout.gates[0].set()
    session_thread.join(timeout=5.0)
    assert not session_thread.is_alive()
    assert len(session_result) == 1


def test_sync_mode_rejects_training_during_active_rollout() -> None:
    rollout = FakeRolloutEngine()
    rollout.started_events = {0: threading.Event()}
    rollout.gates = {0: threading.Event()}

    bridge = DualEngineBridge(
        train_engine=FakeTrainEngine(),
        rollout_engine=rollout,
        weight_sync=FakeWeightSync(),
        mode=DeviceMode.SYNC,
    )
    bridge.initialize()

    session_thread = threading.Thread(target=lambda: bridge.rollout_session(_prompt()))
    session_thread.start()
    try:
        assert rollout.started_events[0].wait(timeout=5.0)
        with pytest.raises(RuntimeError, match="active rollout session"):
            bridge.train(object())
    finally:
        rollout.gates[0].set()
        session_thread.join(timeout=5.0)


def test_overlapping_device_groups_are_rejected() -> None:
    with pytest.raises(BridgeConfigError, match="disjoint"):
        DualEngineBridge(
            train_engine=FakeTrainEngine(),
            rollout_engine=FakeRolloutEngine(),
            weight_sync=FakeWeightSync(),
            train_devices=("cuda:0",),
            rollout_devices=("cuda:0",),
        )


def test_sync_returns_false_when_versions_match() -> None:
    bridge = DualEngineBridge(
        train_engine=FakeTrainEngine(),
        rollout_engine=FakeRolloutEngine(),
        weight_sync=FakeWeightSync(),
    )
    bridge.initialize()
    assert bridge.sync() is False


def test_lifecycle_requires_initialize_and_close_is_idempotent() -> None:
    bridge = DualEngineBridge(
        train_engine=FakeTrainEngine(),
        rollout_engine=FakeRolloutEngine(),
        weight_sync=FakeWeightSync(),
    )
    with pytest.raises(BridgeStateError, match="initialized"):
        bridge.train(object())

    bridge.initialize()
    bridge.initialize()  # idempotent
    bridge.close()
    bridge.close()  # idempotent
    with pytest.raises(BridgeStateError, match="closed"):
        bridge.train(object())


def test_sync_publishes_version_after_training() -> None:
    coordinator = PolicyPipelineCoordinator()
    bridge = DualEngineBridge(
        train_engine=FakeTrainEngine(),
        rollout_engine=FakeRolloutEngine(),
        weight_sync=FakeWeightSync(),
        coordinator=coordinator,
    )
    bridge.initialize()

    assert bridge.train(object()) is True
    assert coordinator.train_policy_version == 1
    assert bridge.sync() is True
    assert coordinator.rollout_policy_version == 1
