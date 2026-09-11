from __future__ import annotations

import threading

import pytest

from areno.experimental.async_policy import PipelineClosed, PolicyPipelineCoordinator, PolicySyncPlan


def test_rollout_and_training_leases_can_overlap() -> None:
    coordinator = PolicyPipelineCoordinator()

    assert coordinator.begin_rollout() == 0
    assert coordinator.begin_train(timeout_s=0.01) == 0

    assert coordinator.end_train(stepped=True) == 1
    assert coordinator.rollout_policy_version == 0
    coordinator.end_rollout()


def test_sync_waits_for_active_training_and_rollout() -> None:
    coordinator = PolicyPipelineCoordinator()
    coordinator.begin_rollout()
    coordinator.begin_train()
    result: list[PolicySyncPlan | None] = []
    started = threading.Event()

    def acquire_sync() -> None:
        started.set()
        result.append(coordinator.begin_sync())

    waiter = threading.Thread(target=acquire_sync)
    waiter.start()
    assert started.wait(timeout=1.0)
    waiter.join(timeout=0.05)
    assert waiter.is_alive()

    coordinator.end_train(stepped=True)
    waiter.join(timeout=0.05)
    assert waiter.is_alive()

    coordinator.end_rollout()
    waiter.join(timeout=1.0)
    assert not waiter.is_alive()
    assert result == [PolicySyncPlan(source_version=0, target_version=1)]
    coordinator.end_sync(result[0], succeeded=True)


def test_new_work_waits_until_policy_sync_finishes() -> None:
    coordinator = PolicyPipelineCoordinator(train_policy_version=2, rollout_policy_version=1)
    plan = coordinator.begin_sync()
    assert plan == PolicySyncPlan(source_version=1, target_version=2)
    rollout_versions: list[int] = []

    waiter = threading.Thread(target=lambda: rollout_versions.append(coordinator.begin_rollout()))
    waiter.start()
    waiter.join(timeout=0.05)
    assert waiter.is_alive()

    coordinator.end_sync(plan, succeeded=True)
    waiter.join(timeout=1.0)
    assert not waiter.is_alive()
    assert rollout_versions == [2]
    coordinator.end_rollout()


def test_failed_sync_does_not_advance_rollout_version() -> None:
    coordinator = PolicyPipelineCoordinator(train_policy_version=3, rollout_policy_version=1)
    plan = coordinator.begin_sync()
    assert plan is not None

    assert coordinator.end_sync(plan, succeeded=False) == 1
    assert coordinator.rollout_policy_version == 1


def test_sync_is_noop_when_versions_match() -> None:
    coordinator = PolicyPipelineCoordinator(train_policy_version=2, rollout_policy_version=2)

    assert coordinator.begin_sync() is None
    assert coordinator.begin_train(timeout_s=0.01) == 2
    coordinator.end_train(stepped=False)


def test_close_unblocks_waiting_work() -> None:
    coordinator = PolicyPipelineCoordinator(train_policy_version=1, rollout_policy_version=0)
    plan = coordinator.begin_sync()
    assert plan is not None
    errors: list[Exception] = []

    def acquire_rollout() -> None:
        try:
            coordinator.begin_rollout()
        except Exception as error:
            errors.append(error)

    waiter = threading.Thread(target=acquire_rollout)
    waiter.start()
    waiter.join(timeout=0.05)
    assert waiter.is_alive()

    coordinator.close()
    waiter.join(timeout=1.0)
    assert not waiter.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], PipelineClosed)
    coordinator.end_sync(plan, succeeded=False)


def test_invalid_lifecycle_and_version_transitions_fail_loudly() -> None:
    coordinator = PolicyPipelineCoordinator()

    with pytest.raises(RuntimeError, match="no active rollout"):
        coordinator.end_rollout()
    with pytest.raises(RuntimeError, match="no active training"):
        coordinator.end_train(stepped=True)

    coordinator.begin_train()
    with pytest.raises(ValueError, match="must advance"):
        coordinator.end_train(stepped=True, policy_version=0)

    with pytest.raises(ValueError, match="cannot exceed"):
        PolicyPipelineCoordinator(train_policy_version=1, rollout_policy_version=2)
