from __future__ import annotations

import threading
import time

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


def test_pending_sync_closes_admission_before_draining() -> None:
    # The prototype waited for active_rollouts == 0 before setting any sync
    # state, so continuously arriving rollouts starved synchronization. Here
    # admission must be closed as soon as begin_sync starts waiting.
    coordinator = PolicyPipelineCoordinator(train_policy_version=1, rollout_policy_version=0)
    coordinator.begin_rollout()
    result: list[PolicySyncPlan | None] = []

    waiter = threading.Thread(target=lambda: result.append(coordinator.begin_sync(timeout_s=2.0)))
    waiter.start()

    deadline = time.monotonic() + 1.0
    while not coordinator.sync_pending and time.monotonic() < deadline:
        time.sleep(0.005)
    assert coordinator.sync_pending

    # New admission is refused while the sync waits for the safe point.
    with pytest.raises(TimeoutError):
        coordinator.begin_rollout(timeout_s=0.05)

    coordinator.end_rollout()
    waiter.join(timeout=2.0)
    assert not waiter.is_alive()
    assert result == [PolicySyncPlan(source_version=0, target_version=1)]
    coordinator.end_sync(result[0], succeeded=True)
    assert not coordinator.sync_pending


def test_sync_reaches_safe_point_under_continuous_load() -> None:
    coordinator = PolicyPipelineCoordinator(train_policy_version=1, rollout_policy_version=0)
    stop = threading.Event()
    entered = threading.Event()

    def churn() -> None:
        while not stop.is_set():
            try:
                coordinator.begin_rollout(timeout_s=0.2)
            except TimeoutError:
                continue
            entered.set()
            try:
                coordinator.end_rollout()
            except Exception:
                pass

    workers = [threading.Thread(target=churn) for _ in range(4)]
    for worker in workers:
        worker.start()
    assert entered.wait(timeout=1.0)

    plan = coordinator.begin_sync(timeout_s=3.0)
    stop.set()
    for worker in workers:
        worker.join(timeout=2.0)
    assert plan == PolicySyncPlan(source_version=0, target_version=1)
    coordinator.end_sync(plan, succeeded=True)


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
