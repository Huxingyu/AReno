from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from areno.api.models import RolloutResult, RolloutSequence
from areno.experimental.async_policy import (
    BatchContractError,
    BoundedReadyQueue,
    InflightClosed,
    InflightLimiter,
    InflightLimitExceeded,
    QueueAborted,
    QueueClosed,
    QueueEmpty,
    QueueFull,
    build_batch_envelope,
)


def _item(prompt_id: str, input_tokens: list[int], prompt: str = "q") -> SimpleNamespace:
    return SimpleNamespace(
        prompt_id=prompt_id,
        input_tokens=input_tokens,
        prompt=prompt,
        record={},
        solutions=None,
    )


def _result(*sequences: tuple[list[int], list[float]]) -> RolloutResult:
    return RolloutResult(
        sequences=[RolloutSequence(resp_tokens=tokens, resp_logprobs=logprobs) for tokens, logprobs in sequences]
    )


def _build(prompt_items, rollout_results, rewards):
    return build_batch_envelope(
        run_id="run-1",
        batch_id="batch-1",
        epoch=0,
        policy_version=3,
        prompt_items=prompt_items,
        rollout_results=rollout_results,
        rewards=rewards,
        eos_token_id=2,
        produced_at=1.0,
    )


def test_envelope_groups_and_advantages_match_reference() -> None:
    items = [_item("p0", [10, 11]), _item("p1", [20])]
    results = [
        _result(([1], [0.1]), ([2, 3], [0.2, 0.3])),
        _result(([4], [0.4]), ([5], [0.5])),
    ]
    envelope = _build(items, results, rewards=[1.0, 0.0, 5.0, 5.0])

    assert envelope.prompt_count == 2
    assert envelope.completion_count == 4
    assert envelope.response_token_count == 1 + 2 + 1 + 1
    assert envelope.group_offsets == (0, 2, 4)
    assert envelope.group_count == 2

    assert len(envelope.group_slice(1)) == 2
    with pytest.raises(IndexError):
        envelope.group_slice(2)

    # prompt + response tokens, zero prefix on logprobs, structured prompt_len.
    first = envelope.sequences[0]
    assert list(first.tokens) == [10, 11, 1]
    assert list(first.logprobs) == [0.0, 0.0, 0.1]
    assert first.prompt_len == 2
    assert first.eos_token_id == 2

    # advantages are standardized within each prompt group.
    group0 = envelope.group_stats[0].rewards
    assert group0 == (1.0, 0.0)
    assert envelope.sequences[0].scalar_advantage == pytest.approx(1.0)
    assert envelope.sequences[1].scalar_advantage == pytest.approx(-1.0)
    # equal-reward groups get zero advantage (the degenerate smoke case).
    assert envelope.sequences[2].scalar_advantage == pytest.approx(0.0)
    assert envelope.sequences[3].scalar_advantage == pytest.approx(0.0)


def test_envelope_is_immutable() -> None:
    envelope = _build([_item("p0", [1])], [_result(([2], [0.1]))], rewards=[1.0])
    with pytest.raises(Exception):
        envelope.policy_version = 99  # type: ignore[misc]


def test_rejects_empty_group() -> None:
    with pytest.raises(BatchContractError, match="no completions"):
        _build([_item("p0", [1])], [_result()], rewards=[])


def test_rejects_reward_count_mismatch() -> None:
    with pytest.raises(BatchContractError, match="rewards count"):
        _build([_item("p0", [1])], [_result(([2], [0.1]), ([3], [0.2]))], rewards=[1.0])


def test_rejects_non_finite_reward() -> None:
    with pytest.raises(BatchContractError, match="not finite"):
        _build([_item("p0", [1])], [_result(([2], [0.1]))], rewards=[float("nan")])


def test_rejects_logprob_length_mismatch() -> None:
    result = RolloutResult(sequences=[RolloutSequence(resp_tokens=[1, 2], resp_logprobs=[0.1])])
    with pytest.raises(BatchContractError, match="equal length"):
        _build([_item("p0", [1])], [result], rewards=[1.0])


def test_rejects_gpu_or_graph_payload() -> None:
    result = _result(([2], [0.1]))
    result.sequences[0].routed_experts = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    with pytest.raises(BatchContractError, match="not CPU"):
        _build([_item("p0", [1])], [result], rewards=[1.0])


def test_queue_capacity_is_enforced() -> None:
    queue: BoundedReadyQueue[int] = BoundedReadyQueue(capacity=1)
    queue.put(1)
    with pytest.raises(QueueFull):
        queue.put(2, timeout_s=0.05)
    assert queue.depth == 1
    assert queue.get() == 1


def test_queue_drains_then_reports_eof() -> None:
    queue: BoundedReadyQueue[int] = BoundedReadyQueue(capacity=2)
    queue.put(1)
    queue.put(2)
    queue.close()
    assert queue.get() == 1
    assert queue.get() == 2
    with pytest.raises(QueueClosed):
        queue.get(timeout_s=0.05)
    with pytest.raises(QueueClosed):
        queue.put(3)


def test_queue_get_times_out_when_empty() -> None:
    queue: BoundedReadyQueue[int] = BoundedReadyQueue(capacity=1)
    with pytest.raises(QueueEmpty):
        queue.get(timeout_s=0.02)


def test_abort_wakes_blocked_get() -> None:
    queue: BoundedReadyQueue[int] = BoundedReadyQueue(capacity=1)
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            queue.get()
        except BaseException as exc:  # noqa: BLE001 - record what the waiter saw
            errors.append(exc)

    thread = threading.Thread(target=consume)
    thread.start()
    queue.abort()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert isinstance(errors[0], QueueAborted)


def test_failure_propagates_to_blocked_put_even_when_full() -> None:
    queue: BoundedReadyQueue[int] = BoundedReadyQueue(capacity=1)
    queue.put(1)
    errors: list[BaseException] = []
    entered = threading.Event()

    def produce() -> None:
        entered.set()
        try:
            queue.put(2)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=produce)
    thread.start()
    entered.wait(timeout=1.0)
    boom = RuntimeError("producer failed")
    queue.fail(boom)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert errors and errors[0] is boom


def test_inflight_limiter_bounds_and_releases() -> None:
    limiter = InflightLimiter(max_inflight=1)
    limiter.acquire()
    assert limiter.active == 1
    with pytest.raises(InflightLimitExceeded):
        limiter.acquire(timeout_s=0.02)
    limiter.release()
    limiter.acquire()
    limiter.release()
    assert limiter.peak == 1


def test_inflight_permit_releases_on_exception() -> None:
    limiter = InflightLimiter(max_inflight=2)
    with pytest.raises(ValueError):
        with limiter.permit():
            raise ValueError("boom")
    assert limiter.active == 0


def test_inflight_close_wakes_waiter() -> None:
    limiter = InflightLimiter(max_inflight=1)
    limiter.acquire()
    errors: list[BaseException] = []

    def waiter() -> None:
        try:
            limiter.acquire()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=waiter)
    thread.start()
    limiter.close()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert isinstance(errors[0], InflightClosed)
