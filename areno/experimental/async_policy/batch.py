"""Immutable ready-batch contract for the async policy trainer (T02).

A ready batch is one *materialized prompt batch*: every completion of a prompt
is scored, the group-relative advantages are computed over the whole group, and
only then does the batch become an element of the ready queue. The envelope is
frozen and CPU-only so the producer thread can hand it to the training main
thread without sharing a computation graph, KV cache, or GPU tensor.

The construction mirrors ``PolicyOnlyTrainer._materialize_train_batch``: tokens
are ``prompt + response``, rollout logprobs get a zero prefix so lengths stay
aligned, and advantages are standardized with the same vectorized helper. The
equivalence is pinned by ``tests/test_async_policy_batch_cpu.py``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from areno.api.advantages import compute_batch_group_advantages
from areno.api.models import RolloutResult, TrainSequence


class BatchContractError(ValueError):
    """Raised when a batch violates the ready-batch envelope contract."""


@dataclass(frozen=True, slots=True)
class GroupStat:
    """Advantage statistics for one prompt group inside a ready batch."""

    group_index: int
    offset: int
    size: int
    rewards: tuple[float, ...]
    mean_reward: float
    std_reward: float


@dataclass(frozen=True, slots=True)
class BatchEnvelope:
    """One immutable, CPU-resident ready batch.

    ``sequences`` are ``TrainSequence`` rows in prompt-group order.
    ``group_offsets`` has ``len(groups) + 1`` entries so group ``g`` is the
    slice ``sequences[group_offsets[g]:group_offsets[g + 1]]``; groups are never
    split when a batch is produced or dropped.
    """

    run_id: str
    batch_id: str
    epoch: int
    policy_version: int
    prompt_ids: tuple[str, ...]
    sequences: tuple[TrainSequence, ...]
    group_offsets: tuple[int, ...]
    rewards: tuple[float, ...]
    group_stats: tuple[GroupStat, ...]
    prompt_count: int
    completion_count: int
    response_token_count: int
    produced_at: float

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def group_count(self) -> int:
        return len(self.group_offsets) - 1

    def group_slice(self, group_index: int) -> tuple[TrainSequence, ...]:
        if not 0 <= group_index < self.group_count:
            raise IndexError(f"group index out of range: {group_index}")
        start = self.group_offsets[group_index]
        end = self.group_offsets[group_index + 1]
        return self.sequences[start:end]


def _finite_float(value: Any, *, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BatchContractError(f"{where}: not a number: {value!r}") from exc
    if not math.isfinite(result):
        raise BatchContractError(f"{where}: not finite: {result!r}")
    return result


def _finite_int(value: Any, *, where: str) -> int:
    result = _finite_float(value, where=where)
    if result != int(result):
        raise BatchContractError(f"{where}: not an integer: {value!r}")
    return int(result)


def _check_cpu_payload(sequence: TrainSequence, *, where: str) -> None:
    """Reject values that would smuggle a graph or GPU state into the queue."""

    routed = sequence.routed_experts
    if routed is None:
        return
    grad_fn = getattr(routed, "grad_fn", None)
    if grad_fn is not None:
        raise BatchContractError(f"{where}: routed_experts is attached to a computation graph")
    if getattr(routed, "requires_grad", False):
        raise BatchContractError(f"{where}: routed_experts requires grad")
    device = getattr(routed, "device", None)
    device_type = getattr(device, "type", None)
    if device_type not in (None, "cpu"):
        raise BatchContractError(f"{where}: routed_experts lives on {device_type!r}, not CPU")


def build_batch_envelope(
    *,
    run_id: str,
    batch_id: str,
    epoch: int,
    policy_version: int,
    prompt_items: Sequence[Any],
    rollout_results: Sequence[RolloutResult],
    rewards: Sequence[float],
    eos_token_id: int,
    produced_at: float | None = None,
) -> BatchEnvelope:
    """Assemble one immutable ready batch from scored rollout results.

    ``prompt_items`` must expose ``input_tokens``, ``prompt``, ``record`` and
    ``solutions`` (the same contract as the synchronous trainer's prompt
    items). ``rewards`` holds one finite score per completion in
    prompt-group order and is scored *before* this call, so reward lifetime is
    owned by the producer stage rather than hidden inside materialization.
    """

    if not isinstance(run_id, str) or not run_id:
        raise BatchContractError("run_id must be a non-empty string")
    if not isinstance(batch_id, str) or not batch_id:
        raise BatchContractError("batch_id must be a non-empty string")
    if _finite_int(epoch, where="epoch") < 0:
        raise BatchContractError("epoch must be non-negative")
    if _finite_int(policy_version, where="policy_version") < 0:
        raise BatchContractError("policy_version must be non-negative")
    if len(prompt_items) != len(rollout_results):
        raise BatchContractError(
            f"prompt_items ({len(prompt_items)}) and rollout_results ({len(rollout_results)}) must have equal length"
        )

    expected_completions = sum(len(result.sequences) for result in rollout_results)
    if expected_completions == 0:
        raise BatchContractError("batch has no completions")
    if len(rewards) != expected_completions:
        raise BatchContractError(f"rewards count ({len(rewards)}) must equal completion count ({expected_completions})")

    all_rewards: list[float] = []
    group_sizes: list[int] = []
    group_offsets: list[int] = [0]
    pending: list[tuple[Any, Any, list[int], list[float], float]] = []
    prompt_ids: list[str] = []
    response_token_count = 0
    reward_cursor = 0

    for prompt_index, (item, result) in enumerate(zip(prompt_items, rollout_results, strict=True)):
        sequences = list(result.sequences)
        if not sequences:
            raise BatchContractError(f"prompt {prompt_index} has no completions")
        prefix_len = len(item.input_tokens)
        if prefix_len <= 0:
            raise BatchContractError(f"prompt {prompt_index} has empty input_tokens")

        group_rewards: list[float] = []
        for sample_index, sequence in enumerate(sequences):
            where = f"prompt {prompt_index} sample {sample_index}"
            resp_tokens = list(sequence.resp_tokens)
            resp_logprobs = list(sequence.resp_logprobs)
            if len(resp_tokens) != len(resp_logprobs):
                raise BatchContractError(
                    f"{where}: resp_tokens ({len(resp_tokens)}) and resp_logprobs "
                    f"({len(resp_logprobs)}) must have equal length"
                )
            for pos, logprob in enumerate(resp_logprobs):
                _finite_float(logprob, where=f"{where} resp_logprobs[{pos}]")
            reward = _finite_float(rewards[reward_cursor], where=f"{where} reward")
            reward_cursor += 1

            tokens = list(item.input_tokens) + resp_tokens
            logprobs = [0.0] * prefix_len + resp_logprobs
            response_token_count += len(resp_tokens)
            group_rewards.append(reward)
            all_rewards.append(reward)
            pending.append((item, sequence, tokens, logprobs, reward))

        group_sizes.append(len(group_rewards))
        group_offsets.append(group_offsets[-1] + len(group_rewards))
        prompt_ids.append(str(getattr(item, "prompt_id", getattr(item, "id", prompt_index))))

    advantages = compute_batch_group_advantages(all_rewards, group_sizes)

    sequences_out: list[TrainSequence] = []
    for (item, sequence, tokens, logprobs, reward), advantage in zip(pending, advantages, strict=True):
        prefix_len = len(item.input_tokens)
        train_sequence = TrainSequence(
            tokens=tokens,
            logprobs=logprobs,
            prompt_len=prefix_len,
            scalar_advantage=float(advantage),
            features=item.record.get("features") if isinstance(item.record, dict) else None,
            reward=reward,
            eos_token_id=int(eos_token_id),
            routed_experts=sequence.routed_experts,
        )
        _check_cpu_payload(train_sequence, where=f"sequence {len(sequences_out)}")
        sequences_out.append(train_sequence)

    group_stats: list[GroupStat] = []
    for group_index, size in enumerate(group_sizes):
        start = group_offsets[group_index]
        group_reward_values = tuple(all_rewards[start : start + size])
        mean = sum(group_reward_values) / size
        variance = sum((value - mean) ** 2 for value in group_reward_values) / size
        group_stats.append(
            GroupStat(
                group_index=group_index,
                offset=start,
                size=size,
                rewards=group_reward_values,
                mean_reward=mean,
                std_reward=math.sqrt(max(variance, 0.0)),
            )
        )

    return BatchEnvelope(
        run_id=run_id,
        batch_id=batch_id,
        epoch=int(epoch),
        policy_version=int(policy_version),
        prompt_ids=tuple(prompt_ids),
        sequences=tuple(sequences_out),
        group_offsets=tuple(group_offsets),
        rewards=tuple(all_rewards),
        group_stats=tuple(group_stats),
        prompt_count=len(prompt_ids),
        completion_count=expected_completions,
        response_token_count=response_token_count,
        produced_at=time.monotonic() if produced_at is None else float(produced_at),
    )
