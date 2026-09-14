"""Whole-group materialization with independent CPU payload ownership."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .contracts import AsyncPrompt, BatchContractError, BatchEnvelope

if TYPE_CHECKING:
    from areno.api.models import RolloutResult
    from areno.api.rewards import RewardRecord


def snapshot_payload(value: Any) -> Any:
    """Copy supported CPU values; reject opaque objects and autograd graphs."""
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise BatchContractError("payload dictionary keys must be strings")
        return {key: snapshot_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [snapshot_payload(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items

    # Imports stay out of package discovery and ordinary protocol-only imports.
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None:
            raise BatchContractError("tensor payloads must be CPU tensors without an autograd graph")
        return value.detach().clone()
    if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
        return value.copy()
    if isinstance(value, np.generic) and value.dtype.kind in "biuf":
        return value.item()
    raise BatchContractError(f"unsupported payload type: {type(value).__name__}")


def snapshot_prompt(prompt: AsyncPrompt) -> AsyncPrompt:
    tokens = tuple(prompt.input_tokens)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise BatchContractError("a prompt must contain nonnegative integer tokens")
    return AsyncPrompt(
        prompt_id=str(prompt.prompt_id),
        input_tokens=tokens,
        prompt=prompt.prompt,
        record=snapshot_payload(prompt.record),
        solutions=snapshot_payload(prompt.solutions),
    )


def snapshot_rollout(result: RolloutResult, version: int) -> RolloutResult:
    from areno.api.models import RolloutResult, RolloutSequence

    if not isinstance(result, RolloutResult):
        raise BatchContractError("generate must return a RolloutResult")
    if result.adapter_version is not None and result.adapter_version != version:
        raise BatchContractError("rollout adapter_version disagrees with the admitted policy version")
    sequences = []
    for row in result.sequences:
        if len(row.resp_tokens) != len(row.resp_logprobs):
            raise BatchContractError("each response token must have one rollout logprob")
        if any(token < 0 for token in row.resp_tokens) or not all(math.isfinite(p) for p in row.resp_logprobs):
            raise BatchContractError("response tokens must be nonnegative and logprobs finite")
        sequences.append(RolloutSequence(
            resp_tokens=list(row.resp_tokens),
            resp_logprobs=list(row.resp_logprobs),
            routed_experts=snapshot_payload(row.routed_experts),
        ))
    return RolloutResult(sequences=sequences, adapter_version=result.adapter_version)


def score_prompt_group(
    prompt: AsyncPrompt,
    result: RolloutResult,
    reward_fn: Callable[[RewardRecord], float],
    *,
    decode: Callable[[list[int]], str] | None,
    check_running: Callable[[], None],
) -> list[float]:
    from areno.api.rewards import make_reward_record

    rewards = []
    prefix = list(prompt.input_tokens)
    for index, sequence in enumerate(result.sequences):
        check_running()
        response = list(sequence.resp_tokens)
        record = make_reward_record(
            prompt=prompt.prompt,
            completion=decode(list(response)) if decode is not None else "",
            source_record=snapshot_payload(prompt.record),
            answer=snapshot_payload(prompt.solutions),
            tokens=prefix + response,
            logprobs=[0.0] * len(prefix) + list(sequence.resp_logprobs),
            loss_mask=[False] * len(prefix) + [True] * len(response),
            metadata={"prompt_id": prompt.prompt_id, "prompt_index": 0, "sample_index": index},
        )
        rewards.append(float(reward_fn(record)))
    check_running()
    return rewards


def build_batch_envelope(
    *,
    run_id: str,
    batch_id: str,
    epoch: int,
    policy_version: int,
    prompt_items: Sequence[AsyncPrompt],
    rollout_results: Sequence[RolloutResult],
    rewards: Sequence[float],
    eos_token_id: int,
) -> BatchEnvelope:
    from areno.api.advantages import compute_batch_group_advantages
    from areno.api.models import TrainSequence

    if type(policy_version) is not int or policy_version < 0:
        raise BatchContractError("policy_version must be a nonnegative integer")
    if len(prompt_items) != len(rollout_results):
        raise BatchContractError("each prompt must have one complete rollout group")
    prompts = [snapshot_prompt(prompt) for prompt in prompt_items]
    results = [snapshot_rollout(result, policy_version) for result in rollout_results]
    sizes = [len(result.sequences) for result in results]
    scores = [float(reward) for reward in rewards]
    if sum(sizes) != len(scores) or not all(math.isfinite(reward) for reward in scores):
        raise BatchContractError("each completion must have one finite reward")
    advantages = compute_batch_group_advantages(scores, [size for size in sizes if size])
    rows = []
    offsets = [0]
    for prompt, result in zip(prompts, results, strict=True):
        prefix = list(prompt.input_tokens)
        for sequence in result.sequences:
            index = len(rows)
            rows.append(TrainSequence(
                tokens=prefix + list(sequence.resp_tokens),
                logprobs=[0.0] * len(prefix) + list(sequence.resp_logprobs),
                prompt_len=len(prefix),
                scalar_advantage=float(advantages[index]),
                features=snapshot_payload(prompt.record.get("features")),
                reward=scores[index],
                eos_token_id=eos_token_id,
                routed_experts=snapshot_payload(sequence.routed_experts),
            ))
        offsets.append(len(rows))
    return BatchEnvelope(
        run_id=run_id, batch_id=batch_id, epoch=epoch, policy_version=policy_version,
        prompt_ids=tuple(prompt.prompt_id for prompt in prompts), group_offsets=tuple(offsets), sequences=tuple(rows),
    )
