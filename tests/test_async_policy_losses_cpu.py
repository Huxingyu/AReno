"""Exact clipped-objective and gradient checks, independent of scheduling."""

import math
import pickle

import pytest
import torch

from areno.api.algorithms import grpo_loss_fn
from areno.experimental.async_policy.losses import grpo_offpolicy_loss_fn


def pack(advantages, old, mask=None):
    count = len(advantages)
    return {
        "packed_response_mask": torch.tensor(mask if mask is not None else [True] * count),
        "packed_logprobs": old,
        "packed_advantages": torch.tensor(advantages, dtype=torch.float64, requires_grad=True),
        "packed_seq_ids": torch.zeros(count, dtype=torch.long),
        "packed_num_sequences": 1,
    }


def test_behavior_ratio_clips_both_advantage_signs_and_masks_prompt():
    ratios = [1.5, 0.5, 1.1, 1.5, 0.5, 0.9]
    old = torch.tensor([1000.0] + [-2.0] * 6, requires_grad=True)
    current = torch.tensor([-1000.0] + [-2.0 + math.log(r) for r in ratios],
                           dtype=torch.float64, requires_grad=True)
    data = pack([99, 1, 1, 1, -1, -1, -1], old, [False] + [True] * 6)
    loss, metrics = grpo_offpolicy_loss_fn(data, current)
    assert loss.item() == pytest.approx(0.4 / 6)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor(
        [0, 0, -0.5 / 6, -1.1 / 6, 1.5 / 6, 0, 0.9 / 6], dtype=torch.float64,
    ))
    assert old.grad is None and data["packed_advantages"].grad is None
    assert metrics["ratio_mean"].item() == pytest.approx(sum(ratios) / 6)
    assert metrics["pg_clipfrac"].item() == pytest.approx(2 / 6)


def test_on_policy_loss_and_gradient_match_upstream():
    old = torch.tensor([-1.0, -2.0, -3.0])
    data = pack([1, -0.5, 0.25], old)
    reference = old.clone().requires_grad_()
    candidate = old.clone().requires_grad_()
    left, _ = grpo_loss_fn(data, reference)
    right, _ = grpo_offpolicy_loss_fn(data, candidate)
    left.backward()
    right.backward()
    torch.testing.assert_close(left, right)
    torch.testing.assert_close(reference.grad, candidate.grad)


def test_stale_policy_has_different_gradient_from_upstream_s1():
    old = torch.tensor([-2.0])
    data = pack([1], old)
    reference = torch.tensor([-1.0], requires_grad=True)
    candidate = reference.detach().clone().requires_grad_()
    left, left_metrics = grpo_loss_fn(data, reference)
    right, right_metrics = grpo_offpolicy_loss_fn(data, candidate)
    left.backward()
    right.backward()
    assert left_metrics["ratio_mean"].item() == 1
    assert right_metrics["ratio_mean"].item() == pytest.approx(math.e)
    assert reference.grad.item() == -1
    assert candidate.grad.item() == 0
    assert right.item() == pytest.approx(-1.2)


def test_empty_response_has_finite_zero_loss_and_gradients():
    values = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    data = pack([1, -1], -values.detach(), [False, False])
    loss, metrics = grpo_offpolicy_loss_fn(data, values)
    loss.backward()
    assert loss.item() == 0 and values.grad.tolist() == [0, 0]
    assert all(torch.isfinite(value).all() for value in metrics.values())


@pytest.mark.parametrize("clip", [-0.1, 1.0, float("nan")])
def test_invalid_clip_is_rejected(clip):
    with pytest.raises(ValueError, match="clip_eps"):
        grpo_offpolicy_loss_fn({}, torch.tensor([]), clip_eps=clip)


def test_ablation_callable_can_cross_spawn_transport():
    assert pickle.loads(pickle.dumps(grpo_offpolicy_loss_fn)) is grpo_offpolicy_loss_fn


def test_behavior_ratio_passes_existing_s1_oracle(monkeypatch):
    from tests.async_policy_validation import torch_cases

    monkeypatch.setattr(torch_cases, "grpo_loss_fn", grpo_offpolicy_loss_fn)
    torch_cases.test_stale_loss_matches_behavior_policy_clipped_reference()
