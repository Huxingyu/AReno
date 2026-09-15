"""Optional loss ablations; the pipeline's default remains upstream GRPO."""

from __future__ import annotations

import math


def grpo_offpolicy_loss_fn(data_pack, logprobs, *, clip_eps: float = 0.2):
    """Token-level clipped objective using recorded behavior log probabilities.

    This is an experimental importance-ratio ablation, not a claim that bounded
    lag or clipping makes asynchronous training unbiased. Prompt and padding
    positions do not contribute; rollout logprobs and advantages are constants.
    """

    import torch

    from areno.api.backend.common import LOGP_METRIC_WEIGHT, TrainMetric
    from areno.api.backend.cuda.losses import masked_mean, response_layout

    if not math.isfinite(clip_eps) or not 0 <= clip_eps < 1:
        raise ValueError("clip_eps must be finite and in [0, 1)")
    layout = response_layout(data_pack, logprobs, need_old_logprobs=True,
                             need_advantages=True, need_sequences=True)
    mask = layout.response_mask.bool()
    old = layout.old_logprobs.detach()
    advantages = torch.where(mask, layout.advantages.detach(), 0.0)
    log_ratio = torch.where(mask, logprobs - old, 0.0)
    ratio = torch.exp(log_ratio)
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    objective, clipped_objective = ratio * advantages, clipped * advantages
    loss = -masked_mean(torch.minimum(objective, clipped_objective), layout)
    valid_ratio = ratio[mask]
    difference = torch.where(mask, old - logprobs.detach(), 0.0)
    return loss, {
        "policy_loss": loss.detach(),
        "total_loss": loss.detach(),
        TrainMetric.RATIO_MEAN: masked_mean(ratio, layout).detach(),
        TrainMetric.RATIO_STD: valid_ratio.std().detach() if valid_ratio.numel() > 1
        else logprobs.new_zeros(()),
        "pg_clipfrac": masked_mean((clipped_objective < objective).float(), layout).detach(),
        "advantage_mean": masked_mean(advantages, layout).detach(),
        "response_len": layout.response_len.mean().detach(),
        LOGP_METRIC_WEIGHT: layout.valid_count.detach(),
        TrainMetric.ROLLOUT_LOGPROBS_MEAN: masked_mean(torch.where(mask, old, 0.0), layout).detach(),
        TrainMetric.TRAIN_LOGPROBS_MEAN: masked_mean(torch.where(mask, logprobs.detach(), 0.0), layout).detach(),
        TrainMetric.LOGP_DIFF_MEAN: masked_mean(difference, layout).detach(),
        TrainMetric.LOGP_ABS_DIFF_MEAN: masked_mean(difference.abs(), layout).detach(),
    }
