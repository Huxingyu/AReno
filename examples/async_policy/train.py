"""Two-GPU GRPO example with bounded async scheduling and an optional sync comparison.

python examples/async_policy/train.py --model Qwen/Qwen3-0.6B --steps 55 --output-dir outputs/async-example
Remote model references are resolved exclusively through ModelScope.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def answer_matches(completion: str, answer: str) -> bool:
    """Score the final numeric answer, never an earlier number in the reasoning."""
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", completion.replace(",", ""))
    return bool(numbers) and Decimal(numbers[-1]) == Decimal(str(answer))


def reward_fn(record) -> float:
    return float(answer_matches(record.completion, record.answer))


def load_prompts(path: Path, tokenizer):
    from areno.api.tokenizer import encode_generation_prompt
    from areno.experimental.async_policy.contracts import AsyncPrompt

    prompts = []
    for index, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        tokens = tuple(encode_generation_prompt(tokenizer, row["prompt"]))
        if not tokens or len(tokens) > 128:
            raise ValueError("example prompts must contain 1–128 tokens")
        prompts.append(AsyncPrompt(str(index), tokens, row["prompt"], row, row["answer"]))
    if not prompts:
        raise ValueError("dataset is empty")
    return prompts


class CheckpointedTrain:
    """Save only while the training consumer owns its model lease."""

    def __init__(self, pair, output: Path, steps: int, *, save_training_state: bool = False):
        self.pair, self.output, self.steps = pair, output, steps
        self.initial_version = pair.initial_policy_version
        self.save_training_state = save_training_state
        self.events = []

    def initialize(self, *, timeout_s):
        self.pair.initialize(timeout_s=timeout_s)

    def train(self, batch, version, *, timeout_s):
        from areno.experimental.async_policy.coordination import Deadline

        deadline = Deadline(timeout_s)
        start = time.time_ns()
        stepped = self.pair.train(batch, version, timeout_s=deadline.remaining())
        event = {"version_before": version, "stepped": stepped, "start_ns": start, "end_ns": time.time_ns(),
                 "batch_version": batch.policy_version, "prompt_ids": batch.prompt_ids,
                 "response_tokens": sum(len(row.tokens) - row.prompt_len for row in batch.sequences),
                 "rewards": [row.reward for row in batch.sequences]}
        event["response_lengths"] = [len(row.tokens) - row.prompt_len for row in batch.sequences]
        event["token_limit_hits"] = sum(length >= self.pair.sampling_params.max_new_tokens
                                        for length in event["response_lengths"])
        self.events.append(event)
        with (self.output / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if stepped and version + 1 in {self.initial_version + 1, self.initial_version + self.steps}:
            save = self.pair.save_training_checkpoint if self.save_training_state else self.pair.save_checkpoint
            save(str(self.output / f"checkpoint-{version + 1}"), timeout_s=deadline.remaining())
        return stepped

    def close(self, *, timeout_s):
        self.pair.close(timeout_s=timeout_s)


def run_training(*, model_path: str, data_path: Path, output: Path, mode: str = "async",
                 steps: int = 55, seed: int = 41, lag: int = 1, worker_cls=None,
                 resume_from: str | None = None, save_training_state: bool = False,
                 base_model_reference: str | None = None, attn_backend: str = "native",
                 lora_rank: int = 8, lora_alpha: float = 16, learning_rate: float = 1e-5,
                 n_samples: int = 4, max_new_tokens: int = 64, loss: str = "grpo",
                 queue_capacity: int = 2, max_inflight_rollouts: int = 1,
                 weight_sync_interval_updates: int = 1) -> dict:
    import torch

    from areno.adapters.config import LoraConfig
    from areno.api.config import CudaConfig
    from areno.api.models import SamplingParams
    from areno.api.tokenizer import configure_chat_template_enable_thinking, load_tokenizer
    from areno.experimental.async_policy.bridge import DualEngineBridge
    from areno.experimental.async_policy.contracts import AsyncPolicyConfig, DeviceMode
    from areno.experimental.async_policy.coordination import PolicyPipelineCoordinator
    from areno.experimental.async_policy.data import build_batch_envelope, score_prompt_group
    from areno.experimental.async_policy.native import NativeCudaEnginePair
    from areno.experimental.async_policy.pipeline import AsyncPolicyPipeline

    if steps < 1 or mode not in {"sync", "async"}:
        raise ValueError("positive steps and mode sync/async are required")
    if attn_backend not in {"native", "flash"}:
        raise ValueError("attn_backend must be native or flash")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    loss_fn = resolve_loss(loss)
    policy = AsyncPolicyConfig(max_steps=steps, max_policy_lag=lag, queue_capacity=queue_capacity,
                               max_inflight_rollouts=max_inflight_rollouts,
                               weight_sync_interval_updates=weight_sync_interval_updates,
                               operation_timeout_s=300, shutdown_timeout_s=30)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("result.json", "updates.jsonl", "samples.jsonl")):
        raise FileExistsError("use a fresh output directory for each training run")
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(model_path)
    configure_chat_template_enable_thinking(tokenizer, False)
    prompts = load_prompts(data_path, tokenizer)
    config = CudaConfig(devices=[0], rollout_devices=[1], tp_size=1, dp_size=1, rollout_tp_size=1,
                        max_running_prompts=n_samples, lora=LoraConfig(rank=lora_rank, alpha=lora_alpha),
                        policy_sync_bucket_mb=8, base_model_name_or_path=base_model_reference or model_path,
                        optimizer={"lr": learning_rate, "grad_clip_norm": 1.0},
                        runtime={"compile_model": False, "eager_decode": True, "attn_backend": attn_backend})
    sampling = SamplingParams(max_prompt_len=128, max_new_tokens=max_new_tokens, seed=seed)
    pair = NativeCudaEnginePair(model_path=model_path, config=config, sampling_params=sampling,
                               tokenizer=tokenizer, n_samples=n_samples, mini_bs=1, seed=seed, worker_cls=worker_cls,
                               resume_from=resume_from, loss_fn=loss_fn)
    training = CheckpointedTrain(pair, output, steps, save_training_state=save_training_state or resume_from is not None)
    coordinator = PolicyPipelineCoordinator(train_policy_version=pair.initial_policy_version)

    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=True)

    def recorded_reward(record):
        reward = reward_fn(record)
        with (output / "samples.jsonl").open("a") as stream:
            stream.write(json.dumps({"prompt": record.prompt, "completion": record.completion,
                                     "answer": record.answer, "reward": reward,
                                     "response_tokens": sum(record.loss_mask),
                                     "hit_token_limit": sum(record.loss_mask) >= max_new_tokens,
                                     **record.metadata}) + "\n")
        return reward

    pipeline = None
    bridge = None
    if mode == "async":
        pipeline = AsyncPolicyPipeline(config=policy, data_source=itertools.cycle(prompts),
                                       train_engine=training, rollout_engine=pair.rollout_engine,
                                       weight_sync=pair.weight_sync, reward_fn=recorded_reward,
                                       decode=decode, eos_token_id=tokenizer.eos_token_id, coordinator=coordinator)
        stop = pipeline.request_stop
    else:
        bridge = DualEngineBridge(train_engine=training, rollout_engine=pair.rollout_engine,
                                  weight_sync=pair.weight_sync, config=policy, mode=DeviceMode.SYNC, coordinator=coordinator)
        stop = bridge.supervisor.stop
    interrupted = []

    def on_signal(signum, frame):
        interrupted.append(signum)
        stop()

    previous = {signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    summary = {"mode": mode, "seed": seed, "policy": asdict(policy), "runtime": config.runtime,
               "model_path": model_path, "sampling": sampling.model_dump(),
               "lora_rank": pair.config.lora.rank, "lora_alpha": pair.config.lora.alpha,
               "n_samples": n_samples, "mini_bs": 1, "learning_rate": learning_rate, "loss": loss,
               "base_model_reference": base_model_reference or model_path,
               "resume_from": resume_from, "initial_policy_version": pair.initial_policy_version,
               "data_path": str(data_path), "source_sha": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    try:
        if pipeline is not None:
            summary["pipeline"] = asdict(pipeline.run())
        else:
            syncs, batches = [], []
            try:
                syncs.append(asdict(bridge.initialize()))
                for index, prompt in enumerate(itertools.islice(itertools.cycle(prompts), steps)):
                    session = bridge.rollout_session(prompt)
                    rewards = score_prompt_group(prompt, session.result, recorded_reward,
                                                  decode=decode, check_running=bridge.supervisor.check_running)
                    batch = build_batch_envelope(run_id="sync", batch_id=str(index), epoch=0,
                                                 policy_version=session.policy_version, prompt_items=[prompt],
                                                 rollout_results=[session.result], rewards=rewards,
                                                 eos_token_id=tokenizer.eos_token_id)
                    batches.append(asdict(bridge.train(batch)))
                    syncs.append(asdict(bridge.sync()))
                summary.update(sync_metrics=syncs, batch_metrics=batches)
            finally:
                bridge.close()
        summary["ok"] = (not interrupted and sum(row["stepped"] for row in training.events) == steps
                         and len(pair.worker_exits) == 2 and not any(row["alive"] for row in pair.worker_exits))
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        summary.update(updates=training.events, native_train_stats=pair.train_stats,
                       native_sync_stats=pair.sync_stats, worker_exits=pair.worker_exits, signals=interrupted)
        if pipeline is not None:
            summary["pipeline"] = asdict(pipeline.report())
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def resolve_loss(name: str):
    if name == "grpo-offpolicy":
        from areno.experimental.async_policy.losses import grpo_offpolicy_loss_fn

        return grpo_offpolicy_loss_fn
    if name not in {"grpo", "gspo"}:
        raise ValueError(f"unsupported example loss: {name}")
    from areno.api.algorithms import get_algorithm

    return get_algorithm(name).default_loss_fn


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-path", type=Path, default=Path(__file__).with_name("train.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("sync", "async"), default="async")
    parser.add_argument("--steps", type=positive_int, default=55)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--attn-backend", choices=("flash", "native"), default="native")
    parser.add_argument("--lora-rank", type=positive_int, default=8)
    parser.add_argument("--lora-alpha", type=positive_float, default=16)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-5)
    parser.add_argument("--n-samples", type=positive_int, default=4)
    parser.add_argument("--max-new-tokens", type=positive_int, default=64)
    parser.add_argument("--loss", choices=("grpo", "grpo-offpolicy", "gspo"), default="grpo",
                        help="Upstream GRPO by default; other objectives are experimental ablations")
    parser.add_argument("--queue-capacity", type=positive_int, default=2)
    parser.add_argument("--max-inflight-rollouts", type=positive_int, default=1)
    parser.add_argument("--weight-sync-interval-updates", type=positive_int, default=1)
    parser.add_argument("--save-training-state", action="store_true", help="Include optimizer and train RNG state")
    parser.add_argument("--resume-from", help="Resume a training-state checkpoint; the prompt iterator starts afresh")
    args = parser.parse_args(argv)
    if args.lag < 0:
        parser.error("--lag must be nonnegative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    model_path = args.model
    if not Path(model_path).is_dir():
        from modelscope import snapshot_download

        model_path = snapshot_download(model_path)
    result = run_training(model_path=model_path, data_path=args.data_path, output=args.output_dir,
                          mode=args.mode, steps=args.steps, seed=args.seed, lag=args.lag,
                          resume_from=args.resume_from, save_training_state=args.save_training_state,
                          base_model_reference=args.model, attn_backend=args.attn_backend,
                          lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, learning_rate=args.learning_rate,
                          n_samples=args.n_samples, max_new_tokens=args.max_new_tokens, loss=args.loss,
                          queue_capacity=args.queue_capacity, max_inflight_rollouts=args.max_inflight_rollouts,
                          weight_sync_interval_updates=args.weight_sync_interval_updates)
    print(json.dumps({"ok": result["ok"], "output": str(args.output_dir)}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
