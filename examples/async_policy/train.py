"""Two-GPU GRPO example with bounded async scheduling and an optional sync comparison.

python examples/async_policy/train.py --model Qwen/Qwen3-0.6B --steps 55 --output-dir runs/async-example
Remote model references are resolved exclusively through ModelScope.
"""

from __future__ import annotations

import argparse
import itertools
import json
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
                 resume_from: str | None = None, save_training_state: bool = False) -> dict:
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
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(model_path)
    configure_chat_template_enable_thinking(tokenizer, False)
    prompts = load_prompts(data_path, tokenizer)
    config = CudaConfig(devices=[0], rollout_devices=[1], tp_size=1, dp_size=1, rollout_tp_size=1,
                        max_running_prompts=4, lora=LoraConfig(rank=8, alpha=16), policy_sync_bucket_mb=8,
                        base_model_name_or_path="Qwen/Qwen3-0.6B", optimizer={"lr": 1e-5, "grad_clip_norm": 1.0},
                        runtime={"compile_model": False, "eager_decode": True, "attn_backend": "flash"})
    sampling = SamplingParams(max_prompt_len=128, max_new_tokens=64, seed=seed)
    pair = NativeCudaEnginePair(model_path=model_path, config=config, sampling_params=sampling,
                               tokenizer=tokenizer, n_samples=4, mini_bs=1, seed=seed, worker_cls=worker_cls,
                               resume_from=resume_from)
    training = CheckpointedTrain(pair, output, steps, save_training_state=save_training_state or resume_from is not None)
    coordinator = PolicyPipelineCoordinator(train_policy_version=pair.initial_policy_version)
    policy = AsyncPolicyConfig(max_steps=steps, max_policy_lag=lag, operation_timeout_s=300, shutdown_timeout_s=30)

    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=True)

    def recorded_reward(record):
        reward = reward_fn(record)
        with (output / "samples.jsonl").open("a") as stream:
            stream.write(json.dumps({"prompt": record.prompt, "completion": record.completion,
                                     "answer": record.answer, "reward": reward}) + "\n")
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
               "lora_rank": 8, "lora_alpha": 16, "n_samples": 4, "mini_bs": 1, "learning_rate": 1e-5,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-path", type=Path, default=Path(__file__).with_name("train.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("sync", "async"), default="async")
    parser.add_argument("--steps", type=int, default=55)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--save-training-state", action="store_true", help="Include optimizer and train RNG state")
    parser.add_argument("--resume-from", help="Resume a training-state checkpoint; the prompt iterator starts afresh")
    args = parser.parse_args()
    model_path = args.model
    if not Path(model_path).is_dir():
        from modelscope import snapshot_download

        model_path = snapshot_download(model_path)
    result = run_training(model_path=model_path, data_path=args.data_path, output=args.output_dir,
                          mode=args.mode, steps=args.steps, seed=args.seed, lag=args.lag,
                          resume_from=args.resume_from, save_training_state=args.save_training_state)
    print(json.dumps({"ok": result["ok"], "output": str(args.output_dir)}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
