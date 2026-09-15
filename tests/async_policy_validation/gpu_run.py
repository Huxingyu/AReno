"""Real native dual-GPU smoke, weight equality, reload, and failure probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def adapter_tensors(path: Path):
    from safetensors.torch import load_file

    return load_file(str(path / "adapter_model.safetensors"), device="cpu")


def tensor_fingerprint(tensors) -> str:
    import torch

    digest = hashlib.sha256()
    for key, tensor in sorted(tensors.items()):
        digest.update(key.encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CheckedSync:
    def __init__(self, pair, output):
        self.pair = pair
        self.output = output
        self.checks = []

    def transfer(self, plan, *, timeout_s):
        import torch

        from areno.experimental.async_policy.coordination import Deadline

        deadline = Deadline(timeout_s)
        self.pair.transfer(plan, timeout_s=deadline.remaining())
        with tempfile.TemporaryDirectory(prefix="weights-", dir=self.output) as directory:
            path = Path(directory)
            for role in ("train", "rollout"):
                self.pair.save_checkpoint(str(path / role), role=role, timeout_s=deadline.remaining())
            train = adapter_tensors(path / "train")
            rollout = adapter_tensors(path / "rollout")
            assert train.keys() == rollout.keys() and train
            assert all(torch.isfinite(value).all() and torch.equal(value, rollout[key]) for key, value in train.items())
            self.checks.append({"version": plan.target_version, "tensors": len(train),
                                "sha256": tensor_fingerprint(train), "max_abs_error": 0.0})
            dump(self.output / "weight-checks.json", self.checks)


class SavingTrain:
    def __init__(self, pair, output):
        self.pair = pair
        self.output = output

    def initialize(self, *, timeout_s):
        self.pair.initialize(timeout_s=timeout_s)

    def train(self, batch, version, *, timeout_s):
        from areno.experimental.async_policy.coordination import Deadline

        deadline = Deadline(timeout_s)
        stepped = self.pair.train(batch, version, timeout_s=deadline.remaining())
        # Saving occurs on the training consumer while it holds its lease.
        if stepped and version in {0, 4}:
            self.pair.save_checkpoint(str(self.output / f"checkpoint-{version + 1}"), timeout_s=deadline.remaining())
        return stepped

    def close(self, *, timeout_s):
        self.pair.close(timeout_s=timeout_s)


def make_inputs(model_path):
    from areno.api.tokenizer import configure_chat_template_enable_thinking, encode_generation_prompt, load_tokenizer
    from areno.experimental.async_policy.contracts import AsyncPrompt

    tokenizer = load_tokenizer(model_path)
    configure_chat_template_enable_thinking(tokenizer, False)
    rows = [json.loads(line) for line in (Path(__file__).parent / "smoke_prompts.jsonl").read_text().splitlines()]
    prompts = []
    for index, row in enumerate(rows):
        tokens = encode_generation_prompt(tokenizer, row["prompt"])
        assert len(tokens) <= 128
        prompts.append(AsyncPrompt(str(index), tuple(tokens), row["prompt"], row, row["answer"]))
    return tokenizer, prompts


def gpu_overlap(output: Path) -> dict:
    """Conservatively align independent Kineto GPU traces using CPU markers."""
    kernels = {"train": [], "rollout": []}
    for path in output.glob("worker-*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "trace" not in row:
                continue
            events = json.loads((output / row["trace"]).read_text())["traceEvents"]
            marker = next(event for event in events if event.get("name") == row["clock_marker"] and event.get("ph") == "X")
            low = row["clock_before_ns"] / 1000 - marker["ts"]
            high = row["clock_inside_ns"] / 1000 - marker["ts"]
            for event in events:
                if event.get("cat") == "kernel" and event.get("ph") == "X":
                    kernels[row["kind"]].append({
                        "start_upper_us": event["ts"] + high,
                        "end_lower_us": event["ts"] + event["dur"] + low,
                        "name": event["name"], "device": event.get("args", {}).get("device"),
                    })
    train = sorted(kernels["train"], key=lambda event: event["start_upper_us"])
    rollout = sorted(kernels["rollout"], key=lambda event: event["start_upper_us"])
    best = {"overlap_us": 0.0}
    left = right = 0
    while left < len(train) and right < len(rollout):
        a, b = train[left], rollout[right]
        overlap = min(a["end_lower_us"], b["end_lower_us"]) - max(a["start_upper_us"], b["start_upper_us"])
        if overlap > best["overlap_us"] and a["device"] != b["device"]:
            best = {"overlap_us": overlap, "train_kernel": a, "rollout_kernel": b}
        if a["end_lower_us"] <= b["end_lower_us"]:
            left += 1
        else:
            right += 1
    return {"train_kernels": len(train), "rollout_kernels": len(rollout), "best": best}


def reload_checkpoint(model_path: str, output: Path, prompt, sampling) -> dict:
    import torch

    from areno import Trainer
    from areno.adapters.config import LoraConfig
    from areno.api.config import CudaConfig

    checkpoint = output / "checkpoint-5"
    trainer = Trainer(world_size=1, model_path=model_path, custom_config=CudaConfig(
        devices=[0], tp_size=1, dp_size=1, max_running_prompts=4,
        lora=LoraConfig(adapter_path=str(checkpoint)),
        runtime={"compile_model": False, "attn_backend": "flash", "eager_decode": True},
    ))
    try:
        trainer.init()
        exported = output / "reloaded-checkpoint"
        trainer.export_adapter(str(exported))
        before, after = adapter_tensors(checkpoint), adapter_tensors(exported)
        assert before.keys() == after.keys()
        assert all(torch.equal(value, after[key]) for key, value in before.items())
        trainer.begin_rollout_session()
        try:
            result = trainer.rollout_token_batch([list(prompt.input_tokens)], n_samples=4, sampling_params=sampling)[0]
        finally:
            trainer.end_rollout_session()
        assert len(result.sequences) == 4 and all(sequence.resp_tokens for sequence in result.sequences)
        return {"equal_tensors": len(before), "max_abs_error": 0.0,
                "generated_tokens": sum(len(row.resp_tokens) for row in result.sequences)}
    finally:
        trainer.close()


def run_case(model_path: str, output: Path, *, mode: str, fault: str | None = None) -> dict:
    import torch
    from gpu_worker import ObservedWorker
    from smoke_reward import reward_fn

    from areno.adapters.config import LoraConfig
    from areno.api.config import CudaConfig
    from areno.api.models import SamplingParams
    from areno.experimental.async_policy.bridge import DualEngineBridge
    from areno.experimental.async_policy.contracts import AsyncPolicyConfig, DeviceMode
    from areno.experimental.async_policy.data import build_batch_envelope, score_prompt_group
    from areno.experimental.async_policy.native import NativeCudaEnginePair
    from areno.experimental.async_policy.pipeline import AsyncPolicyPipeline

    output.mkdir(parents=True, exist_ok=True)
    os.environ["ARENO_R4_OUTPUT"] = str(output)
    os.environ["ARENO_R4_PROFILE"] = "1" if mode == "trace" else "0"
    os.environ["ARENO_R4_FAULT"] = fault or ""
    torch.manual_seed(41)
    tokenizer, prompts = make_inputs(model_path)
    sampling = SamplingParams(max_prompt_len=128, max_new_tokens=64, temperature=1.0, top_p=1.0, seed=41)
    config = CudaConfig(devices=[0], rollout_devices=[1], tp_size=1, dp_size=1, rollout_tp_size=1,
                        max_running_prompts=4, policy_sync_bucket_mb=8, lora=LoraConfig(rank=8, alpha=16),
                        base_model_name_or_path="Qwen/Qwen3-0.6B",
                        optimizer={"lr": 1e-5, "grad_clip_norm": 1.0},
                        runtime={"compile_model": False, "eager_decode": True, "attn_backend": "flash"})
    pair = NativeCudaEnginePair(model_path=model_path, config=config, sampling_params=sampling,
                               tokenizer=tokenizer, n_samples=4, mini_bs=1, worker_cls=ObservedWorker)
    checked_sync = CheckedSync(pair, output)
    train = SavingTrain(pair, output)
    policy_config = AsyncPolicyConfig(max_steps=5, operation_timeout_s=300, shutdown_timeout_s=30)
    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=True)

    def observed_reward(record):
        value = reward_fn(record)
        sample = {"prompt": record.prompt, "completion": record.completion, "answer": record.answer,
                  "reward": value, "metadata": record.metadata}
        with (output / "samples.jsonl").open("a") as stream:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return value

    metadata = {"mode": mode, "fault": fault, "source_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(), "config": asdict(policy_config),
        "runtime": config.runtime, "model_path": model_path, "sampling": sampling.model_dump()}
    dump(output / "config.json", metadata)
    started = time.monotonic()
    error = None
    pipeline = None
    summary = dict(metadata)
    try:
        if mode == "baseline":
            bridge = DualEngineBridge(train_engine=train, rollout_engine=pair.rollout_engine,
                                      weight_sync=checked_sync, config=policy_config, mode=DeviceMode.SYNC,
                                      train_devices=("cuda:0",), rollout_devices=("cuda:1",))
            metrics = []
            try:
                bridge.initialize()
                for index, prompt in enumerate(prompts[:5]):
                    session = bridge.rollout_session(prompt)
                    rewards = score_prompt_group(prompt, session.result, observed_reward, decode=decode, check_running=lambda: None)
                    batch = build_batch_envelope(run_id="sync", batch_id=str(index), epoch=0,
                                                 policy_version=session.policy_version, prompt_items=[prompt],
                                                 rollout_results=[session.result], rewards=rewards,
                                                 eos_token_id=tokenizer.eos_token_id)
                    metrics.append(asdict(bridge.train(batch)))
                    bridge.sync()
                summary["steps"] = sum(row["stepped"] for row in metrics)
                summary["batches"] = metrics
            finally:
                bridge.close()
        else:
            pipeline = AsyncPolicyPipeline(config=policy_config, data_source=prompts,
                                           rollout_engine=pair.rollout_engine, train_engine=train,
                                           weight_sync=checked_sync, reward_fn=observed_reward, decode=decode,
                                           eos_token_id=tokenizer.eos_token_id)
            report = pipeline.run()
            summary["pipeline"] = asdict(report)
            summary["steps"] = report.steps
        assert summary["steps"] == 5
        assert all(math.isfinite(row["loss"]) for row in pair.train_stats)
        assert sum(row["stepped"] for row in pair.train_stats) == 5
        assert any(row.get("metrics", {}).get("grad_norm", 0) > 0 for row in pair.train_stats)
        final_hash = tensor_fingerprint(adapter_tensors(output / "checkpoint-5"))
        assert final_hash != checked_sync.checks[0]["sha256"], "LoRA weights did not change"
        summary["reload"] = reload_checkpoint(model_path, output, prompts[0], sampling)
        if mode == "trace":
            summary["gpu_overlap"] = gpu_overlap(output)
            assert summary["gpu_overlap"]["best"]["overlap_us"] > 0, "no proven cross-GPU kernel overlap"
    except BaseException as exc:
        error = exc
        summary["error"] = f"{type(exc).__name__}: {exc}"
        (output / "traceback.txt").write_text(traceback.format_exc())
        if pipeline is not None:
            summary["pipeline"] = asdict(pipeline.report())
    finally:
        summary["wall_s"] = time.monotonic() - started
        summary["native_train_stats"] = pair.train_stats
        summary["native_sync_stats"] = pair.sync_stats
        summary["weight_checks"] = checked_sync.checks
        summary["worker_exits"] = pair.worker_exits
        summary["live_children"] = [{"pid": process.pid, "name": process.name} for process in multiprocessing.active_children()]
        summary["ok"] = error is None and not summary["live_children"]
        if fault:
            if fault == "receive":
                observed = error is not None and "injected native policy receiver failure" in str(error)
            else:
                observed = error is not None and any(row["exitcode"] == 23 for row in pair.worker_exits)
            summary["expected_fault_observed"] = observed
            summary["ok"] = summary["expected_fault_observed"] and not summary["live_children"]
            preserved = output / "checkpoint-1"
            summary["successful_checkpoint_preserved"] = bool(adapter_tensors(preserved)) if preserved.is_dir() else False
            summary["ok"] = summary["ok"] and summary["successful_checkpoint_preserved"]
            if fault == "receive":
                summary["ok"] = summary["ok"] and summary["pipeline"]["rollout_policy_version"] == 0
        dump(output / "result.json", summary)
    return summary


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "async", "trace", "faults"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                   "cuda_available": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("transformers", "modelscope", "safetensors", "flash-attn")}}
    environment["devices"] = [{"name": torch.cuda.get_device_name(index),
                               "total_memory": torch.cuda.get_device_properties(index).total_memory,
                               "capability": torch.cuda.get_device_capability(index)}
                              for index in range(torch.cuda.device_count())]
    environment["nvidia_smi"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader",
    ], text=True).strip()
    dump(args.output_dir / "environment.json", environment)
    assert environment["gpu_count"] == 2 and all("L4" in item["name"] for item in environment["devices"])
    subprocess.run([sys.executable, ".agents/skills/areno-run-training/scripts/inspect_dataset.py", "--dataset-path",
                    "tests/async_policy_validation/smoke_prompts.jsonl", "--model-hub", "modelscope", "--algo", "grpo"],
                   cwd=ROOT, check=True)
    if args.mode == "faults":
        results = [run_case(args.model_path, args.output_dir / fault, mode="async", fault=fault)
                   for fault in ("receive", "worker_exit")]
        results.append(run_case(args.model_path, args.output_dir / "restart", mode="async"))
        result = {"ok": all(row["ok"] for row in results), "cases": results}
        dump(args.output_dir / "result.json", result)
    else:
        result = run_case(args.model_path, args.output_dir, mode=args.mode)
    print(json.dumps({"ok": result["ok"], "mode": args.mode, "output": str(args.output_dir)}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
