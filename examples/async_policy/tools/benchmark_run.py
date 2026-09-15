"""Repeated steady-state and held-out-quality acceptance using the shipped example."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


class Monitor:
    def __init__(self, output: Path):
        self.output = output
        self.stop = threading.Event()
        self.rows = []
        self.errors = []
        self.thread = threading.Thread(target=self.run, name="acceptance-monitor")

    def run(self):
        import psutil

        previous = {}
        previous_time = time.monotonic()
        parent = psutil.Process()
        while not self.stop.is_set():
            try:
                now = time.monotonic()
                processes = [parent, *parent.children(recursive=True)]
                cpu, rss, current = 0.0, 0, {}
                for process in processes:
                    try:
                        times = process.cpu_times()
                        current[process.pid] = times.user + times.system
                        if process.pid in previous:
                            cpu += 100 * (current[process.pid] - previous[process.pid]) / max(now - previous_time, 1e-6)
                        rss += process.memory_info().rss
                    except psutil.Error:
                        continue
                previous, previous_time = current, now
                csv_text = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ], text=True, timeout=5)
                gpus = [{"device": int(row[0]), "utilization_percent": float(row[1]),
                         "memory_mib": float(row[2]), "power_watts": float(row[3])}
                        for row in csv.reader(io.StringIO(csv_text))]
                row = {"time_ns": time.time_ns(), "cpu_percent": cpu, "summed_process_rss_bytes": rss, "gpus": gpus}
                self.rows.append(row)
                with (self.output / "monitor.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=7)
        if self.thread.is_alive():
            raise TimeoutError("GPU/process monitor did not stop")


def quality_metrics(rows: list[dict], max_new_tokens: int) -> dict:
    """A token-limit hit is a truncation proxy, not an inferred stop reason."""
    if not rows:
        raise ValueError("evaluation set is empty")
    hits = sum(row["response_tokens"] >= max_new_tokens for row in rows)
    return {"correct": sum(row["correct"] for row in rows), "total": len(rows),
            "accuracy": statistics.mean(row["correct"] for row in rows),
            "mean_response_tokens": statistics.mean(row["response_tokens"] for row in rows),
            "token_limit_hits": hits, "token_limit_hit_rate": hits / len(rows),
            "max_new_tokens": max_new_tokens}


def evaluate(model_path: str, data_path: Path, output: Path, adapter: Path | None = None, *,
             max_new_tokens: int = 64, attn_backend: str = "native",
             lora_rank: int = 8, lora_alpha: float = 16) -> dict:
    import torch
    from safetensors.torch import load_file

    from areno import Trainer
    from areno.adapters.config import LoraConfig
    from areno.api.config import CudaConfig
    from areno.api.models import SamplingParams
    from areno.api.tokenizer import configure_chat_template_enable_thinking
    from examples.async_policy.train import answer_matches, load_prompts

    lora = LoraConfig(rank=lora_rank, alpha=lora_alpha) if adapter is None else LoraConfig(adapter_path=str(adapter))
    trainer = Trainer(world_size=1, model_path=model_path, custom_config=CudaConfig(
        devices=[0], tp_size=1, dp_size=1, max_running_prompts=4, lora=lora,
        runtime={"compile_model": False, "eager_decode": True, "attn_backend": attn_backend},
    ))
    output.mkdir(parents=True, exist_ok=True)
    try:
        trainer.init()
        if adapter is not None:
            exported = output / "reloaded"
            trainer.export_adapter(str(exported))
            before = load_file(str(adapter / "adapter_model.safetensors"))
            after = load_file(str(exported / "adapter_model.safetensors"))
            assert before.keys() == after.keys() and all(torch.equal(value, after[key]) for key, value in before.items())
        tokenizer = trainer.get_tokenizer()
        configure_chat_template_enable_thinking(tokenizer, False)
        prompts = load_prompts(data_path, tokenizer)
        trainer.begin_rollout_session()
        try:
            results = trainer.rollout_token_batch([list(prompt.input_tokens) for prompt in prompts], 1,
                                                  SamplingParams(max_prompt_len=128, max_new_tokens=max_new_tokens,
                                                                 greedy=True, seed=123))
        finally:
            trainer.end_rollout_session()
        rows = []
        for prompt, result in zip(prompts, results, strict=True):
            completion = tokenizer.decode(result.sequences[0].resp_tokens, skip_special_tokens=True)
            rows.append({"prompt_id": prompt.prompt_id, "prompt": prompt.prompt,
                         "answer": prompt.solutions, "completion": completion,
                         "correct": answer_matches(completion, prompt.solutions),
                         "response_tokens": len(result.sequences[0].resp_tokens)})
        result = {**quality_metrics(rows, max_new_tokens), "samples": rows, "greedy": True,
                  "reload_equal_tensors": len(before) if adapter is not None else None}
        write(output / "result.json", result)
        return result
    finally:
        trainer.close()


def measure(summary: dict, output: Path, monitor: Monitor, *, warmup: int = 5) -> dict:
    events = [row for row in summary["updates"] if row["stepped"]]
    if not 1 <= warmup < len(events):
        raise ValueError("warmup must leave at least one measured update")
    if len(events) != summary.get("policy", {}).get("max_steps", len(events)):
        raise ValueError("benchmark did not complete the requested optimizer updates")
    measured_updates = len(events) - warmup
    initial = summary.get("initial_policy_version", 0)
    first_version, final_version = initial + warmup, initial + len(events)
    start, end = events[warmup - 1]["end_ns"], events[-1]["end_ns"]
    duration = (end - start) / 1e9
    samples = [row for row in monitor.rows if start <= row["time_ns"] <= end]
    assert len(samples) >= 2, "insufficient steady-state GPU/process observations"
    tokens = sum(row["response_tokens"] for row in events[warmup:])
    rewards = [reward for row in events[warmup:] for reward in row["rewards"]]
    gpus = {}
    for device in (0, 1):
        readings = [gpu for row in samples for gpu in row["gpus"] if gpu["device"] == device]
        gpus[str(device)] = {"mean_utilization_percent": statistics.mean(row["utilization_percent"] for row in readings),
                             "mean_memory_mib": statistics.mean(row["memory_mib"] for row in readings),
                             "peak_memory_mib": max(row["memory_mib"] for row in readings),
                             "mean_power_watts": statistics.mean(row["power_watts"] for row in readings)}
    pipeline = summary.get("pipeline")
    batches = ([row for row in pipeline["batch_metrics"] if first_version <= row["train_version"] < final_version]
               if pipeline else summary["batch_metrics"][warmup:])
    syncs = pipeline["sync_metrics"] if pipeline else summary["sync_metrics"]
    syncs = [row for row in syncs if first_version <= row["target_version"] < final_version]
    workers = [json.loads(line) for path in output.glob("worker-*.jsonl") for line in path.read_text().splitlines()]
    conflicts = [(sync, operation) for sync in workers if sync["kind"] == "sync"
                 for operation in workers if operation["kind"] in {"train", "rollout"}
                 and max(sync["start_ns"], operation["start_ns"]) < min(sync["end_ns"], operation["end_ns"])]
    assert not conflicts, "weight sync overlapped a model operation"
    stage_seconds = {}
    for row in workers:
        overlap = max(0, min(end, row["end_ns"]) - max(start, row["start_ns"])) / 1e9
        key = f'{row["role"]}/{row["kind"]}'
        stage_seconds[key] = stage_seconds.get(key, 0.0) + overlap
    return {"warmup_updates": warmup, "measured_updates": measured_updates, "start_ns": start, "end_ns": end,
            "duration_s": duration, "updates_per_s": measured_updates / duration, "trained_response_tokens": tokens,
            "trained_response_tokens_per_s": tokens / duration, "mean_train_reward": statistics.mean(rewards),
            "mean_train_rpc_s": statistics.mean((row["end_ns"] - row["start_ns"]) / 1e9 for row in events[warmup:]),
            "sync_transfer_s": sum(row["transfer_s"] for row in syncs),
            "sync_wait_s": sum(row["wait_s"] for row in syncs),
            "queue_wait_s": sum(row.get("wait_s", 0.0) for row in batches),
            "stale_drops": pipeline["batches_dropped_stale"] if pipeline else 0,
            "stale_drop_rate": (pipeline["batches_dropped_stale"] / max(pipeline["batches_seen"], 1)) if pipeline else 0,
            "max_lag": max((row["lag"] for row in pipeline["batch_metrics"] if row["stepped"]), default=0) if pipeline else 0,
            "queue_peak": pipeline["queue_max_depth"] if pipeline else 0,
            "inflight_peak": pipeline["inflight_peak"] if pipeline else 0,
            "gpu": gpus, "monitor_samples": len(samples), "monitor_errors": monitor.errors,
            "mean_process_cpu_percent": statistics.mean(row["cpu_percent"] for row in samples),
            "peak_summed_process_rss_bytes": max(row["summed_process_rss_bytes"] for row in samples),
            "worker_stage_seconds": stage_seconds, "sync_model_overlap_violations": 0}


def parse_args(argv=None):
    from examples.async_policy.train import positive_float, positive_int

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    data = ROOT / "examples" / "async_policy"
    parser.add_argument("--data-path", type=Path, default=data / "train.jsonl")
    parser.add_argument("--eval-path", type=Path, default=data / "eval.jsonl")
    parser.add_argument("--steps", type=positive_int, default=55)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--max-new-tokens", type=positive_int, default=64)
    parser.add_argument("--eval-max-new-tokens", type=positive_int, nargs="+", default=[64])
    parser.add_argument("--n-samples", type=positive_int, default=4)
    parser.add_argument("--lora-rank", type=positive_int, default=8)
    parser.add_argument("--lora-alpha", type=positive_float, default=16)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-5)
    parser.add_argument("--attn-backend", choices=("native", "flash"), default="native")
    parser.add_argument("--loss", choices=("grpo", "gspo"), default="grpo")
    parser.add_argument("--cases", nargs="+", choices=("sync", "lag1", "lag0", "offpolicy"),
                        default=["sync", "lag1", "lag0"])
    parser.add_argument("--queue-capacity", type=positive_int, default=2)
    parser.add_argument("--max-inflight-rollouts", type=positive_int, default=1)
    parser.add_argument("--weight-sync-interval-updates", type=positive_int, default=1)
    parser.add_argument("--lag", type=int, default=1, help="Lag for lag1/offpolicy cases; zero control stays zero")
    parser.add_argument("--dry-run", action="store_true", help="Print the cases without importing GPU workers")
    args = parser.parse_args(argv)
    if args.warmup >= args.steps or args.lag < 0:
        parser.error("require 0 < warmup < steps and lag >= 0")
    if args.loss != "grpo" and "offpolicy" in args.cases:
        parser.error("the offpolicy case is a GRPO ablation")
    if len(set(args.cases)) != len(args.cases) or len(set(args.eval_max_new_tokens)) != len(args.eval_max_new_tokens):
        parser.error("cases and evaluation token limits must be unique")
    return args


def case_plan(args) -> list[dict]:
    cases = []
    for name in args.cases:
        mode = "sync" if name == "sync" else "async"
        lag = 0 if name == "lag0" else args.lag
        loss = "grpo-offpolicy" if name == "offpolicy" else args.loss
        label = f"{mode}-lag{lag}" + ("-offpolicy" if name == "offpolicy" else "")
        cases.append({"case": name, "label": label, "mode": mode, "lag": lag, "loss": loss})
    if len({row["label"] for row in cases}) != len(cases):
        raise ValueError("case settings would overwrite another case's output")
    offset = (args.seed - 41) % len(cases)
    return cases[offset:] + cases[:offset]


def main(argv=None) -> int:
    args = parse_args(argv)
    cases = case_plan(args)
    if args.dry_run:
        print(json.dumps({"cases": cases, "settings": vars(args)}, default=str, indent=2))
        return 0
    from examples.async_policy.tools.gpu_worker import ObservedWorker
    from examples.async_policy.train import run_training

    if (args.output_dir / "environment.json").exists():
        raise FileExistsError("use a fresh benchmark output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    packages = {}
    for name in ("transformers", "modelscope", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write(args.output_dir / "environment.json", {
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "packages": packages,
        "devices": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                             "--format=csv,noheader"], text=True),
    })
    write(args.output_dir / "settings.json", {key: str(value) if isinstance(value, Path) else value
                                              for key, value in vars(args).items()})
    dataset_digests = {"train": hashlib.sha256(args.data_path.read_bytes()).hexdigest(),
                       "eval": hashlib.sha256(args.eval_path.read_bytes()).hexdigest()}
    subprocess.run([sys.executable, ".agents/skills/areno-run-training/scripts/inspect_dataset.py",
                    "--dataset-path", str(args.data_path), "--model-hub", "modelscope", "--algo", args.loss],
                   cwd=ROOT, check=True)
    eval_options = dict(attn_backend=args.attn_backend, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    before = {str(limit): evaluate(args.model_path, args.eval_path, args.output_dir / f"before-{limit}",
                                  max_new_tokens=limit, **eval_options) for limit in args.eval_max_new_tokens}
    results = []
    for case in cases:
        mode, lag = case["mode"], case["lag"]
        output = args.output_dir / case["label"]
        output.mkdir(parents=True, exist_ok=True)
        os.environ.update(ARENO_R4_OUTPUT=str(output), ARENO_R4_PROFILE="0", ARENO_R4_FAULT="")
        monitor = Monitor(output)
        monitor.thread.start()
        try:
            summary = run_training(model_path=args.model_path, data_path=args.data_path, output=output,
                                   mode=mode, steps=args.steps, seed=args.seed, lag=lag, worker_cls=ObservedWorker,
                                   attn_backend=args.attn_backend, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                                   learning_rate=args.learning_rate, n_samples=args.n_samples,
                                   max_new_tokens=args.max_new_tokens, loss=case["loss"],
                                   queue_capacity=args.queue_capacity, max_inflight_rollouts=args.max_inflight_rollouts,
                                   weight_sync_interval_updates=args.weight_sync_interval_updates)
        finally:
            monitor.finish()
        assert summary["ok"]
        assert all(math.isfinite(row["loss"]) for row in summary["native_train_stats"])
        assert any(row.get("metrics", {}).get("grad_norm", 0) > 0 for row in summary["native_train_stats"])
        metrics = measure(summary, output, monitor, warmup=args.warmup)
        after = {str(limit): evaluate(args.model_path, args.eval_path, output / f"evaluation-{limit}",
                                     output / f"checkpoint-{args.steps}", max_new_tokens=limit, **eval_options)
                 for limit in args.eval_max_new_tokens}
        primary = str(args.eval_max_new_tokens[0])
        evaluation = {limit: {"before": {key: value for key, value in before[limit].items() if key != "samples"},
                              "after": {key: value for key, value in value.items() if key != "samples"}}
                      for limit, value in after.items()}
        result = {**case, "seed": args.seed, "ok": True, "source_sha": summary["source_sha"],
                  "metrics": metrics, "quality_before": before[primary]["accuracy"],
                  "quality_after": after[primary]["accuracy"], "evaluations": evaluation,
                  "train_max_new_tokens": args.max_new_tokens, "n_samples": args.n_samples,
                  "dataset_sha256": dataset_digests,
                  "policy": summary["policy"], "attn_backend": args.attn_backend,
                  "held_out_questions": after[primary]["total"],
                  "reload_equal_tensors": after[primary]["reload_equal_tensors"],
                  "reward_curve": [{"version": row["version_before"] + 1,
                                    "mean_reward": statistics.mean(row["rewards"]),
                                    "prompt_ids": row["prompt_ids"], "batch_version": row["batch_version"],
                                    "mean_response_tokens": statistics.mean(row["response_lengths"]),
                                    "token_limit_hits": row["token_limit_hits"]}
                                   for row in summary["updates"] if row["stepped"]]}
        write(output / "benchmark.json", result)
        results.append(result)
        print(json.dumps(result), flush=True)
    write(args.output_dir / "result.json", {"ok": True, "seed": args.seed, "cases": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
