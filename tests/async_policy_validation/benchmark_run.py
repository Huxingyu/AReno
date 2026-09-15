"""Repeated steady-state and held-out-quality acceptance using the shipped example."""

from __future__ import annotations

import argparse
import csv
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

ROOT = Path(__file__).resolve().parents[2]
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


def evaluate(model_path: str, data_path: Path, output: Path, adapter: Path | None = None) -> dict:
    import torch
    from safetensors.torch import load_file

    from areno import Trainer
    from areno.adapters.config import LoraConfig
    from areno.api.config import CudaConfig
    from areno.api.models import SamplingParams
    from areno.api.tokenizer import configure_chat_template_enable_thinking
    from examples.async_policy.train import answer_matches, load_prompts

    lora = LoraConfig(rank=8, alpha=16) if adapter is None else LoraConfig(adapter_path=str(adapter))
    trainer = Trainer(world_size=1, model_path=model_path, custom_config=CudaConfig(
        devices=[0], tp_size=1, dp_size=1, max_running_prompts=4, lora=lora,
        runtime={"compile_model": False, "eager_decode": True, "attn_backend": "flash"},
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
                                                  SamplingParams(max_prompt_len=128, max_new_tokens=64,
                                                                 greedy=True, seed=123))
        finally:
            trainer.end_rollout_session()
        rows = []
        for prompt, result in zip(prompts, results, strict=True):
            completion = tokenizer.decode(result.sequences[0].resp_tokens, skip_special_tokens=True)
            rows.append({"prompt": prompt.prompt, "answer": prompt.solutions, "completion": completion,
                         "correct": answer_matches(completion, prompt.solutions),
                         "response_tokens": len(result.sequences[0].resp_tokens)})
        result = {"correct": sum(row["correct"] for row in rows), "total": len(rows), "samples": rows,
                  "accuracy": statistics.mean(row["correct"] for row in rows), "greedy": True,
                  "reload_equal_tensors": len(before) if adapter is not None else None}
        write(output / "result.json", result)
        return result
    finally:
        trainer.close()


def measure(summary: dict, output: Path, monitor: Monitor, *, warmup: int = 5) -> dict:
    events = [row for row in summary["updates"] if row["stepped"]]
    assert len(events) == 55
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
    batches = ([row for row in pipeline["batch_metrics"] if warmup <= row["train_version"] < 55]
               if pipeline else summary["batch_metrics"][warmup:])
    syncs = pipeline["sync_metrics"] if pipeline else summary["sync_metrics"]
    syncs = [row for row in syncs if warmup <= row["target_version"] < 55]
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
    return {"warmup_updates": warmup, "measured_updates": 50, "start_ns": start, "end_ns": end,
            "duration_s": duration, "updates_per_s": 50 / duration, "trained_response_tokens": tokens,
            "trained_response_tokens_per_s": tokens / duration, "mean_train_reward": statistics.mean(rewards),
            "mean_train_rpc_s": statistics.mean((row["end_ns"] - row["start_ns"]) / 1e9 for row in events[warmup:]),
            "sync_transfer_s": sum(row["transfer_s"] for row in syncs),
            "sync_wait_s": sum(row["wait_s"] for row in syncs),
            "queue_wait_s": sum(row.get("wait_s", 0.0) for row in batches),
            "stale_drops": pipeline["batches_dropped_stale"] if pipeline else 0,
            "max_lag": max((row["lag"] for row in pipeline["batch_metrics"] if row["stepped"]), default=0) if pipeline else 0,
            "queue_peak": pipeline["queue_max_depth"] if pipeline else 0,
            "inflight_peak": pipeline["inflight_peak"] if pipeline else 0,
            "gpu": gpus, "monitor_samples": len(samples), "monitor_errors": monitor.errors,
            "mean_process_cpu_percent": statistics.mean(row["cpu_percent"] for row in samples),
            "peak_summed_process_rss_bytes": max(row["summed_process_rss_bytes"] for row in samples),
            "worker_stage_seconds": stage_seconds, "sync_model_overlap_violations": 0}


def main() -> int:
    from gpu_worker import ObservedWorker

    from examples.async_policy.train import run_training

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    write(args.output_dir / "environment.json", {
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in ("transformers", "modelscope", "flash-attn")},
        "devices": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                             "--format=csv,noheader"], text=True),
    })
    data = ROOT / "examples" / "async_policy"
    subprocess.run([sys.executable, ".agents/skills/areno-run-training/scripts/inspect_dataset.py",
                    "--dataset-path", str(data / "train.jsonl"), "--model-hub", "modelscope", "--algo", "grpo"], check=True)
    before = evaluate(args.model_path, data / "eval.jsonl", args.output_dir / "before")
    cases = [("sync", 1), ("async", 1), ("async", 0)]
    offset = (args.seed - 41) % len(cases)
    cases = cases[offset:] + cases[:offset]
    results = []
    for mode, lag in cases:
        output = args.output_dir / f"{mode}-lag{lag}"
        output.mkdir(parents=True, exist_ok=True)
        os.environ.update(ARENO_R4_OUTPUT=str(output), ARENO_R4_PROFILE="0", ARENO_R4_FAULT="")
        monitor = Monitor(output)
        monitor.thread.start()
        try:
            summary = run_training(model_path=args.model_path, data_path=data / "train.jsonl", output=output,
                                   mode=mode, steps=55, seed=args.seed, lag=lag, worker_cls=ObservedWorker)
        finally:
            monitor.finish()
        assert summary["ok"]
        assert all(math.isfinite(row["loss"]) for row in summary["native_train_stats"])
        assert any(row.get("metrics", {}).get("grad_norm", 0) > 0 for row in summary["native_train_stats"])
        metrics = measure(summary, output, monitor)
        after = evaluate(args.model_path, data / "eval.jsonl", output / "evaluation", output / "checkpoint-55")
        result = {"mode": mode, "lag": lag, "seed": args.seed, "ok": True, "source_sha": summary["source_sha"],
                  "metrics": metrics, "quality_before": before["accuracy"], "quality_after": after["accuracy"],
                  "held_out_questions": after["total"], "reload_equal_tensors": after["reload_equal_tensors"]}
        write(output / "benchmark.json", result)
        results.append(result)
        print(json.dumps(result), flush=True)
    write(args.output_dir / "result.json", {"ok": True, "seed": args.seed, "cases": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
