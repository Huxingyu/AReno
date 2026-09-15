"""Preview or execute paired quality, output-length and capacity experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def build_jobs(args) -> list[dict]:
    seeds = args.seeds or ([41, 42, 43, 44, 45] if args.suite == "quality" else [41, 42, 43])
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    jobs = []
    if args.suite in {"quality", "gspo"}:
        configurations = [(limit, 4, 2, 1, 1, 1) for limit in (64, 256)]
        cases = ["sync", "lag1", "lag0", "offpolicy"] if args.suite == "quality" else ["sync", "lag1"]
    elif args.suite == "throughput":
        configurations = [(limit, samples, 2, 1, 1, 1) for limit, samples in itertools.product((256, 512), (4, 8))]
        cases = ["sync", "lag1"]
    else:
        # Lag=1 would force C=4 to sync after two updates, confounding the scan.
        configurations = [(64, 4, q, k, c, 4) for q, k, c in itertools.product((1, 2, 4), (1, 2), (1, 2, 4))]
        cases = ["lag1"]
    for seed, config in itertools.product(seeds, configurations):
        limit, samples, q, k, c, lag = config
        name = f"seed{seed}-tokens{limit}-n{samples}-q{q}-k{k}-c{c}"
        jobs.append({"name": name, "seed": seed, "max_new_tokens": limit, "n_samples": samples,
                     "queue_capacity": q, "max_inflight_rollouts": k, "weight_sync_interval_updates": c,
                     "lag": lag, "cases": cases})
    if args.suite == "capacity":
        jobs.extend({"name": f"seed{seed}-sync", "seed": seed, "max_new_tokens": 64, "n_samples": 4,
                     "queue_capacity": 2, "max_inflight_rollouts": 1, "weight_sync_interval_updates": 1,
                     "lag": 4, "cases": ["sync"]} for seed in seeds)
    for job in jobs:
        job.update(steps=args.steps, warmup=args.warmup, attn_backend=args.attn_backend,
                   loss="gspo" if args.suite == "gspo" else "grpo")
        command = [sys.executable, str(ROOT / "examples/async_policy/tools/benchmark_run.py"),
                   "--model-path", args.model_path, "--output-dir", str(args.output_dir / job["name"]),
                   "--steps", str(args.steps), "--warmup", str(args.warmup), "--attn-backend", args.attn_backend,
                   "--eval-path", str(ROOT / "examples/async_policy/eval128.jsonl"),
                   "--eval-max-new-tokens", "64", "256", "--loss", "gspo" if args.suite == "gspo" else "grpo"]
        for key in ("seed", "max_new_tokens", "n_samples", "queue_capacity", "max_inflight_rollouts",
                    "weight_sync_interval_updates", "lag"):
            command.extend(["--" + key.replace("_", "-"), str(job[key])])
        command.extend(["--cases", *job["cases"]])
        job["command"] = command
    return jobs


def summarize(jobs: list[dict], output: Path) -> dict:
    rows, missing = [], []
    for job in jobs:
        path = output / job["name"] / "result.json"
        if not path.is_file():
            missing.append(job["name"])
            continue
        result = json.loads(path.read_text())
        if (not result["ok"] or len(result["cases"]) != len(job["cases"])
                or {row["case"] for row in result["cases"]} != set(job["cases"])):
            raise ValueError(f"incomplete or failed benchmark: {path}")
        for row in result["cases"]:
            expected_loss = "grpo-offpolicy" if row["case"] == "offpolicy" else job["loss"]
            expected_lag = 0 if row["case"] == "lag0" else job["lag"]
            expected = (job["seed"], job["max_new_tokens"], job["n_samples"], job["attn_backend"], expected_loss,
                        expected_lag, job["steps"], job["warmup"])
            actual = (row["seed"], row["train_max_new_tokens"], row["n_samples"], row["attn_backend"], row["loss"],
                      row["lag"], row["policy"]["max_steps"], row["metrics"]["warmup_updates"])
            if actual != expected or not row["ok"]:
                raise ValueError(f"benchmark settings do not match the manifest: {path}")
            for setting in ("queue_capacity", "max_inflight_rollouts", "weight_sync_interval_updates"):
                if row["policy"][setting] != job[setting]:
                    raise ValueError(f"benchmark {setting} does not match the manifest: {path}")
        rows.extend(result["cases"])
    if len({row["source_sha"] for row in rows}) > 1:
        raise ValueError("comparison mixes source commits")
    if len({json.dumps(row["dataset_sha256"], sort_keys=True) for row in rows}) > 1:
        raise ValueError("comparison mixes datasets")
    baselines = {(row["seed"], row["train_max_new_tokens"], row["n_samples"], row["loss"], row["attn_backend"]): row
                 for row in rows if row["case"] == "sync"}
    groups = defaultdict(list)
    for row in rows:
        policy = row["policy"]
        key = (row["case"], row["loss"], row["train_max_new_tokens"], row["n_samples"], row["lag"],
               policy["queue_capacity"], policy["max_inflight_rollouts"], policy["weight_sync_interval_updates"],
               row["attn_backend"])
        groups[key].append(row)
    aggregate = []
    for key, values in sorted(groups.items()):
        if len({row["seed"] for row in values}) != len(values):
            raise ValueError("duplicate seeds in a comparison group")
        paired = []
        for row in values:
            loss = "grpo" if row["loss"] == "grpo-offpolicy" else row["loss"]
            baseline = baselines.get((row["seed"], row["train_max_new_tokens"], row["n_samples"], loss, row["attn_backend"]))
            if baseline is None:
                continue
            paired.append({"seed": row["seed"],
                           "updates_per_s_ratio": row["metrics"]["updates_per_s"] / baseline["metrics"]["updates_per_s"],
                           "accuracy_delta_by_eval_limit": {
                               limit: score["after"]["accuracy"] - baseline["evaluations"][limit]["after"]["accuracy"]
                               for limit, score in row["evaluations"].items()}})
        aggregate.append({"configuration": dict(zip(("case", "loss", "train_max_new_tokens", "n_samples", "lag",
                                                     "queue_capacity", "max_inflight_rollouts",
                                                     "weight_sync_interval_updates", "attn_backend"), key, strict=True)),
                          "seeds": sorted(row["seed"] for row in values), "paired_with_sync": paired,
                          "mean_updates_per_s": statistics.mean(row["metrics"]["updates_per_s"] for row in values),
                          "evaluations": {limit: {
                              "accuracy_by_seed": {str(row["seed"]): row["evaluations"][limit]["after"]["accuracy"] for row in values},
                              "mean_accuracy": statistics.mean(row["evaluations"][limit]["after"]["accuracy"] for row in values),
                              "mean_token_limit_hit_rate": statistics.mean(row["evaluations"][limit]["after"]["token_limit_hit_rate"] for row in values),
                              "mean_response_tokens": statistics.mean(row["evaluations"][limit]["after"]["mean_response_tokens"] for row in values),
                          } for limit in values[0]["evaluations"]}})
    return {"complete": not missing, "missing_jobs": missing, "expected_jobs": len(jobs),
            "completed_jobs": len(jobs) - len(missing), "groups": aggregate,
            "source_sha": rows[0]["source_sha"] if rows else None,
            "dataset_sha256": rows[0]["dataset_sha256"] if rows else None,
            "interpretation": "Paired seed observations, not proof of convergence or causal attribution."}


def run_job(command: list[str], timeout_s: float) -> None:
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout_s)
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        # Kill the whole owned group, including worker grandchildren.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quality", "throughput", "capacity", "gspo"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=55)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--attn-backend", choices=("native", "flash"), default="native")
    parser.add_argument("--job-timeout-s", type=int, default=3600)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Start the GPU jobs; omitted means preview only")
    action.add_argument("--summarize", action="store_true", help="Aggregate existing results without starting jobs")
    args = parser.parse_args()
    if not 0 < args.warmup < args.steps or args.job_timeout_s < 1:
        parser.error("require 0 < warmup < steps and a positive job timeout")
    args.output_dir = args.output_dir.resolve()
    jobs = build_jobs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"suite": args.suite, "status": "planned", "job_count": len(jobs),
                "training_runs": sum(len(job["cases"]) for job in jobs),
                "job_timeout_s": args.job_timeout_s, "jobs": jobs}
    if not args.summarize:
        (args.output_dir / "matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.execute:
        for job in jobs:
            run_job(job["command"], args.job_timeout_s)
    if args.execute or args.summarize:
        result = summarize(jobs, args.output_dir)
        (args.output_dir / "benchmark-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "groups"}))
        return 0 if result["complete"] else 1
    print(json.dumps({key: value for key, value in manifest.items() if key != "jobs"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
