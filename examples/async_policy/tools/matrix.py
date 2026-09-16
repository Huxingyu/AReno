"""Preview or execute paired quality, output-length and capacity experiments."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import itertools
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.async_policy.tools.campaign_state import (  # noqa: E402
    BudgetLedger,
    code_fingerprint,
    complete_directory,
    exclusive_lock,
    fingerprint,
    sha256_file,
    training_code_fingerprint,
    verify_complete,
    write_json,
)


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
    individual = []
    for job in jobs:
        job.update(steps=args.steps, warmup=args.warmup, attn_backend=args.attn_backend,
                   loss="gspo" if args.suite == "gspo" else "grpo",
                   lora_rank=getattr(args, "lora_rank", 8), lora_alpha=getattr(args, "lora_alpha", 16.0),
                   learning_rate=getattr(args, "learning_rate", 1e-5),
                   eval_max_new_tokens=[64, 256] if args.suite in {"quality", "gspo"} else [256])
        # Rotate paired case order without letting selection change the order.
        offset = (job["seed"] - 41) % len(job["cases"])
        for case in job["cases"][offset:] + job["cases"][:offset]:
            row = {**job, "name": f"{job['name']}-{case}", "cases": [case]}
            row["task_id"] = fingerprint({key: value for key, value in row.items() if key != "name"})[:16]
            individual.append(row)
    return individual


def expected_settings(job: dict) -> dict:
    case = job["cases"][0]
    keys = ("seed", "steps", "warmup", "max_new_tokens", "n_samples", "lora_rank", "lora_alpha",
            "learning_rate", "attn_backend", "queue_capacity", "max_inflight_rollouts",
            "weight_sync_interval_updates")
    return {**{key: job[key] for key in keys}, "mode": "sync" if case == "sync" else "async",
            "lag": 0 if case == "lag0" else job["lag"],
            "loss": "grpo-offpolicy" if case == "offpolicy" else job["loss"]}


def distribution(values: list[float]) -> dict:
    interval = None
    if len(values) >= 2:
        rng = random.Random(0)
        means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2500))
        interval = [means[62], means[2437]]
    return {"values": values, "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "seed_bootstrap_95_percent_interval": interval,
            "interval_interpretation": "Exploratory paired-seed bootstrap; few seeds do not establish stability."}


def completed_attempt(stage: Path, identity: dict | None = None) -> Path | None:
    for attempt in sorted(stage.glob("attempt-*"), reverse=True):
        if not (attempt / "complete.json").exists():
            continue
        marker = json.loads((attempt / "complete.json").read_text())
        if identity is not None and marker["identity"] != identity:
            continue
        verify_complete(attempt, identity)
        return attempt
    return None


def stage_result(attempt: Path, stage: str) -> dict:
    name = "training-result.json" if stage == "train" else "result.json"
    return json.loads((attempt / "artifacts" / name).read_text())


def summarize(jobs: list[dict], output: Path) -> dict:
    rows, missing, failures = [], [], []
    expected_eval_hash = sha256_file(ROOT / "examples/async_policy/eval128.jsonl")
    expected_train_hash = sha256_file(ROOT / "examples/async_policy/train.jsonl")
    expected_questions = len((ROOT / "examples/async_policy/eval128.jsonl").read_text().splitlines())
    for job in jobs:
        attempt = completed_attempt(output / job["name"] / "evaluate")
        if attempt is None:
            missing.append(job["name"])
            attempts = sorted((output / job["name"]).glob("*/attempt-*/attempt.json"))
            for path in attempts:
                state = json.loads(path.read_text())
                if state["status"] != "succeeded":
                    failures.append({"job": job["name"], "attempt": str(path.relative_to(output)),
                                     "status": state["status"]})
            continue
        path = attempt / "artifacts/result.json"
        result = json.loads(path.read_text())
        if (not result["ok"] or len(result["cases"]) != len(job["cases"])
                or {row["case"] for row in result["cases"]} != set(job["cases"])):
            raise ValueError(f"incomplete or failed benchmark: {path}")
        for row in result["cases"]:
            if row["training_settings"] != expected_settings(job):
                raise ValueError(f"training parameters do not match the manifest: {path}")
            if set(row["evaluations"]) != {str(limit) for limit in job["eval_max_new_tokens"]}:
                raise ValueError(f"missing or incompatible evaluation token limits: {path}")
            if (row["held_out_questions"] != expected_questions
                    or row["dataset_sha256"]["eval"] != expected_eval_hash
                    or row["dataset_sha256"]["train"] != expected_train_hash
                    or not row.get("checkpoint_unchanged")):
                raise ValueError(f"incompatible evaluation dataset or changed checkpoint: {path}")
            for evaluation in row["evaluations"].values():
                if any(evaluation[which]["total"] != expected_questions for which in ("before", "after")):
                    raise ValueError(f"incomplete held-out evaluation: {path}")
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
    for field in ("training_code_sha256", "evaluator_sha256", "model_sha256", "environment", "evaluation_environment"):
        if len({json.dumps(row[field], sort_keys=True) for row in rows}) > 1:
            raise ValueError(f"comparison mixes incompatible {field}")
    if len({json.dumps(row["dataset_sha256"], sort_keys=True) for row in rows}) > 1:
        raise ValueError("comparison mixes datasets")
    def baseline_key(row):
        settings = row["training_settings"]
        return (row["seed"], row["train_max_new_tokens"], row["n_samples"],
                "grpo" if row["loss"] == "grpo-offpolicy" else row["loss"], row["attn_backend"],
                settings["learning_rate"], settings["lora_rank"], settings["lora_alpha"],
                settings["steps"], settings["warmup"])

    baselines = {}
    for row in rows:
        if row["case"] == "sync":
            key = baseline_key(row)
            if key in baselines:
                raise ValueError("duplicate synchronous baseline")
            baselines[key] = row
    groups = defaultdict(list)
    for row in rows:
        policy = row["policy"]
        key = (row["case"], row["loss"], row["train_max_new_tokens"], row["n_samples"], row["lag"],
               policy["queue_capacity"], policy["max_inflight_rollouts"], policy["weight_sync_interval_updates"],
               row["attn_backend"])
        groups[key].append(row)
    aggregate, missing_baselines = [], []
    for key, values in sorted(groups.items()):
        if len({row["seed"] for row in values}) != len(values):
            raise ValueError("duplicate seeds in a comparison group")
        paired = []
        for row in values:
            baseline = baselines.get(baseline_key(row))
            if baseline is None:
                missing_baselines.append({"seed": row["seed"], "case": row["case"],
                                          "train_max_new_tokens": row["train_max_new_tokens"],
                                          "n_samples": row["n_samples"]})
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
                          "paired_updates_per_s_ratio": distribution([row["updates_per_s_ratio"] for row in paired]) if paired else None,
                          "paired_accuracy_delta": {
                              limit: distribution([row["accuracy_delta_by_eval_limit"][limit] for row in paired])
                              for limit in values[0]["evaluations"]} if paired else {},
                          "updates_per_s_statistics": distribution([row["metrics"]["updates_per_s"] for row in values]),
                          "mean_updates_per_s": statistics.mean(row["metrics"]["updates_per_s"] for row in values),
                          "evaluations": {limit: {
                              "accuracy_by_seed": {str(row["seed"]): row["evaluations"][limit]["after"]["accuracy"] for row in values},
                              "mean_accuracy": statistics.mean(row["evaluations"][limit]["after"]["accuracy"] for row in values),
                              "mean_token_limit_hit_rate": statistics.mean(row["evaluations"][limit]["after"]["token_limit_hit_rate"] for row in values),
                              "mean_response_tokens": statistics.mean(row["evaluations"][limit]["after"]["mean_response_tokens"] for row in values),
                          } for limit in values[0]["evaluations"]}})
    return {"complete": not missing and not missing_baselines, "missing_jobs": missing,
            "failed_or_interrupted_attempts": failures, "missing_sync_baselines": missing_baselines,
            "expected_jobs": len(jobs),
            "completed_jobs": len(jobs) - len(missing), "groups": aggregate,
            "training_source_shas": sorted({row["training_source_sha"] for row in rows}),
            "evaluation_source_shas": sorted({row["evaluation_source_sha"] for row in rows}),
            "dataset_sha256": rows[0]["dataset_sha256"] if rows else None,
            "interpretation": "Paired seed observations, not proof of convergence or causal attribution."}


def run_job(command: list[str], timeout_s: float, *, log=None) -> None:
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                               stdout=log, stderr=subprocess.STDOUT if log else None)
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


def benchmark_arguments(job: dict) -> list[str]:
    command = ["--train-only", "--eval-path", "examples/async_policy/eval128.jsonl"]
    for key in ("seed", "max_new_tokens", "n_samples", "queue_capacity", "max_inflight_rollouts",
                "weight_sync_interval_updates", "lag", "steps", "warmup", "attn_backend", "loss",
                "lora_rank", "lora_alpha", "learning_rate"):
        command.extend(["--" + key.replace("_", "-"), str(job[key])])
    command.extend(["--cases", *job["cases"]])
    return command


def stage_identity(job: dict, stage: str, *, training_code: str, evaluator_code: str) -> dict:
    identity = {"task_id": job["task_id"], "stage": stage, "training_code_sha256": training_code,
                "settings": expected_settings(job), "train_sha256": sha256_file(ROOT / "examples/async_policy/train.jsonl")}
    if stage == "evaluate":
        identity.update(evaluator_sha256=evaluator_code, eval_limits=job["eval_max_new_tokens"],
                        eval_sha256=sha256_file(ROOT / "examples/async_policy/eval128.jsonl"))
    return identity


def run_stage(args, job: dict, stage: str, identity: dict, *, deadline: float,
              training: Path | None = None) -> Path | None:
    directory = args.output_dir / job["name"] / stage
    complete = completed_attempt(directory, identity)
    if complete is not None:
        return complete
    previous = sorted(directory.glob("attempt-*"))
    matching = [path for path in previous if (path / "attempt.json").exists()
                and json.loads((path / "attempt.json").read_text())["identity"] == identity]
    if len(matching) >= args.max_attempts:
        print(json.dumps({"job": job["name"], "stage": stage, "status": "attempt_limit"}), flush=True)
        return None
    remaining = min(args.job_timeout_s, deadline - time.time())
    if remaining < (180 if args.backend == "modal" else 60):
        return None
    attempt = directory / f"attempt-{len(previous) + 1:04d}"
    attempt.mkdir(parents=True)
    if stage == "train":
        forwarded = benchmark_arguments(job)
        phase = "benchmark"
    else:
        if training is None:
            raise ValueError("evaluation requires a completed training attempt")
        if args.backend == "modal":
            reference = json.loads((training / "result.json").read_text())["volume_output_dir"]
        else:
            reference = str(training / "artifacts")
        phase = "checkpoint-eval"
        forwarded = ["--training-output", reference,
                     "--eval-max-new-tokens", *map(str, job["eval_max_new_tokens"])]
    if args.backend == "modal":
        # Leave cleanup inside the job/campaign wall-clock allowance.
        work_seconds = max(30, int(remaining) - 120)
        command = [args.modal_python, str(ROOT / "examples/async_policy/tools/modal_run.py"),
                   "--phase", phase, "--output-dir", str(attempt),
                   "--task-timeout-s", str(max(300, work_seconds)),
                   "--controller-timeout-s", str(work_seconds),
                   "--budget-file", str(args.budget_file), "--benchmark-args", *forwarded]
    else:
        script = "benchmark_run.py" if stage == "train" else "checkpoint_eval.py"
        command = [sys.executable, str(ROOT / "examples/async_policy/tools" / script),
                   "--model-path", args.model_path, "--output-dir", str(attempt / "artifacts"), *forwarded]
        if stage == "evaluate":
            command.extend(["--cache-dir", str(args.output_dir / "evaluation-cache")])
    state = {"identity": identity, "command": command, "started_at": time.time(), "status": "running"}
    write_json(attempt / "attempt.json", state)
    print(json.dumps({"job": job["name"], "stage": stage, "attempt": attempt.name, "status": "running"}), flush=True)
    try:
        with (attempt / "console.log").open("x") as log:
            # Modal owns a tighter deadline and cleanup. This outer guard also
            # covers a broken controller, while retaining its full reservation.
            run_job(command, remaining + (180 if args.backend == "modal" else 0), log=log)
        result = stage_result(attempt, stage)
        if (not result["ok"] or len(result["cases"]) != 1
                or result["cases"][0]["training_settings"] != expected_settings(job)):
            raise ValueError("case result is incomplete or incompatible with its requested training settings")
        state.update(status="succeeded", finished_at=time.time())
        write_json(attempt / "attempt.json", state)
        complete_directory(attempt, identity)
        return attempt
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     finished_at=time.time(), error=f"{type(exc).__name__}: {exc}")
        write_json(attempt / "attempt.json", state)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(json.dumps({"job": job["name"], "stage": stage, **state}), flush=True)
        return None


def execute(args, jobs: list[dict], manifest: dict) -> None:
    deadline = time.time() + args.campaign_timeout_s
    evaluator_code = code_fingerprint("areno", "examples/async_policy/train.py",
                                      "examples/async_policy/tools/benchmark_run.py",
                                      "examples/async_policy/tools/checkpoint_eval.py")
    selected = [job for job in jobs if not args.select or any(fnmatch.fnmatchcase(job["name"], pattern)
                                                            for pattern in args.select)]
    for job in selected:
        if time.time() >= deadline - 60:
            break
        train_identity = stage_identity(job, "train", training_code=manifest["training_code_sha256"],
                                        evaluator_code=evaluator_code)
        training = completed_attempt(args.output_dir / job["name"] / "train", train_identity)
        if training is None and args.stage != "evaluate":
            training = run_stage(args, job, "train", train_identity, deadline=deadline)
            if training is None:
                # Infrastructure/finite-loss/integrity errors stop the batch.
                break
        if training is not None and args.stage != "train":
            eval_identity = stage_identity(job, "evaluate", training_code=manifest["training_code_sha256"],
                                           evaluator_code=evaluator_code)
            evaluation = run_stage(args, job, "evaluate", eval_identity, deadline=deadline, training=training)
            if evaluation is None:
                break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quality", "throughput", "capacity", "gspo"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=55)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--attn-backend", choices=("native", "flash"), default="native")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--job-timeout-s", type=int, default=1200)
    parser.add_argument("--campaign-timeout-s", type=int, default=7200)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--backend", choices=("local", "modal"), default="local")
    parser.add_argument("--modal-python", default=sys.executable)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--select", nargs="+", help="Execute matching case names; the full manifest remains incomplete")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Start the GPU jobs; omitted means preview only")
    action.add_argument("--summarize", action="store_true", help="Aggregate existing results without starting jobs")
    args = parser.parse_args()
    if (not 0 < args.warmup < args.steps or args.job_timeout_s < 60
            or args.campaign_timeout_s < 60 or args.max_attempts < 1
            or min(args.lora_rank, args.lora_alpha, args.learning_rate) <= 0):
        parser.error("require valid training settings, positive attempts and timeouts of at least 60 seconds")
    if args.execute and args.backend == "modal" and args.budget_file is None:
        parser.error("Modal execution requires a shared --budget-file")
    args.output_dir = args.output_dir.resolve()
    if args.budget_file:
        args.budget_file = args.budget_file.resolve()
    jobs = build_jobs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 2, "suite": args.suite, "model": args.model_path, "job_count": len(jobs),
                "training_runs": len(jobs), "training_code_sha256": training_code_fingerprint(),
                "datasets": {name: hashlib.sha256((ROOT / "examples/async_policy" / name).read_bytes()).hexdigest()
                             for name in ("train.jsonl", "eval128.jsonl")}, "jobs": jobs}
    with exclusive_lock(args.output_dir / ".controller.lock"):
        path = args.output_dir / "matrix.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("campaign settings/source changed; use a new output directory and audit reuse explicitly")
        if not path.exists():
            write_json(path, manifest)
        preview = {key: value for key, value in manifest.items() if key != "jobs"}
        preview.update(backend=args.backend, job_timeout_s=args.job_timeout_s,
                       campaign_timeout_s=args.campaign_timeout_s, max_attempts=args.max_attempts,
                       pending=[job["name"] for job in jobs
                                if completed_attempt(args.output_dir / job["name"] / "evaluate") is None])
        if args.execute and args.backend == "modal":
            preview["budget"] = BudgetLedger(args.budget_file, args.budget_usd).summary()
        print(json.dumps(preview), flush=True)
        if args.execute:
            execute(args, jobs, manifest)
        if args.execute or args.summarize:
            result = summarize(jobs, args.output_dir)
            write_json(args.output_dir / "benchmark-summary.json", result)
            print(json.dumps({key: value for key, value in result.items() if key != "groups"}))
            return 0 if result["complete"] else 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
