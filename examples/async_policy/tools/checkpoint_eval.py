"""Evaluate saved training cases on one GPU without repeating paid training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.async_policy.tools.benchmark_run import (  # noqa: E402
    environment_metadata,
    evaluate,
    model_manifest,
    quality_metrics,
)
from examples.async_policy.tools.campaign_state import (  # noqa: E402
    code_fingerprint,
    complete_directory,
    fingerprint,
    sha256_file,
    training_code_fingerprint,
    verify_complete,
    verify_manifest,
    write_json,
)


def cached_evaluate(identity: dict, cache: Path, evaluator) -> tuple[dict, dict]:
    """Cache only a complete, hashed result; failed attempts remain inspectable."""
    key = fingerprint(identity)
    entry = cache / key
    if (entry / "complete.json").exists():
        verify_complete(entry, identity)
        return json.loads((entry / "result.json").read_text()), {"key": key, "hit": True}
    attempt = entry / f"attempt-{uuid.uuid4().hex[:12]}"
    attempt.mkdir(parents=True)
    write_json(attempt / "identity.json", identity)
    result = evaluator(attempt)
    # Non-finite metrics cannot be cached by write_json's strict JSON encoder.
    if result["total"] <= 0 or len(result["samples"]) != result["total"]:
        raise ValueError("evaluation result is incomplete")
    write_json(entry / "result.json", result)
    complete_directory(entry, identity, paths=[entry / "result.json"])
    return result, {"key": key, "hit": False, "attempt": attempt.name}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, default=ROOT / "examples/async_policy/eval128.jsonl")
    parser.add_argument("--eval-max-new-tokens", type=int, nargs="+", default=[64, 256])
    args = parser.parse_args(argv)
    if (any(limit < 1 for limit in args.eval_max_new_tokens)
            or len(set(args.eval_max_new_tokens)) != len(args.eval_max_new_tokens)):
        parser.error("evaluation token limits must be positive and unique")
    if (args.output_dir / "result.json").exists():
        raise FileExistsError("use a fresh evaluation attempt directory")
    training = json.loads((args.training_output / "training-result.json").read_text())
    if not training["ok"] or not training["cases"]:
        raise ValueError("training did not complete")
    environment = environment_metadata()
    model = model_manifest(args.model_path)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    evaluator_sha = code_fingerprint("areno", "examples/async_policy/train.py",
                                    "examples/async_policy/tools/benchmark_run.py",
                                    "examples/async_policy/tools/checkpoint_eval.py")
    eval_hash = sha256_file(args.eval_path)
    train_code = training_code_fingerprint()
    reference_prompts = {json.loads(line)["prompt"] for line in
                         (ROOT / "examples/async_policy/eval.jsonl").read_text().splitlines() if line.strip()}
    results = []
    for row in training["cases"]:
        if row["training_code_sha256"] != train_code:
            raise ValueError("training implementation changed; audit compatibility before reusing this checkpoint")
        if row["model_sha256"] != model["sha256"]:
            raise ValueError("base model/tokenizer files changed")
        checkpoint = args.training_output / row["checkpoint"]
        if Path(row["checkpoint"]).is_absolute() or ".." in Path(row["checkpoint"]).parts:
            raise ValueError("checkpoint reference escapes training output")
        verify_manifest(checkpoint, row["checkpoint_manifest"])
        settings = row["training_settings"]
        options = {key: settings[key] for key in ("attn_backend", "lora_rank", "lora_alpha")}
        evaluations, cache_records = {}, {}
        for limit in args.eval_max_new_tokens:
            identity = {"model_sha256": model["sha256"], "dataset_sha256": eval_hash,
                        "evaluator_sha256": evaluator_sha, "environment": environment,
                        "sampling": {"max_new_tokens": limit, "max_prompt_len": 128,
                                     "greedy": True, "seed": 123, "n_samples": 1,
                                     "enable_thinking": False, **options}}
            before, before_cache = cached_evaluate(
                {**identity, "checkpoint_sha256": None}, args.cache_dir,
                lambda output: evaluate(args.model_path, args.eval_path, output,
                                        max_new_tokens=limit, **options))
            after, after_cache = cached_evaluate(
                {**identity, "checkpoint_sha256": row["checkpoint_manifest"]["sha256"]}, args.cache_dir,
                lambda output: evaluate(args.model_path, args.eval_path, output, checkpoint,
                                        max_new_tokens=limit, **options))
            original = [sample for sample in after["samples"] if sample["prompt"] in reference_prompts]
            evaluations[str(limit)] = {
                "before": {key: value for key, value in before.items() if key != "samples"},
                "after": {key: value for key, value in after.items() if key != "samples"},
                "original_subset": quality_metrics(original, limit) if original else None,
            }
            # Retain per-question evidence independently of the shared cache.
            write_json(args.output_dir / row["label"] / f"evaluation-{limit}.json", after)
            cache_records[str(limit)] = {"before": before_cache, "after": after_cache}
            write_json(args.output_dir / "progress.json", {"case": row["case"], "evaluations": evaluations})
        verify_manifest(checkpoint, row["checkpoint_manifest"])
        primary = str(args.eval_max_new_tokens[0])
        result = {**row, "evaluation_source_sha": source_sha, "evaluator_sha256": evaluator_sha,
                  "evaluation_environment": environment, "evaluations": evaluations,
                  "dataset_sha256": {**row["dataset_sha256"], "eval": eval_hash},
                  "cache": cache_records, "checkpoint_unchanged": True,
                  "held_out_questions": evaluations[primary]["after"]["total"],
                  "reload_equal_tensors": evaluations[primary]["after"]["reload_equal_tensors"],
                  "quality_before": evaluations[primary]["before"]["accuracy"],
                  "quality_after": evaluations[primary]["after"]["accuracy"]}
        results.append(result)
    write_json(args.output_dir / "result.json", {"ok": True, "cases": results})
    write_json(args.output_dir / "environment.json", environment)
    complete_directory(args.output_dir, {"stage": "evaluate", "eval_sha256": eval_hash,
                                        "limits": args.eval_max_new_tokens})
    print(json.dumps({"ok": True, "cases": len(results), "cache": [row["cache"] for row in results]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
