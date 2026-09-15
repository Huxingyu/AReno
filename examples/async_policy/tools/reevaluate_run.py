"""Reevaluate archived benchmark checkpoints without changing their weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.async_policy.tools.benchmark_run import evaluate, quality_metrics, write  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, default=ROOT / "examples/async_policy/eval128.jsonl")
    parser.add_argument("--eval-max-new-tokens", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--attn-backend", choices=("native", "flash"), default="flash")
    args = parser.parse_args(argv)
    if Path(args.archive).name != args.archive:
        parser.error("--archive must name a file within --archive-root")
    if any(limit < 1 for limit in args.eval_max_new_tokens) or len(set(args.eval_max_new_tokens)) != len(args.eval_max_new_tokens):
        parser.error("evaluation token limits must be positive and unique")
    if (args.output_dir / "result.json").exists():
        raise FileExistsError("use a fresh reevaluation output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.archive_root / args.archive
    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    summary = {"ok": False, "archive": args.archive, "archive_sha256": digest.hexdigest(),
               "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "dataset_sha256": hashlib.sha256(args.eval_path.read_bytes()).hexdigest(),
               "attn_backend": args.attn_backend, "cases": []}
    with tempfile.TemporaryDirectory(prefix="archived-", dir=args.output_dir) as temporary:
        source = Path(temporary)
        with tarfile.open(archive_path) as archive:
            archive.extractall(source, filter="data")
        summary["training_source_sha"] = json.loads((source / "modal-task.json").read_text())["source_sha"]
        summary["seed"] = json.loads((source / "result.json").read_text())["seed"]
        for name in ("before", "sync-lag1", "async-lag1", "async-lag0"):
            reference_path = source / name / ("result.json" if name == "before" else "evaluation/result.json")
            reference = json.loads(reference_path.read_text())
            reference_rows = {row["prompt"]: row for row in reference["samples"]}
            adapter = None
            if name != "before":
                checkpoints = sorted((source / name).glob("checkpoint-*"),
                                     key=lambda path: int(path.name.rsplit("-", 1)[-1]))
                if not checkpoints:
                    raise ValueError(f"missing archived checkpoint for {name}")
                adapter = checkpoints[-1]
            for limit in args.eval_max_new_tokens:
                output = args.output_dir / name / f"tokens-{limit}"
                result = evaluate(args.model_path, args.eval_path, output, adapter,
                                  max_new_tokens=limit, attn_backend=args.attn_backend)
                original = [row for row in result["samples"] if row["prompt"] in reference_rows]
                if len(original) != len(reference_rows):
                    raise ValueError("evaluation dataset must retain every original question")
                entry = {"case": name, "max_new_tokens": limit,
                         "checkpoint": str(adapter.relative_to(source)) if adapter else None,
                         "metrics": quality_metrics(result["samples"], limit),
                         "original_subset": quality_metrics(original, limit),
                         "reload_equal_tensors": result["reload_equal_tensors"],
                         "reference64_matches": all(
                             all(row[key] == reference_rows[row["prompt"]][key]
                                 for key in ("completion", "correct", "response_tokens"))
                             for row in original) if limit == 64 else None}
                summary["cases"].append(entry)
                write(args.output_dir / "progress.json", summary)
                print(json.dumps(entry), flush=True)
    summary["ok"] = True
    write(args.output_dir / "result.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
