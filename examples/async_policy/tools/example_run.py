"""Exercise the shipped CLI example, including optimizer-aware continuation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for label, steps in (("fresh", 3), ("resumed", 2)):
        output = args.output_dir / label
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "examples/async_policy/train.py", "--model", args.model_path,
                   "--steps", str(steps), "--output-dir", str(output), "--save-training-state"]
        if label == "resumed":
            command.extend(["--resume-from", str(args.output_dir / "fresh" / "checkpoint-3")])
        with (output / "console.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)
        result = json.loads((output / "result.json").read_text())
        assert result["ok"] and result["pipeline"]["steps"] == steps
        expected = 3 if label == "fresh" else 5
        assert result["pipeline"]["train_policy_version"] == expected
        checkpoint = json.loads((output / f"checkpoint-{expected}" / "training_state.json").read_text())
        assert checkpoint["global_step"] == expected
        results.append({"mode": label, "updates": steps, "final_version": expected,
                        "command": command, "source_sha": result["source_sha"]})
    (args.output_dir / "result.json").write_text(json.dumps({"ok": True, "cases": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
