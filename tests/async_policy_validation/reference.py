"""Compare synchronous baseline and async materialization through a real CPU step."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("sync", "async"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    from tiny_backend import engines, token_reward

    import areno
    from areno.api.trainers.policy_only import PolicyOnlyTrainer

    repo_root = args.repo_root.resolve()
    assert Path(areno.__file__).resolve().parents[1] == repo_root, "reference imported a different checkout"

    rollout, train, _, _ = engines()
    items = [SimpleNamespace(prompt_id=str(i), input_tokens=[1, i + 3, 4], prompt=f"prompt {i}",
                             record={}, solutions=None) for i in range(4)]
    results = [rollout.generate(item, 0) for item in items]

    class Tokenizer:
        eos_token_id = 2

        def batch_decode(self, rows):
            return [" ".join(str(token) for token in row) for row in rows]

    if args.mode == "sync":
        stub = SimpleNamespace(reward_fn=lambda record: token_reward([int(t) for t in record.completion.split()]))
        stub._score_reward_records = types.MethodType(PolicyOnlyTrainer._score_reward_records, stub)
        rows, _, _ = PolicyOnlyTrainer._materialize_train_batch(stub, Tokenizer(), SimpleNamespace(items=items), results)
        batch = SimpleNamespace(batch_id="reference", policy_version=0, sequences=rows)
    else:
        from areno.experimental.async_policy import build_batch_envelope
        batch = build_batch_envelope(run_id="reference", batch_id="reference", epoch=0, policy_version=0,
                                    prompt_items=items, rollout_results=results,
                                    rewards=[token_reward(seq.resp_tokens) for result in results for seq in result.sequences],
                                    eos_token_id=2)
    assert train.train(batch, 0)
    arrays = train.last_arrays
    arrays["rewards"] = [row.reward for row in batch.sequences]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(arrays, indent=2) + "\n")
    metadata = {
        "mode": args.mode, "repo_root": str(repo_root), "python": sys.version,
        "areno_file": areno.__file__, "materializer_file": inspect.getfile(PolicyOnlyTrainer),
        "commit": subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "status", "--porcelain"], cwd=repo_root, text=True,
        ),
        "sequences": len(batch.sequences),
    }
    if args.mode == "async":
        metadata["materializer_file"] = inspect.getfile(build_batch_envelope)
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.compare:
        import numpy as np
        reference = json.loads(args.compare.read_text())
        differences = {}
        assert arrays.keys() == reference.keys()
        for key, value in arrays.items():
            actual = np.asarray(value, dtype=float)
            expected = np.asarray(reference[key], dtype=float)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, err_msg=key)
            differences[key] = float(np.max(np.abs(actual - expected)))
        comparison = {"status": "pass", "atol": 1e-7, "rtol": 1e-6, "max_absolute_errors": differences}
        (args.output.parent / "reference-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2))
    else:
        print(json.dumps({"status": "pass", "mode": args.mode, "sequences": len(batch.sequences)}))


if __name__ == "__main__":
    main()
