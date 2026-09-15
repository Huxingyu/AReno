"""Run the existing synchronous GRPO/GSPO SDK independently in two checkouts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def run_checkout(args) -> None:
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    import torch

    import areno
    from areno import Trainer
    from areno.adapters.config import LoraConfig
    from areno.api.algorithms import get_algorithm
    from areno.api.config import CudaConfig
    from areno.api.models import SamplingParams
    from areno.api.tokenizer import configure_chat_template_enable_thinking
    from areno.api.trainers.policy_only import PolicyOnlyTrainer

    assert Path(areno.__file__).resolve().parents[1] == root
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    torch.manual_seed(41)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.data_path.read_text().splitlines()][:3]
    config = CudaConfig(devices=[0], tp_size=1, dp_size=1, max_running_prompts=4,
                        lora=LoraConfig(rank=8, alpha=16), optimizer={"lr": 1e-5},
                        runtime={"compile_model": False, "eager_decode": True, "attn_backend": "flash"})
    trainer = Trainer(world_size=1, model_path=args.model_path, custom_config=config)

    def reward(record):
        tokens = [token for token, selected in zip(record.tokens, record.loss_mask, strict=True) if selected]
        return len(set(tokens)) / max(len(tokens), 1)

    materializer = SimpleNamespace(reward_fn=reward)
    materializer._score_reward_records = types.MethodType(PolicyOnlyTrainer._score_reward_records, materializer)
    results = []
    try:
        trainer.init()
        trainer.export_adapter(str(args.output_dir / "initial"))
        tokenizer = trainer.get_tokenizer()
        configure_chat_template_enable_thinking(tokenizer, False)
        for index, batch in enumerate(trainer.load_prompt_batches(rows, batch_size=1, max_prompt_tokens=128,
                                                                 solutions_key="answer")):
            sampling = SamplingParams(max_prompt_len=128, max_new_tokens=64, seed=41 + index)
            trainer.begin_rollout_session()
            try:
                rollout = trainer.rollout_token_batch([item.input_tokens for item in batch.items], 4, sampling)
            finally:
                trainer.end_rollout_session()
            sequences, _, _ = PolicyOnlyTrainer._materialize_train_batch(materializer, tokenizer, batch, rollout)
            metrics = trainer.train(sequences, get_algorithm(args.algorithm).default_loss_fn,
                                    mini_bs=1, gradient_accumulation_steps=4)
            selected = {key: value for key, value in metrics.items()
                        if key in {"loss", "grad_norm", "grad_norm_unclipped", "adapter_version"}}
            assert all(math.isfinite(value) for value in selected.values())
            assert selected["adapter_version"] == index + 1
            results.append({"rollout": [result.model_dump() for result in rollout],
                            "sequences": [sequence.model_dump() for sequence in sequences], "metrics": selected})
        trainer.export_adapter(str(args.output_dir / "final"))
    finally:
        trainer.close()
    assert len(results) == 3 and any(row["metrics"].get("grad_norm", 0) > 0 for row in results)
    write(args.output_dir / "result.json", {
        "ok": True, "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "import_path": areno.__file__, "algorithm": args.algorithm, "rows": results,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--algorithm", choices=("grpo", "gspo"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repo_root:
        run_checkout(args)
        return 0
    if args.baseline_root is None:
        parser.error("--baseline-root is required for the comparison suite")
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for algorithm in ("grpo", "gspo"):
        outputs = []
        for label, checkout in (("baseline", args.baseline_root), ("candidate", root)):
            output = args.output_dir / algorithm / label
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--repo-root", str(checkout),
                       "--algorithm", algorithm, "--model-path", args.model_path,
                       "--data-path", str(args.data_path), "--output-dir", str(output)]
            with (output / "console.log").open("w") as log:
                subprocess.run(command, cwd=checkout, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
            outputs.append(output)
        before, after = [json.loads((output / "result.json").read_text()) for output in outputs]
        # The unchanged SDK runs on identical hardware and inputs. Require
        # bitwise equality here; asynchronous trajectories are not compared.
        assert before["rows"] == after["rows"], f"{algorithm}: synchronous SDK results changed"
        import torch
        from safetensors.torch import load_file

        baseline = load_file(str(outputs[0] / "final" / "adapter_model.safetensors"))
        candidate = load_file(str(outputs[1] / "final" / "adapter_model.safetensors"))
        initial = load_file(str(outputs[1] / "initial" / "adapter_model.safetensors"))
        assert baseline.keys() == candidate.keys() == initial.keys()
        assert all(torch.equal(value, candidate[key]) for key, value in baseline.items())
        assert any(not torch.equal(value, initial[key]) for key, value in candidate.items())
        comparisons.append({"algorithm": algorithm, "updates_per_checkout": 3,
                            "equal_tensors": len(candidate), "max_abs_error": 0.0,
                            "baseline_sha": before["source_sha"], "candidate_sha": after["source_sha"]})
    write(args.output_dir / "result.json", {"ok": True, "comparisons": comparisons})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
