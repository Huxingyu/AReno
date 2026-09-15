"""Prove that optimizer/RNG checkpoint continuation matches uninterrupted CUDA training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch
    from gpu_run import adapter_tensors, make_inputs
    from gpu_worker import ObservedWorker

    from areno.adapters.config import LoraConfig
    from areno.api.config import CudaConfig
    from areno.api.models import RolloutResult, SamplingParams
    from areno.experimental.async_policy.bridge import DualEngineBridge
    from areno.experimental.async_policy.contracts import AsyncPolicyConfig, DeviceMode
    from areno.experimental.async_policy.coordination import PolicyPipelineCoordinator
    from areno.experimental.async_policy.data import build_batch_envelope
    from areno.experimental.async_policy.native import NativeCudaEnginePair

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, prompts = make_inputs(args.model_path)
    rollout = RolloutResult.model_validate(json.loads(Path(__file__).with_name("sdk_train_fixture.json").read_text())["rollouts"][0][0])
    batch = build_batch_envelope(run_id="resume", batch_id="fixed", epoch=0, policy_version=0,
                                 prompt_items=[prompts[0]], rollout_results=[rollout], rewards=[0.0, 0.25, 0.5, 1.0],
                                 eos_token_id=tokenizer.eos_token_id)
    config = CudaConfig(devices=[0], rollout_devices=[1], tp_size=1, dp_size=1, rollout_tp_size=1,
                        max_running_prompts=4, lora=LoraConfig(rank=8, alpha=16), optimizer={"lr": 1e-5},
                        runtime={"compile_model": False, "eager_decode": True, "attn_backend": "flash"})
    results = []
    for label, resume in (("continuous", None), ("resumed", str(args.output_dir / "continuous" / "checkpoint-1"))):
        output = args.output_dir / label
        output.mkdir(parents=True, exist_ok=True)
        os.environ.update(ARENO_R4_OUTPUT=str(output), ARENO_R4_PROFILE="0", ARENO_R4_FAULT="")
        pair = NativeCudaEnginePair(model_path=args.model_path, config=config, tokenizer=tokenizer,
                                   sampling_params=SamplingParams(max_new_tokens=64, max_prompt_len=128),
                                   worker_cls=ObservedWorker, resume_from=resume)

        class SavingTrain:
            def initialize(self, *, timeout_s):
                pair.initialize(timeout_s=timeout_s)

            def train(self, value, version, *, timeout_s):
                from areno.experimental.async_policy.coordination import Deadline

                deadline = Deadline(timeout_s)
                stepped = pair.train(value, version, timeout_s=deadline.remaining())
                if stepped and version in {0, 2}:
                    pair.save_training_checkpoint(str(output / f"checkpoint-{version + 1}"),
                                                  timeout_s=deadline.remaining())
                return stepped

            def close(self, *, timeout_s):
                pair.close(timeout_s=timeout_s)

        coordinator = PolicyPipelineCoordinator(train_policy_version=pair.initial_policy_version)
        bridge = DualEngineBridge(train_engine=SavingTrain(), rollout_engine=pair.rollout_engine,
                                  weight_sync=pair.weight_sync, coordinator=coordinator, mode=DeviceMode.SYNC,
                                  config=AsyncPolicyConfig(max_policy_lag=4, operation_timeout_s=300, shutdown_timeout_s=30))
        try:
            bridge.initialize()
            for version in range(pair.initial_policy_version, 3):
                result = bridge.train(batch)
                assert result.stepped and result.train_version_after == version + 1
                bridge.sync()
        finally:
            bridge.close()
        assert pair.worker_exits and not any(row["alive"] for row in pair.worker_exits)
        results.append({"mode": label, "initial_version": pair.initial_policy_version,
                        "final_version": coordinator.train_policy_version, "worker_exits": pair.worker_exits})
    left = adapter_tensors(args.output_dir / "continuous" / "checkpoint-3")
    right = adapter_tensors(args.output_dir / "resumed" / "checkpoint-3")
    assert left.keys() == right.keys() and all(torch.equal(value, right[key]) for key, value in left.items())
    states = [torch.load(args.output_dir / label / "checkpoint-3" / "training_state.pt", weights_only=True)
              for label in ("continuous", "resumed")]

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(value, b[key]) for key, value in a.items())
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    assert equal(states[0]["optimizer"], states[1]["optimizer"]), "optimizer state diverged after continuation"
    result = {"ok": True, "equal_tensors": len(left), "max_abs_error": 0.0,
              "optimizer_state_equal": True, "cases": results}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
