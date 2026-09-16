"""Check exact restoration and distinguish continuation drift from native repeatability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def state_fingerprints(value):
    """Replace tensors with exact, typed hashes without a second byte copy."""
    import torch

    if isinstance(value, torch.Tensor):
        data = value.detach().reshape(-1).contiguous().cpu().view(torch.uint8)
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(memoryview(data.numpy())).hexdigest()}
    if isinstance(value, dict):
        return {key: state_fingerprints(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [state_fingerprints(item) for item in value]
    return value


def optimizer_fingerprints(optimizer):
    """Hash the canonical FP32 state one bucket at a time in this audit tool.

    The native state_dict materializes all CPU master parameters and moments
    together. Use its exact bucket serialization helpers without retaining the
    complete multi-GB snapshot. Other optimizer formats keep their own contract.
    """
    from areno.engine.optim.adamw_fp32_master import AdamWFP32Master

    if type(optimizer) is not AdamWFP32Master:
        return state_fingerprints(optimizer.state_dict())
    result = state_fingerprints({key: getattr(optimizer, key) for key in (
        "lr", "betas", "weight_decay", "eps", "dp_rank", "dp_size")})
    result.update(master_params=[], state=[])
    for index, bucket in enumerate(optimizer.buckets):
        payload = optimizer._bucket_cpu_payload(index, bucket)
        result["master_params"].append(state_fingerprints(
            optimizer._materialize_master_cpu(bucket, payload["master_storage"])))
        result["state"].append(state_fingerprints({
            "exp_avg": payload["exp_avg"], "exp_avg_sq": payload["exp_avg_sq"], "step": bucket.step,
        }))
        del payload
    return result


def compare_tensors(left, right) -> dict:
    """Measure all differences in bounded chunks, including sparse BF16 drift."""
    import torch

    rows = []
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        row = {"name": key, "shape": list(a.shape), "dtype": str(a.dtype),
               "layout_equal": a.shape == b.shape and a.dtype == b.dtype,
               "different_elements": 0, "max_abs_error": 0.0, "finite": True}
        if row["layout_equal"]:
            a, b = a.reshape(-1), b.reshape(-1)
            for offset in range(0, a.numel(), 1024**2):
                x, y = a[offset:offset + 1024**2], b[offset:offset + 1024**2]
                row["different_elements"] += int(torch.count_nonzero(x != y))
                row["finite"] &= bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                row["max_abs_error"] = max(row["max_abs_error"], float((x.float() - y.float()).abs().max()))
        rows.append(row)
    keys_equal = left.keys() == right.keys()
    return {"exact": bool(left) and keys_equal and all(row["layout_equal"] and row["finite"]
                                                     and row["different_elements"] == 0 for row in rows),
            "keys_equal": keys_equal, "compared_tensors": len(rows),
            "different_tensors": sum(not row["layout_equal"] or row["different_elements"] > 0 for row in rows),
            "max_abs_error": max((row["max_abs_error"] for row in rows), default=0.0), "tensors": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-parameters", action="store_true")
    parser.add_argument("--attn-backend", choices=("native", "flash"), default="native")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic PyTorch algorithms and AReno reduction kernels")
    parser.add_argument("--audit-each-step", action="store_true")
    args = parser.parse_args()
    os.environ.update(ARENO_RESUME_SEED=str(args.seed),
                      ARENO_RESUME_DETERMINISTIC="1" if args.deterministic else "0",
                      ARENO_RESUME_AUDIT_EACH_STEP="1" if args.audit_each_step else "0")
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    from gpu_run import adapter_tensors, make_inputs
    from gpu_worker import ResumeAuditWorker

    from areno.adapters.config import LoraConfig
    from areno.api.config import CudaConfig
    from areno.api.models import RolloutResult, SamplingParams
    from areno.experimental.async_policy.bridge import DualEngineBridge
    from areno.experimental.async_policy.contracts import AsyncPolicyConfig, DeviceMode
    from areno.experimental.async_policy.coordination import PolicyPipelineCoordinator
    from areno.experimental.async_policy.data import build_batch_envelope
    from areno.experimental.async_policy.native import NativeCudaEnginePair

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, prompts = make_inputs(args.model_path)
    rollout = RolloutResult.model_validate(json.loads((ROOT / "tests/async_policy_validation/sdk_train_fixture.json").read_text())["rollouts"][0][0])
    batch = build_batch_envelope(run_id="resume", batch_id="fixed", epoch=0, policy_version=0,
                                 prompt_items=[prompts[0]], rollout_results=[rollout], rewards=[0.0, 0.25, 0.5, 1.0],
                                 eos_token_id=tokenizer.eos_token_id)
    config = CudaConfig(devices=[0], rollout_devices=[1], tp_size=1, dp_size=1, rollout_tp_size=1,
                        max_running_prompts=4, lora=None if args.full_parameters else LoraConfig(rank=8, alpha=16),
                        optimizer={"lr": 1e-5},
                        runtime={"compile_model": False, "eager_decode": True, "attn_backend": args.attn_backend})
    results = []
    cases = [("continuous", None), ("resumed", str(args.output_dir / "continuous" / "checkpoint-1"))]
    if args.full_parameters:
        cases.append(("control", None))
    for label, resume in cases:
        output = args.output_dir / label
        output.mkdir(parents=True, exist_ok=True)
        os.environ.update(ARENO_R4_OUTPUT=str(output), ARENO_R4_PROFILE="0", ARENO_R4_FAULT="")
        pair = NativeCudaEnginePair(model_path=args.model_path, config=config, tokenizer=tokenizer,
                                   sampling_params=SamplingParams(max_new_tokens=64, max_prompt_len=128),
                                   worker_cls=ResumeAuditWorker, resume_from=resume, seed=args.seed)

        class SavingTrain:
            def initialize(self, *, timeout_s):
                pair.initialize(timeout_s=timeout_s)

            def train(self, value, version, *, timeout_s):
                from areno.experimental.async_policy.coordination import Deadline

                deadline = Deadline(timeout_s)
                stepped = pair.train(value, version, timeout_s=deadline.remaining())
                if stepped and version == 0 and label == "continuous":
                    pair.save_training_checkpoint(str(output / "checkpoint-1"), timeout_s=deadline.remaining())
                if stepped and version == 2:
                    # The worker hashes final optimizer state directly. Saving
                    # two more full optimizer files would add about 18 GB.
                    pair.save_checkpoint(str(output / "checkpoint-3"), timeout_s=deadline.remaining())
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
        assert pair.train_stats and all(math.isfinite(row["loss"]) for row in pair.train_stats)
        assert any(row.get("metrics", {}).get("grad_norm", 0) > 0 for row in pair.train_stats)
        results.append({"mode": label, "initial_version": pair.initial_policy_version,
                        "final_version": coordinator.train_policy_version, "worker_exits": pair.worker_exits,
                        "train_stats": pair.train_stats})
        (args.output_dir / "progress.json").write_text(json.dumps(results, indent=2) + "\n")

    def audit(label, operation, step):
        return json.loads((args.output_dir / label / f"resume-audit-{operation}-{step}.json").read_text())

    saved = audit("continuous", "save_checkpoint", 1)
    restored = audit("resumed", "load_training_state", 1)
    boundary = {key: saved[key] == restored[key] for key in saved}
    left = adapter_tensors(args.output_dir / "continuous" / "checkpoint-3")
    right = adapter_tensors(args.output_dir / "resumed" / "checkpoint-3")
    comparison = compare_tensors(left, right)
    del right
    final = [audit(label, "save_checkpoint", 3) for label in ("continuous", "resumed")]
    optimizer_equal = final[0]["optimizer"] == final[1]["optimizer"]
    rng_equal = all(final[0][key] == final[1][key] for key in ("cpu_rng", "cuda_rng"))
    result = {"ok": all(boundary.values()) and comparison["exact"] and optimizer_equal and rng_equal,
              "seed": args.seed, "deterministic": args.deterministic,
              "full_parameters": args.full_parameters, "restore_boundary": boundary,
              "restore_exact": all(boundary.values()), "continuation": comparison,
              "optimizer_state_equal": optimizer_equal, "rng_state_equal": rng_equal, "cases": results}
    if args.full_parameters:
        control = adapter_tensors(args.output_dir / "control" / "checkpoint-3")
        control_final = audit("control", "save_checkpoint", 3)
        result["independent_control"] = {
            "initial_state_equal": audit("continuous", "before_train", 0) == audit("control", "before_train", 0),
            "comparison": compare_tensors(left, control),
            "optimizer_state_equal": final[0]["optimizer"] == control_final["optimizer"],
        }
        result["ok"] &= result["independent_control"]["initial_state_equal"]
        if args.deterministic:
            result["ok"] &= (result["independent_control"]["comparison"]["exact"]
                             and result["independent_control"]["optimizer_state_equal"])
    if args.audit_each_step:
        result["step_audits"] = {}
        for step in (1, 2, 3):
            reference = audit("continuous", "after_train", step)
            labels = (["control"] if args.full_parameters else []) + (["resumed"] if step > 1 else [])
            result["step_audits"][str(step)] = {
                label: {key: reference[key] == audit(label, "after_train", step)[key] for key in reference}
                for label in labels}
        if args.deterministic:
            result["ok"] &= all(all(flags.values()) for step in result["step_audits"].values()
                                for flags in step.values())
    # Keep exact continuation as a separate, strict gate even when the native
    # continuous control also differs. No tolerance is chosen to hide drift.
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
