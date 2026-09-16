"""Check deterministic gradients against FP64 references and CUDA graph replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def repeat(operation, repetitions=16):
    import torch

    first = tuple(value.detach().clone() for value in operation())
    maximum = 0.0
    exact = 0
    for _ in range(repetitions - 1):
        values = operation()
        exact += all(torch.equal(a, b) for a, b in zip(first, values, strict=True))
        for a, b in zip(first, values, strict=True):
            if a.numel():
                maximum = max(maximum, float((a.double() - b.detach().double()).abs().max()))
    return {"comparisons": repetitions - 1, "exact": exact, "max_abs_delta": maximum}


def graph_check(operation):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = operation()
    graph.replay()
    expected = tuple(value.clone() for value in outputs)
    for _ in range(3):
        graph.replay()
        assert all(torch.equal(a, b) for a, b in zip(expected, outputs, strict=True))
    return True


def embedding_case(dtype, rows, hidden):
    import torch
    import torch.nn.functional as F

    from areno.accel.embedding import areno_vocab_embedding

    ids = torch.randint(0, 48, (rows * 2,), device="cuda")[::2]
    weight = torch.randn((32, hidden * 2), device="cuda", dtype=dtype)[:, ::2].requires_grad_()
    grad = torch.randn((rows, hidden * 2), device="cuda", dtype=dtype)[:, ::2]
    reference_weight = weight.detach().cpu().double().requires_grad_()
    local_ids = ids.cpu() - 8
    reference = F.embedding(local_ids.clamp(0, 31), reference_weight)
    reference = (reference * ((local_ids >= 0) & (local_ids < 32)).unsqueeze(-1)).to(dtype)
    expected = torch.autograd.grad(reference, reference_weight, grad.cpu())[0].to(dtype)

    def operation():
        output = areno_vocab_embedding(ids, weight, 8, 40)
        return output, *torch.autograd.grad(output, weight, grad)

    output, gradient = operation()
    torch.testing.assert_close(output.cpu(), reference.detach(), rtol=0, atol=0)
    # FP16/BF16 references round once after FP64 accumulation. The CUDA path
    # accumulates in FP32; allow two rounding units, independently of equality.
    tolerance = {torch.float64: 1e-12, torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(gradient.cpu(), expected, rtol=tolerance, atol=tolerance)
    observed = repeat(operation)
    assert observed["exact"] == observed["comparisons"]
    return {"operator": "embedding", "dtype": str(dtype), "rows": rows, "hidden": hidden,
            "reference_passed": True, "repeatability": observed,
            "graph_replay": graph_check(operation) if rows else None}


def norm_case(dtype, rows, hidden, variant):
    import torch
    import torch.nn.functional as F

    from areno.accel.normalization import (
        areno_optional_scale_rmsnorm,
        areno_rmsnorm,
        areno_rmsnorm_silu_gate,
    )

    x = torch.randn((rows, hidden * 2), device="cuda", dtype=dtype)[:, ::2].requires_grad_()
    gate = torch.randn_like(x).requires_grad_()
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn((rows, hidden * 2), device="cuda", dtype=dtype)[:, ::2]
    cpu_x = x.detach().cpu().double().requires_grad_()
    cpu_gate = gate.detach().cpu().double().requires_grad_()
    cpu_weight = weight.detach().cpu().double().requires_grad_()
    reference = cpu_x * torch.rsqrt(cpu_x.square().mean(-1, keepdim=True) + 1e-6)
    if variant == "gated":
        reference = reference * F.silu(cpu_gate)
    if variant != "unscaled":
        reference = reference * cpu_weight
    reference = reference.to(dtype)
    inputs = (x, gate, weight) if variant == "gated" else (x,) if variant == "unscaled" else (x, weight)
    cpu_inputs = ((cpu_x, cpu_gate, cpu_weight) if variant == "gated"
                  else (cpu_x,) if variant == "unscaled" else (cpu_x, cpu_weight))
    expected = torch.autograd.grad(reference, cpu_inputs, grad.cpu())

    def operation():
        if variant == "gated":
            output = areno_rmsnorm_silu_gate(x, gate, weight, 1e-6)
        elif variant in {"optional", "unscaled"}:
            output = areno_optional_scale_rmsnorm(x, None if variant == "unscaled" else weight, 1e-6)
        else:
            output = areno_rmsnorm(x, weight, 1e-6)
        return output, *torch.autograd.grad(output, inputs, grad)

    output, *gradients = operation()
    tolerance = {torch.float32: 3e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(output.cpu(), reference.detach(), rtol=tolerance, atol=tolerance)
    for actual, target in zip(gradients, expected, strict=True):
        # Scale parameters are FP32 even for low precision activation tensors.
        bound = 3e-4 if actual.dtype == torch.float32 else tolerance
        torch.testing.assert_close(actual.cpu(), target.to(actual.dtype), rtol=bound, atol=bound)
        assert bool(torch.isfinite(actual).all())
    observed = repeat(operation)
    assert observed["exact"] == observed["comparisons"]
    return {"operator": variant, "dtype": str(dtype), "rows": rows, "hidden": hidden,
            "reference_passed": True, "repeatability": observed,
            "graph_replay": graph_check(operation) if rows else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnose-graphs", action="store_true")
    parser.add_argument("--graph-probe", choices=("sort", "torch-embedding", "areno-embedding"))
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    from examples.async_policy.tools.campaign_state import write_json

    torch.manual_seed(41)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.diagnose_graphs:
        probes = []
        for name in ("sort", "torch-embedding", "areno-embedding"):
            with (args.output_dir / f"graph-{name}.log").open("w") as log:
                process = subprocess.run([sys.executable, __file__, "--graph-probe", name,
                                          "--output-dir", str(args.output_dir)],
                                         stdout=log, stderr=subprocess.STDOUT, timeout=60)
            probes.append({"probe": name, "return_code": process.returncode})
        write_json(args.output_dir / "result.json", {"ok": all(row["return_code"] == 0 for row in probes),
                                                     "probes": probes})
        raise SystemExit(0 if all(row["return_code"] == 0 for row in probes) else 1)
    if args.graph_probe:
        ids = torch.randint(0, 32, (34,), device="cuda")[::2]
        weight = torch.randn((32, 66), device="cuda")[:, ::2].requires_grad_()
        grad = torch.randn((17, 66), device="cuda")[:, ::2]

        def operation():
            if args.graph_probe == "sort":
                return torch.sort(ids, stable=True)
            if args.graph_probe == "torch-embedding":
                output = torch.nn.functional.embedding(ids, weight)
            else:
                from areno.accel.embedding import areno_vocab_embedding

                output = areno_vocab_embedding(ids, weight, 0, 32)
            return output, *torch.autograd.grad(output, weight, grad)

        repeat(operation)
        graph_check(operation)
        return
    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        for shape in ((0, 33), (17, 33), (513, 128)):
            rows.append(embedding_case(dtype, *shape))
            write_json(args.output_dir / "progress.json", rows)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for variant in ("rmsnorm", "optional", "unscaled", "gated"):
            for shape in ((0, 128), (1, 1), (17, 33), (257, 128), (1025, 1024)):
                rows.append(norm_case(dtype, *shape, variant))
                write_json(args.output_dir / "progress.json", rows)
    result = {"ok": True, "cases": rows, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "deterministic": torch.are_deterministic_algorithms_enabled()}
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"ok": True, "cases": len(rows)}))


if __name__ == "__main__":
    main()
