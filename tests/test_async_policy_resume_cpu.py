"""Resume evidence must distinguish exact restoration from small training drift."""

import weakref

import torch

from examples.async_policy.tools.resume_run import compare_tensors, optimizer_fingerprints, state_fingerprints


def test_resume_fingerprint_preserves_bits_shapes_and_optimizer_metadata():
    value = {"master": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
             "step": torch.tensor(3), "betas": (0.9, 0.95)}
    fingerprint = state_fingerprints(value)
    assert fingerprint == state_fingerprints({**value, "master": value["master"].clone()})
    assert fingerprint != state_fingerprints({**value, "step": torch.tensor(4)})
    assert fingerprint != state_fingerprints({**value, "master": value["master"].view(1, 2)})
    assert fingerprint != state_fingerprints({**value, "master": value["master"].float()})


def test_streamed_optimizer_audit_matches_public_state_without_retaining_buckets(monkeypatch):
    from areno.engine.optim.adamw_fp32_master import AdamWFP32Master

    parameter = torch.nn.Parameter(torch.arange(1, 22, dtype=torch.bfloat16))
    optimizer = AdamWFP32Master([parameter], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01, bucket_numel=7)
    original_state = optimizer.state_dict
    original_bucket = optimizer._bucket_cpu_payload
    pending = []

    def one_bucket(index, bucket):
        assert not any(reference() is not None for reference in pending)
        payload = original_bucket(index, bucket)
        pending[:] = [weakref.ref(payload[key]) for key in ("exp_avg", "exp_avg_sq")
                      if payload[key] is not None]
        return payload

    def forbidden_snapshot():
        raise AssertionError("audit materialized the complete optimizer")

    monkeypatch.setattr(optimizer, "state_dict", forbidden_snapshot)
    monkeypatch.setattr(optimizer, "_bucket_cpu_payload", one_bucket)
    for _ in range(2):
        # Use the unmodified public serializer for the independent reference.
        monkeypatch.setattr(optimizer, "_bucket_cpu_payload", original_bucket)
        expected = state_fingerprints(original_state())
        monkeypatch.setattr(optimizer, "_bucket_cpu_payload", one_bucket)
        assert optimizer_fingerprints(optimizer) == expected
        assert not any(reference() is not None for reference in pending)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()


def test_resume_comparison_detects_one_element_after_a_chunk_boundary():
    left = {"weight": torch.zeros(1024**2 + 1, dtype=torch.bfloat16)}
    right = {"weight": left["weight"].clone()}
    right["weight"][-1] = 2**-20
    comparison = compare_tensors(left, right)
    assert not comparison["exact"]
    assert comparison["different_tensors"] == 1
    assert comparison["tensors"][0]["different_elements"] == 1
    assert comparison["max_abs_error"] == 2**-20
    assert compare_tensors(left, left)["exact"]


def test_resume_comparison_rejects_missing_keys_layout_changes_and_nonfinite_values():
    left = {"weight": torch.tensor([1.0])}
    assert not compare_tensors(left, {})["exact"]
    assert not compare_tensors(left, {"weight": torch.tensor([[1.0]])})["exact"]
    assert not compare_tensors(left, {"weight": torch.tensor([1.0], dtype=torch.bfloat16)})["exact"]
    invalid = {"weight": torch.tensor([float("nan")])}
    assert not compare_tensors(invalid, invalid)["exact"]
