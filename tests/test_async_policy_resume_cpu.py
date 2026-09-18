"""Resume evidence must distinguish exact restoration from small training drift."""

import torch

from examples.async_policy.tools.resume_run import compare_tensors, state_fingerprints


def test_resume_fingerprint_preserves_bits_shapes_and_optimizer_metadata():
    value = {"master": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
             "step": torch.tensor(3), "betas": (0.9, 0.95)}
    fingerprint = state_fingerprints(value)
    assert fingerprint == state_fingerprints({**value, "master": value["master"].clone()})
    assert fingerprint != state_fingerprints({**value, "step": torch.tensor(4)})
    assert fingerprint != state_fingerprints({**value, "master": value["master"].view(1, 2)})
    assert fingerprint != state_fingerprints({**value, "master": value["master"].float()})


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
