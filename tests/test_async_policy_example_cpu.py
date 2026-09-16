"""Reward and holdout integrity for the shipped async-policy example."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.async_policy.train import answer_matches, parse_args


@pytest.mark.parametrize(("completion", "answer", "expected"), [
    ("ANSWER: 2,087", "2087", True),
    ("Maybe 2087, but ANSWER: 2086", "2087", False),
    ("ANSWER: 2087.5", "2087", False),
    ("ANSWER: -42", "-42", True),
    ("I cannot answer.", "42", False),
])
def test_async_example_scores_final_number(completion, answer, expected):
    assert answer_matches(completion, answer) is expected


def test_async_example_holdout_is_disjoint():
    root = Path(__file__).resolve().parents[1] / "examples" / "async_policy"
    train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    held_out = [json.loads(line) for line in (root / "eval.jsonl").read_text().splitlines()]
    assert len(train) == 64 and len(held_out) == 32
    assert len({row["expression"] for row in train + held_out}) == 96


def test_expanded_holdout_preserves_original_subset_and_is_reproducible():
    from examples.async_policy.tools.make_quality_data import extend_holdout

    root = Path(__file__).resolve().parents[1] / "examples" / "async_policy"
    train, original, expanded = ([json.loads(line) for line in (root / name).read_text().splitlines()]
                                 for name in ("train.jsonl", "eval.jsonl", "eval128.jsonl"))
    assert len(expanded) == 128 and expanded[:32] == original
    assert len({row["expression"] for row in train + expanded}) == 192
    assert expanded == extend_holdout(train, original)


def test_example_default_parameters():
    args = parse_args(["--output-dir", "output"])
    assert (args.attn_backend, args.loss, args.lora_rank, args.lora_alpha) == ("native", "grpo", 8, 16)
    assert (args.learning_rate, args.n_samples, args.max_new_tokens) == (1e-5, 4, 64)
    assert (args.queue_capacity, args.max_inflight_rollouts, args.weight_sync_interval_updates) == (2, 1, 1)
    assert args.rollout_batch_groups == 1


def test_cli_preserves_model_reference_and_forwards_experiment_parameters(monkeypatch, tmp_path):
    import sys

    from examples.async_policy import train

    received = {}
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=lambda ref: str(tmp_path)))

    def run(**kwargs):
        received.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(train, "run_training", run)
    assert train.main(["--model", "Example/AnotherModel", "--output-dir", str(tmp_path / "output"),
                       "--attn-backend", "flash", "--lora-rank", "16", "--lora-alpha", "32",
                       "--learning-rate", "0.0002", "--n-samples", "8", "--max-new-tokens", "256",
                       "--loss", "grpo-offpolicy", "--queue-capacity", "4", "--max-inflight-rollouts", "2",
                       "--weight-sync-interval-updates", "4", "--rollout-batch-groups", "2"]) == 0
    assert received["base_model_reference"] == "Example/AnotherModel"
    assert received["model_path"] == str(tmp_path)
    assert (received["lora_rank"], received["lora_alpha"], received["learning_rate"]) == (16, 32, 0.0002)
    assert (received["attn_backend"], received["n_samples"], received["max_new_tokens"]) == ("flash", 8, 256)
    assert (received["queue_capacity"], received["max_inflight_rollouts"],
            received["weight_sync_interval_updates"]) == (4, 2, 4)
    assert received["loss"] == "grpo-offpolicy"
    assert received["rollout_batch_groups"] == 2


@pytest.mark.parametrize("arguments", [
    ["--max-new-tokens", "0"], ["--learning-rate", "nan"], ["--lag", "-1"], ["--lora-rank", "0"],
    ["--rollout-batch-groups", "2"],
    ["--mode", "sync", "--rollout-batch-groups", "2", "--max-inflight-rollouts", "2"],
])
def test_example_rejects_invalid_parameters_before_resolving_model(arguments):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "output", *arguments])
