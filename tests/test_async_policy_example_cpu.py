"""Reward and holdout integrity for the shipped async-policy example."""

import json
from pathlib import Path

import pytest

from examples.async_policy.train import answer_matches


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
