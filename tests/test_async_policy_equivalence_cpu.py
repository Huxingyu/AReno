"""T02 equivalence: async envelope vs the synchronous materializer.

The completion condition is "under a fixed rollout input, the async sample
construction matches the synchronous result". This test drives the real
`PolicyOnlyTrainer._materialize_train_batch` with a fixed rollout input and a
fixed reward rule, then builds the async envelope from the same input and
compares every TrainSequence field. Because it calls the production method
instead of a copy, drift in the sync path fails this test.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from areno.api.models import RolloutResult, RolloutSequence
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.experimental.async_policy import build_batch_envelope


class _FakeTokenizer:
    """Deterministic tokenizer surface used by the sync materializer."""

    eos_token_id = 2

    def batch_decode(self, token_lists):
        return [" ".join(str(token) for token in tokens) for tokens in token_lists]


def _reward_rule(sample_index: int) -> float:
    # A nontrivial per-sample rule so group advantages are non-degenerate.
    return float(sample_index) * 0.5 - 0.25


def _sync_reward(record) -> float:
    return _reward_rule(int(record.metadata["sample_index"]))


def _item(prompt_id: str, input_tokens: list[int], features=None):
    return SimpleNamespace(
        prompt_id=prompt_id,
        input_tokens=input_tokens,
        prompt=f"prompt-{prompt_id}",
        record={"features": features},
        solutions=[str(input_tokens)],
    )


def _result(*sequences):
    return RolloutResult(
        sequences=[RolloutSequence(resp_tokens=tokens, resp_logprobs=logprobs) for tokens, logprobs in sequences]
    )


def test_async_envelope_matches_sync_materialize() -> None:
    items = [
        _item("p0", [10, 11], features={"tag": "a"}),
        _item("p1", [20, 21, 22]),
    ]
    results = [
        _result(([1, 2], [-0.1, -0.2]), ([3], [-0.3])),
        _result(([4, 5], [-0.4, -0.5]), ([6], [-0.6]), ([7, 8, 9], [-0.7, -0.8, -0.9])),
    ]
    prompt_batch = SimpleNamespace(items=items)

    # --- synchronous production path -------------------------------------
    stub = SimpleNamespace(reward_fn=_sync_reward)
    stub._score_reward_records = types.MethodType(PolicyOnlyTrainer._score_reward_records, stub)
    sync_rows, sync_rewards, _stats = PolicyOnlyTrainer._materialize_train_batch(
        stub, _FakeTokenizer(), prompt_batch, results
    )

    # --- async production path, fed the same per-sample rewards -----------
    expected_rewards = [_reward_rule(sample) for sample in (0, 1, 0, 1, 2)]
    assert [row.reward for row in sync_rows] == pytest.approx(expected_rewards)
    envelope = build_batch_envelope(
        run_id="run",
        batch_id="b0",
        epoch=0,
        policy_version=7,
        prompt_items=items,
        rollout_results=results,
        rewards=expected_rewards,
        eos_token_id=_FakeTokenizer.eos_token_id,
        produced_at=0.0,
    )

    assert list(sync_rewards) == pytest.approx(list(envelope.rewards))
    assert len(sync_rows) == len(envelope.sequences) == 5
    assert envelope.group_offsets == (0, 2, 5)

    for sync_row, async_row in zip(sync_rows, envelope.sequences, strict=True):
        assert list(async_row.tokens) == list(sync_row.tokens)
        assert list(async_row.logprobs) == pytest.approx(list(sync_row.logprobs))
        assert async_row.prompt_len == sync_row.prompt_len
        assert async_row.reward == pytest.approx(sync_row.reward)
        assert async_row.scalar_advantage == pytest.approx(sync_row.scalar_advantage)
        assert async_row.eos_token_id == sync_row.eos_token_id
        assert async_row.features == sync_row.features

    # prompt groups stay intact and in order.
    assert envelope.prompt_ids == ("p0", "p1")
    assert envelope.group_slice(0)[0].features == {"tag": "a"}
    assert envelope.group_slice(1)[0].tokens[0] == 20
