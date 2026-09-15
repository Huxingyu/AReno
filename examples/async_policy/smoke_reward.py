"""Toy completion reward for runtime tests; not a task-quality benchmark."""

from __future__ import annotations

import re


def reward_fn(record) -> float:
    answer = str(record.answer if record.answer is not None else record.source_record.get("answer", ""))
    correct = bool(answer and re.search(r"(?<!\d)" + re.escape(answer) + r"(?!\d)", record.completion))
    response = [token for token, mask in zip(record.tokens, record.loss_mask, strict=True) if mask]
    # A small token-diversity term exercises nonzero gradients even when all
    # samples have the same binary correctness reward.
    diversity = len(set(response)) / max(len(response), 1)
    return float(correct) + 0.01 * diversity
