"""Extend the original holdout deterministically without overlapping training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def extend_holdout(train: list[dict], held_out: list[dict], *, size: int = 128, seed: int = 20260916) -> list[dict]:
    seen = {row["expression"] for row in train + held_out}
    if len(seen) != len(train) + len(held_out) or not len(held_out) <= size <= 1000:
        raise ValueError("require disjoint input rows and a holdout size between the original size and 1000")
    rows = list(held_out)
    rng = random.Random(seed)
    while len(rows) < size:
        a, b, c = (rng.randint(11, 89) for _ in range(3))
        kind = len(rows) % 3
        expression, answer = ((f"{a} * {b} - {c}", a * b - c),
                              (f"({a} + {b}) * {c}", (a + b) * c),
                              (f"{a} * {b} + {c}", a * b + c))[kind]
        if expression in seen:
            continue
        seen.add(expression)
        rows.append({"prompt": f"Calculate {expression}. Explain briefly and end with ANSWER: followed by the final integer.",
                     "answer": str(answer), "expression": expression})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = Path(__file__).resolve().parents[1]
    def read(name):
        return [json.loads(line) for line in (data / name).read_text().splitlines()]

    rows = extend_holdout(read("train.jsonl"), read("eval.jsonl"))
    with args.output.open("x") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
