"""Generates the paired versus unpaired case study data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    rng = np.random.default_rng(3831)
    n = 60
    difficulty = np.clip(rng.normal(0.55, 0.2, n), 0.05, 0.95)
    noise = rng.normal(0.0, 0.02, n)
    variants = [
        ("baseline", difficulty),
        ("candidate", difficulty + 0.03 + noise),
    ]
    for variant, values in variants:
        rows = [
            {
                "example_id": f"q{i:03d}",
                "variant": variant,
                "metric": "score",
                "value": round(float(value), 4),
            }
            for i, value in enumerate(values)
        ]
        path = Path(__file__).parents[2] / "examples" / f"case-{variant}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
