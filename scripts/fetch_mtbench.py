"""Fetches a small MT Bench sample joining human and gpt4 pairwise judgments.

Source: lmsys/mt_bench_human_judgments on Hugging Face, licensed CC BY 4.0.
https://huggingface.co/datasets/lmsys/mt_bench_human_judgments
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

import certifi

BASE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=lmsys%2Fmt_bench_human_judgments&config=default"
)
WINNER_VALUE = {"model_a": 0.0, "tie": 0.5, "model_b": 1.0}
Key = tuple[int, str, str, int]


def fetch(split: str, offset: int, length: int) -> list[dict[str, object]]:
    url = f"{BASE}&split={split}&offset={offset}&length={length}"
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context) as response:
        payload = json.load(response)
    return [row["row"] for row in payload["rows"]]


def main() -> None:
    judgments: dict[str, dict[Key, float]] = {"human": {}, "gpt4_pair": {}}
    for split, store in judgments.items():
        for offset in (0, 100, 200):
            for row in fetch(split, offset, 100):
                winner = str(row["winner"])
                if winner not in WINNER_VALUE:
                    continue
                key = (
                    int(str(row["question_id"])),
                    str(row["model_a"]),
                    str(row["model_b"]),
                    int(str(row["turn"])),
                )
                store.setdefault(key, WINNER_VALUE[winner])

    human = judgments["human"]
    gpt4 = judgments["gpt4_pair"]
    keys = sorted(key for key in human if key in gpt4)[:80]
    records = []
    for qid, model_a, model_b, turn in keys:
        example = f"q{qid}_{model_a}_{model_b}_t{turn}"
        for variant, store in [("human", human), ("gpt4", gpt4)]:
            records.append(
                {
                    "example_id": example,
                    "variant": variant,
                    "metric": "winner",
                    "value": store[(qid, model_a, model_b, turn)],
                    "meta": {"model_a": model_a, "model_b": model_b},
                }
            )
    target = Path(__file__).parents[1] / "examples" / "mtbench-sample.jsonl"
    target.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} records from {len(keys)} joined battles to {target}")


if __name__ == "__main__":
    main()
