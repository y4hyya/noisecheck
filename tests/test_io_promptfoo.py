from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from noisecheck import LoadError, read_promptfoo

SAMPLE = Path(__file__).parent.parent / "examples" / "promptfoo-sample.json"


def write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "out.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def item(test_idx: int = 0, provider: Any = "modelA", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": provider,
        "testIdx": test_idx,
        "success": True,
        "score": 0.5,
        "latencyMs": 1000,
    }
    payload.update(overrides)
    return payload


class TestReadPromptfoo:
    def test_reads_the_committed_sample(self) -> None:
        ds = read_promptfoo(SAMPLE)
        assert ds.variants() == {"baseline", "candidate"}
        assert ds.metrics() == {"pass", "score", "grounding", "latency_ms"}
        assert len(ds) == 32
        row = next(
            r
            for r in ds.records
            if r.variant == "baseline" and r.example_id == "t1" and r.metric == "pass"
        )
        assert row.value == 0.0
        assert row.run_id is None

    def test_flat_results_list_accepted(self, tmp_path: Path) -> None:
        ds = read_promptfoo(write(tmp_path, {"results": [item()]}))
        assert ds.variants() == {"modelA"}

    def test_string_provider_becomes_the_variant(self, tmp_path: Path) -> None:
        ds = read_promptfoo(write(tmp_path, {"results": [item(provider="my model")]}))
        assert ds.variants() == {"my model"}

    def test_repeats_get_run_ids(self, tmp_path: Path) -> None:
        ds = read_promptfoo(write(tmp_path, {"results": [item(), item(), item(1)]}))
        runs = {r.run_id for r in ds.records if r.example_id == "t0"}
        assert runs == {"r0", "r1"}
        single = {r.run_id for r in ds.records if r.example_id == "t1"}
        assert single == {"r0"}

    def test_missing_provider_rejected(self, tmp_path: Path) -> None:
        broken = item()
        del broken["provider"]
        with pytest.raises(LoadError, match=r"results\[0\].*provider"):
            read_promptfoo(write(tmp_path, {"results": [broken]}))

    def test_missing_test_idx_rejected(self, tmp_path: Path) -> None:
        broken = item()
        del broken["testIdx"]
        with pytest.raises(LoadError, match=r"results\[0\].*testIdx"):
            read_promptfoo(write(tmp_path, {"results": [broken]}))

    def test_item_without_any_metric_rejected(self, tmp_path: Path) -> None:
        bare = {"provider": "m", "testIdx": 0}
        with pytest.raises(LoadError, match=r"results\[0\].*no metrics"):
            read_promptfoo(write(tmp_path, {"results": [bare]}))

    def test_other_json_shapes_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(LoadError, match="promptfoo"):
            read_promptfoo(write(tmp_path, {"hello": "world"}))

    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(LoadError):
            read_promptfoo(tmp_path / "missing.json")
