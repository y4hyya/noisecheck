from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from noisecheck.cli import app

runner = CliRunner()
EXAMPLES = Path(__file__).parent.parent / "examples"
BASELINE = str(EXAMPLES / "baseline.jsonl")
CANDIDATE = str(EXAMPLES / "candidate.jsonl")
PROMPTFOO = str(EXAMPLES / "promptfoo-sample.json")


class TestCompare:
    def test_prints_a_verdict_table_and_passes(self) -> None:
        result = runner.invoke(app, ["compare", BASELINE, CANDIDATE])
        assert result.exit_code == 0
        assert "task_success" in result.output
        assert "latency_s" in result.output
        assert "gate non regression" in result.output

    def test_lower_is_better_turns_latency_into_a_regression(self) -> None:
        result = runner.invoke(
            app, ["compare", BASELINE, CANDIDATE, "--lower-is-better", "latency_s"]
        )
        assert result.exit_code == 1
        assert "regression" in result.output

    def test_cluster_flag_reads_the_meta_key(self) -> None:
        result = runner.invoke(
            app, ["compare", BASELINE, CANDIDATE, "--cluster", "conversation_id"]
        )
        assert result.exit_code == 0
        assert "only 8 clusters" in result.output
        assert "underpowered" in result.output

    def test_writes_json_and_markdown_reports(self, tmp_path: Path) -> None:
        json_path = tmp_path / "report.json"
        md_path = tmp_path / "report.md"
        result = runner.invoke(
            app,
            [
                "compare",
                BASELINE,
                CANDIDATE,
                "--json",
                str(json_path),
                "--md",
                str(md_path),
            ],
        )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["assessment"]["exit_code"] == result.exit_code
        assert payload["info"]["package_version"]
        assert "| task_success |" in md_path.read_text(encoding="utf-8")

    def test_metric_filter(self) -> None:
        result = runner.invoke(app, ["compare", BASELINE, CANDIDATE, "--metric", "task_success"])
        assert result.exit_code == 0
        assert "latency_s" not in result.output

    def test_missing_file_is_a_user_error(self) -> None:
        result = runner.invoke(app, ["compare", "missing.jsonl", CANDIDATE])
        assert result.exit_code == 3
        assert "error" in result.output

    def test_multi_variant_file_is_rejected(self) -> None:
        result = runner.invoke(app, ["compare", PROMPTFOO, CANDIDATE])
        assert result.exit_code == 3
        assert "variant" in result.output

    def test_unknown_cluster_key_is_a_user_error(self) -> None:
        result = runner.invoke(app, ["compare", BASELINE, CANDIDATE, "--cluster", "nope"])
        assert result.exit_code == 3

    def test_unknown_gate_is_a_user_error(self) -> None:
        result = runner.invoke(app, ["compare", BASELINE, CANDIDATE, "--gate", "yolo"])
        assert result.exit_code == 3


class TestImportCommand:
    def test_promptfoo_roundtrip_feeds_compare(self, tmp_path: Path) -> None:
        base = tmp_path / "base.jsonl"
        cand = tmp_path / "cand.jsonl"
        first = runner.invoke(app, ["import", PROMPTFOO, "-o", str(base), "--variant", "baseline"])
        second = runner.invoke(
            app, ["import", PROMPTFOO, "-o", str(cand), "--variant", "candidate"]
        )
        assert first.exit_code == 0
        assert second.exit_code == 0
        result = runner.invoke(app, ["compare", str(base), str(cand), "--metric", "score"])
        assert result.exit_code in (0, 1, 2)
        assert "score" in result.output

    def test_unknown_variant_is_a_user_error(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["import", PROMPTFOO, "-o", str(tmp_path / "x.jsonl"), "--variant", "nope"],
        )
        assert result.exit_code == 3


class TestJudgeCli:
    def test_agreement_runs_on_the_mtbench_sample(self) -> None:
        sample = str(EXAMPLES / "mtbench-sample.jsonl")
        result = runner.invoke(
            app,
            ["judge", "agreement", sample, "--human", "human", "--judge", "gpt4"],
        )
        assert result.exit_code == 0
        assert "kappa" in result.output

    def test_agreement_needs_a_metric_choice_when_several_exist(self) -> None:
        result = runner.invoke(
            app, ["judge", "agreement", BASELINE, "--human", "a", "--judge", "b"]
        )
        assert result.exit_code == 3
        assert "metric" in result.output

    def test_position_reports_bias(self, tmp_path: Path) -> None:
        rows: list[dict[str, object]] = []
        for i in range(12):
            first = 1.0 if i < 5 else 0.0
            second = 1.0 if i < 8 else 0.0
            rows.append(
                {
                    "example_id": f"p{i}",
                    "variant": "original",
                    "metric": "candidate_preferred",
                    "value": first,
                }
            )
            rows.append(
                {
                    "example_id": f"p{i}",
                    "variant": "swapped",
                    "metric": "candidate_preferred",
                    "value": second,
                }
            )
        path = tmp_path / "swap.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        result = runner.invoke(app, ["judge", "position", str(path)])
        assert result.exit_code == 0
        assert "position bias" in result.output
        assert "flip rate" in result.output

    def test_stability_reports_flips(self, tmp_path: Path) -> None:
        rows: list[dict[str, object]] = []
        for run, values in [
            ("r1", [1.0, 1.0, 0.0]),
            ("r2", [1.0, 0.0, 0.0]),
            ("r3", [0.0, 1.0, 0.0]),
        ]:
            for i, value in enumerate(values):
                rows.append(
                    {
                        "example_id": f"q{i}",
                        "variant": "judge",
                        "metric": "pass",
                        "value": value,
                        "run_id": run,
                    }
                )
        path = tmp_path / "repeats.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        result = runner.invoke(app, ["judge", "stability", str(path)])
        assert result.exit_code == 0
        assert "flip rate" in result.output

    def test_scaffold_swap_prints_a_promptfoo_config(self) -> None:
        result = runner.invoke(app, ["judge", "scaffold-swap"])
        assert result.exit_code == 0
        assert "prompts:" in result.output
        assert "Answer A" in result.output


class TestPower:
    def test_reports_mde_and_items_for_a_target(self) -> None:
        result = runner.invoke(app, ["power", BASELINE, CANDIDATE, "--target", "0.05"])
        assert result.exit_code == 0
        assert "mde" in result.output
        assert "items" in result.output


class TestFloor:
    def test_needs_runs(self) -> None:
        result = runner.invoke(app, ["floor", BASELINE])
        assert result.exit_code == 3
        assert "run" in result.output

    def test_reports_the_floor_for_repeated_runs(self, tmp_path: Path) -> None:
        rows: list[dict[str, object]] = []
        for run, wobble in [("r1", 0.0), ("r2", 0.02), ("r3", -0.01)]:
            for i, value in enumerate([0.8, 0.9, 0.7, 0.85]):
                rows.append(
                    {
                        "example_id": f"q{i}",
                        "variant": "baseline",
                        "metric": "score",
                        "value": round(value + wobble, 3),
                        "run_id": run,
                    }
                )
        path = tmp_path / "runs.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        result = runner.invoke(app, ["floor", str(path)])
        assert result.exit_code == 0
        assert "floor" in result.output
