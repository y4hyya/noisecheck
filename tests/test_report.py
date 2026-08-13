from __future__ import annotations

import json

import pytest

from noisecheck import Assessment, ComparisonResult, judge
from noisecheck.report.info import RunInfo, file_sha256
from noisecheck.report.json_out import render_json
from noisecheck.report.markdown import render_markdown
from noisecheck.report.rows import report_rows
from noisecheck.report.terminal import build_terminal, render_text


def comparison(
    metric: str = "accuracy",
    estimate: float = 0.05,
    se: float = 0.01,
    p: float = 0.001,
    ci: tuple[float, float] = (0.03, 0.07),
    warnings: tuple[str, ...] = (),
) -> ComparisonResult:
    return ComparisonResult(
        metric=metric,
        n=200,
        estimate=estimate,
        se=se,
        ci_low=ci[0],
        ci_high=ci[1],
        p_value=p,
        mcnemar_p=None,
        level=0.95,
        b=1000,
        seed=42,
        warnings=warnings,
    )


def scenario() -> tuple[list[ComparisonResult], Assessment, RunInfo]:
    comparisons = [
        comparison(),
        comparison(
            metric="latency",
            estimate=0.01,
            se=0.05,
            p=0.4,
            ci=(-0.09, 0.11),
            warnings=("only 20 pairs, the permutation p value is more reliable",),
        ),
    ]
    assessment = judge(comparisons, min_effect=0.02)
    info = RunInfo(
        baseline_path="baseline.jsonl",
        candidate_path="candidate.jsonl",
        baseline_sha256="a" * 64,
        candidate_sha256="b" * 64,
        package_version="1.2.3",
        seed=42,
        b=1000,
        level=0.95,
    )
    return comparisons, assessment, info


class TestJsonReport:
    def test_is_deterministic(self) -> None:
        comparisons, assessment, info = scenario()
        first = render_json(comparisons, assessment, info)
        second = render_json(comparisons, assessment, info)
        assert first == second
        assert first.endswith("\n")

    def test_carries_provenance_and_the_verdict(self) -> None:
        comparisons, assessment, info = scenario()
        parsed = json.loads(render_json(comparisons, assessment, info))
        assert parsed["info"]["package_version"] == "1.2.3"
        assert parsed["info"]["baseline_sha256"] == "a" * 64
        assert parsed["info"]["seed"] == 42
        assert parsed["assessment"]["exit_code"] == assessment.exit_code
        assert parsed["assessment"]["gate"] == "non_regression"

    def test_metrics_are_sorted_and_merged(self) -> None:
        comparisons, assessment, info = scenario()
        parsed = json.loads(render_json(comparisons, assessment, info))
        names = [entry["metric"] for entry in parsed["metrics"]]
        assert names == ["accuracy", "latency"]
        first = parsed["metrics"][0]
        assert first["outcome"] == "improvement"
        assert first["q_value"] is not None
        assert first["warnings"] == []


class TestMarkdownReport:
    def test_has_a_row_per_metric_and_the_gate_line(self) -> None:
        comparisons, assessment, info = scenario()
        text = render_markdown(comparisons, assessment, info)
        assert "| accuracy |" in text
        assert "| latency |" in text
        assert f"exit {assessment.exit_code}" in text

    def test_humanizes_outcome_names(self) -> None:
        comparisons, assessment, info = scenario()
        text = render_markdown(comparisons, assessment, info)
        assert "no_detectable_difference" not in text

    def test_lists_warnings(self) -> None:
        comparisons, assessment, info = scenario()
        text = render_markdown(comparisons, assessment, info)
        assert "only 20 pairs" in text


class TestTerminalReport:
    def test_renders_metrics_and_verdicts(self) -> None:
        comparisons, assessment, info = scenario()
        text = render_text(build_terminal(comparisons, assessment, info), width=120)
        assert "accuracy" in text
        assert "improvement" in text
        assert "latency" in text

    def test_narrow_width_does_not_crash(self) -> None:
        comparisons, assessment, info = scenario()
        text = render_text(build_terminal(comparisons, assessment, info), width=60)
        assert isinstance(text, str) and text


class TestRows:
    def test_mismatched_metric_sets_rejected(self) -> None:
        comparisons, assessment, _ = scenario()
        with pytest.raises(ValueError, match="metric"):
            report_rows([comparisons[0]], assessment)


class TestFileSha:
    def test_hashes_file_contents(self, tmp_path: object) -> None:
        from pathlib import Path

        target = Path(str(tmp_path)) / "x.jsonl"
        target.write_text("hello\n", encoding="utf-8")
        digest = file_sha256(target)
        assert len(digest) == 64
        assert digest == file_sha256(target)
