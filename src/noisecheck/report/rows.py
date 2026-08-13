from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from noisecheck.stats.paired import ComparisonResult
from noisecheck.verdict import Assessment, MetricVerdict

EXIT_WORDS = {0: "pass", 1: "fail", 2: "cannot tell"}


@dataclass(frozen=True)
class ReportRow:
    comparison: ComparisonResult
    verdict: MetricVerdict


def report_rows(comparisons: Sequence[ComparisonResult], assessment: Assessment) -> list[ReportRow]:
    by_metric = {comparison.metric: comparison for comparison in comparisons}
    verdict_metrics = {verdict.metric for verdict in assessment.verdicts}
    if set(by_metric) != verdict_metrics:
        raise ValueError("comparisons and assessment cover different metric sets")
    ordered = sorted(assessment.verdicts, key=lambda verdict: verdict.metric)
    return [ReportRow(by_metric[verdict.metric], verdict) for verdict in ordered]


def humanize(value: str) -> str:
    return str(value).replace("_", " ")


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4g}"


def fmt_signed(value: float) -> str:
    return f"{value:+.4g}"
