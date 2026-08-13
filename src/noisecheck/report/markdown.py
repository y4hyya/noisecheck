"""Markdown report for pull requests and docs."""

from __future__ import annotations

from collections.abc import Sequence

from noisecheck.report.info import RunInfo
from noisecheck.report.rows import EXIT_WORDS, fmt, fmt_signed, humanize, report_rows
from noisecheck.stats.paired import ComparisonResult
from noisecheck.verdict import Assessment


def render_markdown(
    comparisons: Sequence[ComparisonResult],
    assessment: Assessment,
    info: RunInfo,
) -> str:
    """Render the result as a markdown page with a table and plain reasons."""
    rows = report_rows(comparisons, assessment)
    lines = [
        "# noisecheck report",
        "",
        f"baseline `{info.baseline_path}` (sha {info.baseline_sha256[:12]}) vs "
        f"candidate `{info.candidate_path}` (sha {info.candidate_sha256[:12]})",
        "",
        f"noisecheck {info.package_version}, seed {info.seed}, b {info.b}, level {info.level:g}",
        "",
        "| metric | verdict | estimate | interval | q | mde |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        comparison, verdict = row.comparison, row.verdict
        lines.append(
            f"| {comparison.metric} | {humanize(verdict.outcome)} | "
            f"{fmt_signed(comparison.estimate)} | "
            f"[{fmt(comparison.ci_low)}, {fmt(comparison.ci_high)}] | "
            f"{fmt(verdict.q_value)} | {fmt(verdict.mde)} |"
        )
    lines.append("")
    for row in rows:
        lines.append(f"- {row.comparison.metric}: {row.verdict.reason}")
        for warning in row.comparison.warnings:
            lines.append(f"  - warning: {warning}")
    lines.append("")
    lines.append(
        f"gate {humanize(assessment.gate)}: {EXIT_WORDS[assessment.exit_code]} "
        f"(exit {assessment.exit_code})"
    )
    return "\n".join(lines) + "\n"
