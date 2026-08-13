"""Terminal report rendered with rich."""

from __future__ import annotations

import io
from collections.abc import Sequence

from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

from noisecheck.report.info import RunInfo
from noisecheck.report.rows import EXIT_WORDS, fmt, fmt_signed, humanize, report_rows
from noisecheck.stats.paired import ComparisonResult
from noisecheck.verdict import Assessment, Outcome

_OUTCOME_STYLE = {
    Outcome.IMPROVEMENT: "bold green",
    Outcome.REGRESSION: "bold red",
    Outcome.UNDERPOWERED: "bold yellow",
    Outcome.NO_DETECTABLE_DIFFERENCE: "bold",
}
_EXIT_STYLE = {0: "bold green", 1: "bold red", 2: "bold yellow"}


def build_terminal(
    comparisons: Sequence[ComparisonResult],
    assessment: Assessment,
    info: RunInfo,
) -> RenderableType:
    """Build the terminal report as one rich renderable."""
    rows = report_rows(comparisons, assessment)
    table = Table(title=f"noisecheck {info.package_version}  seed {info.seed}")
    for header in ("metric", "verdict", "estimate", "interval", "q", "mde"):
        table.add_column(header)
    for row in rows:
        comparison, verdict = row.comparison, row.verdict
        table.add_row(
            comparison.metric,
            Text(humanize(verdict.outcome), style=_OUTCOME_STYLE[verdict.outcome]),
            fmt_signed(comparison.estimate),
            f"[{fmt(comparison.ci_low)}, {fmt(comparison.ci_high)}]",
            fmt(verdict.q_value),
            fmt(verdict.mde),
        )
    pieces: list[RenderableType] = [table]
    for row in rows:
        pieces.append(Text(f"{row.comparison.metric}: {row.verdict.reason}"))
        for warning in row.comparison.warnings:
            pieces.append(Text(f"  {row.comparison.metric}: {warning}", style="dim"))
    pieces.append(
        Text(
            f"gate {humanize(assessment.gate)}: {EXIT_WORDS[assessment.exit_code]} "
            f"(exit {assessment.exit_code})",
            style=_EXIT_STYLE[assessment.exit_code],
        )
    )
    return Group(*pieces)


def render_text(renderable: RenderableType, width: int = 100) -> str:
    """Render any rich renderable to plain text, for tests and piping."""
    console = Console(file=io.StringIO(), width=width, record=True, force_terminal=False)
    console.print(renderable)
    return console.export_text()
