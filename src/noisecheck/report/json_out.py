"""Machine readable report, deterministic down to the byte."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict

from noisecheck.report.info import RunInfo
from noisecheck.report.rows import report_rows
from noisecheck.stats.paired import ComparisonResult
from noisecheck.verdict import Assessment


def render_json(
    comparisons: Sequence[ComparisonResult],
    assessment: Assessment,
    info: RunInfo,
) -> str:
    """Render the full result as stable json with provenance included."""
    rows = report_rows(comparisons, assessment)
    payload = {
        "info": asdict(info),
        "assessment": {
            "overall": str(assessment.overall),
            "gate": str(assessment.gate),
            "exit_code": assessment.exit_code,
            "alpha": assessment.alpha,
            "min_effect": assessment.min_effect,
            "power": assessment.power,
        },
        "metrics": [
            {
                "metric": row.comparison.metric,
                "n": row.comparison.n,
                "estimate": row.comparison.estimate,
                "se": row.comparison.se,
                "ci_low": row.comparison.ci_low,
                "ci_high": row.comparison.ci_high,
                "p_value": row.comparison.p_value,
                "q_value": row.verdict.q_value,
                "mcnemar_p": row.comparison.mcnemar_p,
                "outcome": str(row.verdict.outcome),
                "reason": row.verdict.reason,
                "mde": row.verdict.mde,
                "items_needed": row.verdict.items_needed,
                "n_clusters": row.comparison.n_clusters,
                "icc": row.comparison.icc,
                "design_effect": row.comparison.design_effect,
                "effective_n": row.comparison.effective_n,
                "warnings": list(row.comparison.warnings),
            }
            for row in rows
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
