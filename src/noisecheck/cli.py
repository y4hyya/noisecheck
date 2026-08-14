"""The noisecheck command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from noisecheck import __version__
from noisecheck.errors import DataError, NoisecheckError
from noisecheck.io.csv_ import read_csv
from noisecheck.io.jsonl import read_jsonl
from noisecheck.io.promptfoo import read_promptfoo
from noisecheck.judge.agreement import judge_agreement
from noisecheck.judge.position import judge_position
from noisecheck.judge.scaffold import scaffold_swap_config
from noisecheck.judge.stability import judge_stability
from noisecheck.report.info import RunInfo, file_sha256
from noisecheck.report.json_out import render_json
from noisecheck.report.markdown import render_markdown
from noisecheck.report.rows import fmt
from noisecheck.report.terminal import build_terminal
from noisecheck.schema import Dataset, Record, pair
from noisecheck.stats.paired import ComparisonResult, compare_paired
from noisecheck.stats.power import items_needed, mde
from noisecheck.stats.variance import noise_floor
from noisecheck.verdict import Gate, judge

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Statistical verdicts for AI evals: is it a real change, or just noise?",
)
console = Console()
error_console = Console(stderr=True)

USER_ERROR_EXIT = 3

judge_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Check the judge itself: agreement with humans, position bias, stability.",
)
app.add_typer(judge_app, name="judge")


@app.command()
def compare(
    baseline_file: Path,
    candidate_file: Path,
    metric: Annotated[list[str] | None, typer.Option("--metric", "-m")] = None,
    cluster: Annotated[str | None, typer.Option("--cluster")] = None,
    gate: Annotated[str, typer.Option("--gate")] = "non-regression",
    alpha: Annotated[float, typer.Option("--alpha")] = 0.05,
    min_effect: Annotated[float, typer.Option("--min-effect")] = 0.0,
    power: Annotated[float, typer.Option("--power")] = 0.8,
    lower_is_better: Annotated[list[str] | None, typer.Option("--lower-is-better")] = None,
    b: Annotated[int, typer.Option("--b")] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    level: Annotated[float, typer.Option("--level")] = 0.95,
    json_path: Annotated[Path | None, typer.Option("--json")] = None,
    md_path: Annotated[Path | None, typer.Option("--md")] = None,
) -> None:
    """Compare two eval result files and print a verdict with an honest exit code."""
    try:
        comparisons, info = _build_comparisons(
            baseline_file, candidate_file, metric, cluster, b, seed, level
        )
        assessment = judge(
            comparisons,
            gate=_parse_gate(gate),
            alpha=alpha,
            min_effect=min_effect,
            power=power,
            lower_is_better=set(lower_is_better or ()),
        )
        console.print(build_terminal(comparisons, assessment, info))
        if json_path is not None:
            json_path.write_text(render_json(comparisons, assessment, info), encoding="utf-8")
        if md_path is not None:
            md_path.write_text(render_markdown(comparisons, assessment, info), encoding="utf-8")
    except NoisecheckError as error:
        error_console.print(f"error: {error}")
        raise typer.Exit(USER_ERROR_EXIT) from error
    raise typer.Exit(assessment.exit_code)


@app.command()
def power(
    baseline_file: Path,
    candidate_file: Path,
    metric: Annotated[list[str] | None, typer.Option("--metric", "-m")] = None,
    cluster: Annotated[str | None, typer.Option("--cluster")] = None,
    target: Annotated[float | None, typer.Option("--target")] = None,
    alpha: Annotated[float, typer.Option("--alpha")] = 0.05,
    power_goal: Annotated[float, typer.Option("--power")] = 0.8,
    b: Annotated[int, typer.Option("--b")] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Show the smallest difference these files can detect, and what a target needs."""
    try:
        comparisons, _ = _build_comparisons(
            baseline_file, candidate_file, metric, cluster, b, seed, 0.95
        )
        for comparison in comparisons:
            if comparison.se == 0.0:
                console.print(f"{comparison.metric}: zero variance, nothing to detect")
                continue
            detectable = mde(comparison.se, alpha=alpha, power=power_goal)
            line = f"{comparison.metric}: mde {fmt(detectable)} with {comparison.n} pairs"
            if target is not None:
                needed = items_needed(
                    comparison.se, comparison.n, target, alpha=alpha, power=power_goal
                )
                line = f"{line}, items for {target:g}: {needed}"
            console.print(line)
    except NoisecheckError as error:
        error_console.print(f"error: {error}")
        raise typer.Exit(USER_ERROR_EXIT) from error


@app.command()
def floor(
    file: Path,
    variant: Annotated[str | None, typer.Option("--variant")] = None,
    metric: Annotated[list[str] | None, typer.Option("--metric", "-m")] = None,
) -> None:
    """Show how much each metric moves between identical runs."""
    try:
        dataset = _load(file)
        chosen_variant = _sole_variant(dataset, file) if variant is None else variant
        metrics = sorted(metric or dataset.metrics())
        for name in metrics:
            result = noise_floor(dataset, chosen_variant, name)
            console.print(
                f"{name}: floor ±{fmt(result.floor)} "
                f"(run sd {fmt(result.run_sd)} over {result.n_runs} runs)"
            )
            if result.flip_rates:
                flips = ", ".join(
                    f"{example} {rate:.0%}" for example, rate in result.flip_rates[:5]
                )
                console.print(f"  flippiest examples: {flips}")
    except NoisecheckError as error:
        error_console.print(f"error: {error}")
        raise typer.Exit(USER_ERROR_EXIT) from error


@app.command("import")
def import_(
    source: Path,
    output: Annotated[Path, typer.Option("--output", "-o")],
    variant: Annotated[str | None, typer.Option("--variant")] = None,
) -> None:
    """Convert promptfoo or csv results into canonical jsonl."""
    try:
        dataset = _load(source)
        records = dataset.records
        if variant is not None:
            records = tuple(r for r in records if r.variant == variant)
            if not records:
                raise DataError(
                    f"variant {variant!r} not found, the file has "
                    f"{', '.join(sorted(dataset.variants()))}"
                )
        lines = [
            json.dumps(record.model_dump(exclude_none=True), sort_keys=True) for record in records
        ]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"wrote {len(lines)} records to {output}")
    except NoisecheckError as error:
        error_console.print(f"error: {error}")
        raise typer.Exit(USER_ERROR_EXIT) from error


@judge_app.command()
def agreement(
    file: Path,
    human: Annotated[str, typer.Option("--human")] = "human",
    judge_variant: Annotated[str, typer.Option("--judge")] = "judge",
    metric: Annotated[str | None, typer.Option("--metric", "-m")] = None,
    weights: Annotated[str, typer.Option("--weights")] = "none",
    b: Annotated[int, typer.Option("--b")] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    level: Annotated[float, typer.Option("--level")] = 0.95,
) -> None:
    """How well the judge tracks human labels, corrected for lucky agreement."""
    try:
        dataset = _load(file)
        chosen = metric or _sole_metric(dataset, file)
        result = judge_agreement(
            pair(dataset, human, judge_variant, chosen),
            b=b,
            seed=seed,
            level=level,
            weights=weights,
        )
        console.print(
            f"{chosen}: kappa {fmt(result.kappa)} "
            f"[{fmt(result.ci_low)}, {fmt(result.ci_high)}], "
            f"raw agreement {result.observed_agreement:.0%} over {result.n} pairs"
        )
        for human_label, judge_label, count in result.confusion:
            console.print(f"  human {human_label} and judge {judge_label}: {count}")
        if result.worst_disagreements:
            worst = ", ".join(result.worst_disagreements[:5])
            console.print(f"  worst disagreements: {worst}")
        for warning in result.warnings:
            console.print(f"  {warning}", style="dim")
    except NoisecheckError as error:
        error_console.print(f"error: {error}")
        raise typer.Exit(USER_ERROR_EXIT) from error


@judge_app.command()
def position(
    file: Path,
    original: Annotated[str, typer.Option("--original")] = "original",
    swapped: Annotated[str, typer.Option("--swapped")] = "swapped",
    metric: Annotated[str | None, typer.Option("--metric", "-m")] = None,
    b: Annotated[int, typer.Option("--b")] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    level: Annotated[float, typer.Option("--level")] = 0.95,
) -> None:
    """Whether the judge's preference follows the seat instead of the answer."""
    try:
        dataset = _load(file)
        chosen = metric or _sole_metric(dataset, file)
        result = judge_position(
            pair(dataset, original, swapped, chosen), b=b, seed=seed, level=level
        )
        console.print(
            f"{chosen}: position bias {fmt(result.bias)} "
            f"[{fmt(result.ci_low)}, {fmt(result.ci_high)}], "
            f"flip rate {result.flip_rate:.0%} over {result.n} pairs"
        )
        if result.mcnemar_p is not None:
            console.print(f"  mcnemar p {fmt(result.mcnemar_p)}")
        for warning in result.warnings:
            console.print(f"  {warning}", style="dim")
    except NoisecheckError as error:
        error_console.print(f"error: {error}")
        raise typer.Exit(USER_ERROR_EXIT) from error


@judge_app.command()
def stability(
    file: Path,
    variant: Annotated[str | None, typer.Option("--variant")] = None,
    metric: Annotated[str | None, typer.Option("--metric", "-m")] = None,
) -> None:
    """How often the judge changes its mind on identical inputs."""
    try:
        dataset = _load(file)
        chosen_variant = _sole_variant(dataset, file) if variant is None else variant
        chosen = metric or _sole_metric(dataset, file)
        result = judge_stability(dataset, chosen_variant, chosen)
        line = (
            f"{chosen}: floor ±{fmt(result.floor)} over {result.n_runs} runs "
            f"of {result.n_examples} examples"
        )
        if result.flip_rate is not None:
            line = f"{line}, flip rate {result.flip_rate:.0%}"
        console.print(line)
        if result.flip_rates:
            flips = ", ".join(f"{example} {rate:.0%}" for example, rate in result.flip_rates[:5])
            console.print(f"  flippiest examples: {flips}")
        for warning in result.warnings:
            console.print(f"  {warning}", style="dim")
    except NoisecheckError as error:
        error_console.print(f"error: {error}")
        raise typer.Exit(USER_ERROR_EXIT) from error


@judge_app.command("scaffold-swap")
def scaffold_swap() -> None:
    """Print a promptfoo config template for an order swap experiment."""
    console.print(scaffold_swap_config(), highlight=False)


def _sole_metric(dataset: Dataset, path: Path) -> str:
    metrics = sorted(dataset.metrics())
    if len(metrics) != 1:
        raise DataError(
            f"{path} contains {len(metrics)} metrics ({', '.join(metrics)}), pick one with --metric"
        )
    return metrics[0]


def _build_comparisons(
    baseline_file: Path,
    candidate_file: Path,
    chosen_metrics: list[str] | None,
    cluster: str | None,
    b: int,
    seed: int,
    level: float,
) -> tuple[list[ComparisonResult], RunInfo]:
    baseline_ds = _load(baseline_file)
    candidate_ds = _load(candidate_file)
    baseline_variant = _sole_variant(baseline_ds, baseline_file)
    candidate_variant = _sole_variant(candidate_ds, candidate_file)
    if baseline_variant == candidate_variant:
        raise DataError(f"both files contain variant {baseline_variant!r}, rename one side")
    records = [*baseline_ds.records, *candidate_ds.records]
    if cluster is not None:
        records = [_with_cluster(record, cluster) for record in records]
    merged = Dataset(records)

    shared = sorted(baseline_ds.metrics() & candidate_ds.metrics())
    if chosen_metrics:
        unknown = sorted(set(chosen_metrics) - set(shared))
        if unknown:
            raise DataError(f"metrics not present on both sides: {', '.join(unknown)}")
        shared = sorted(set(chosen_metrics))
    if not shared:
        raise DataError("the two files share no metrics")

    comparisons = [
        compare_paired(
            pair(merged, baseline_variant, candidate_variant, name),
            b=b,
            seed=seed,
            level=level,
        )
        for name in shared
    ]
    info = RunInfo(
        baseline_path=str(baseline_file),
        candidate_path=str(candidate_file),
        baseline_sha256=file_sha256(baseline_file),
        candidate_sha256=file_sha256(candidate_file),
        package_version=__version__,
        seed=seed,
        b=b,
        level=level,
    )
    return comparisons, info


def _load(path: Path) -> Dataset:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)
    if suffix == ".json":
        return read_promptfoo(path)
    raise DataError(
        f"{path}: unsupported file type {suffix or 'none'}, expected .jsonl, .csv, or .json"
    )


def _sole_variant(dataset: Dataset, path: Path) -> str:
    variants = sorted(dataset.variants())
    if len(variants) != 1:
        raise DataError(
            f"{path} contains {len(variants)} variants ({', '.join(variants)}), "
            f"split it with noisecheck import --variant"
        )
    return variants[0]


def _with_cluster(record: Record, key: str) -> Record:
    if key == "example_id":
        return record.model_copy(update={"cluster_id": record.example_id})
    meta = record.meta or {}
    value = meta.get(key)
    if value is None:
        raise DataError(
            f"record {record.example_id!r}: meta key {key!r} not found, cannot cluster by it"
        )
    return record.model_copy(update={"cluster_id": str(value)})


def _parse_gate(text: str) -> Gate:
    normalized = text.replace("-", "_").strip().lower()
    try:
        return Gate(normalized)
    except ValueError as error:
        raise DataError(f"unknown gate {text!r}, choose non-regression or improvement") from error
