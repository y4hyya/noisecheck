# methods

This page states what every number in noisecheck means, what it assumes, and
where it breaks. The claims about interval coverage are not taken on faith:
they are measured in [validation.md](validation.md), including the regimes
where they fall short.

## the data model

One record is one metric value for one example under one variant. Records can
carry a `cluster_id` (a conversation, a task, any group whose rows move
together) and a `run_id` (which repeat of the same eval produced the row).
Everything below consumes this shape.

## pairing

Comparisons are paired: the same example under baseline and candidate, matched
by example id and run id. Pairing removes the between example difficulty
variance from the comparison, which is why a paired test detects differences
an unpaired one cannot (see the
[case study](case-study-paired.md)). If more than 10 percent of either side
has no partner, noisecheck refuses to pair instead of silently dropping rows,
because a comparison built on a quietly shifted subset is biased in an
invisible way. The unpaired Welch fallback exists for data that genuinely
cannot be paired, and every unpaired result says plainly that pairing detects
far smaller differences.

## the interval: studentized bootstrap

The 95 percent interval on a mean delta comes from a studentized bootstrap:
resample the deltas with replacement b times (default 10000), compute each
resample's t statistic (its mean minus the observed mean, divided by its own
standard error), and read the interval from the quantiles of those t values.
Studentizing gives better small sample behavior than the plain percentile
method. Two guards handle floating point reality: a resample whose values are
all identical has no standard error of its own, so it borrows the full sample
standard error (detected by its spread being exactly zero, because a rounded
standard deviation of identical values is not reliably zero), and a sample
whose variance underflows below float precision is treated as degenerate
rather than divided by. Degenerate data (all deltas equal) gets a point
interval and a loud warning, never a fabricated range.

## the p value: sign flip permutation

Under the hypothesis that baseline and candidate are exchangeable, flipping
the sign of any delta produces an equally likely world. noisecheck flips
signs at random b times and counts how often the flipped world shows a mean
gap at least as large as the observed one. The reported p is (count + 1)
divided by (b + 1), so finite resampling can never claim impossible
certainty. With clusters, whole clusters flip together. The test assumes the
no difference world is symmetric; with fewer than about 30 heavily skewed
deltas it over rejects (measured honestly in validation.md), which is one
reason every small comparison carries a warning.

## binary metrics: mcnemar

For pass or fail metrics, only the flips carry information about the
difference: examples that pass both times or fail both times say nothing.
McNemar's exact test asks whether the flips lean one way more than a fair
coin would, via an exact binomial on the discordant pairs. With clustered
data McNemar's independence assumption fails, so the result carries a warning
telling you to trust the permutation p instead.

## clusters

Rows inside one conversation succeed or fail together, so treating them as
independent overstates the evidence. When cluster ids are present noisecheck
switches every piece of machinery to the cluster level: the standard error
becomes the cluster robust estimate (residuals of cluster totals against the
grand mean, with a k over k minus 1 correction), the bootstrap resamples
whole clusters, and the permutation flips whole clusters. The result also
reports the intraclass correlation (a one way analysis of variance estimator,
clipped to zero when negative), the design effect 1 + (m − 1) · icc for
average cluster size m, and the effective sample size n divided by the design
effect: how many independent rows the data is really worth.

Limits, stated plainly: below 10 clusters the cluster bootstrap is unstable
(warned, and visible in validation.md), a single cluster is refused outright,
and clusters are assumed exchangeable with each other. Shared drift that
moves all clusters together, like a provider change halfway through a run,
is not captured; run level clustering absorbs the worst of it, and crossed
random effects are the honest next step on the roadmap.

## power: mde and items needed

The minimum detectable effect is (z for the confidence level plus z for the
power) times the standard error, the smallest true difference the eval would
flag with the requested power (default 80 percent). Items needed inverts it:
standard errors shrink with the square root of n, so reaching a target
difference costs n times (current mde over target) squared items. Both are
normal approximations on the sampling distribution of the mean, cross checked
by the bootstrap, and both assume the noise structure stays as measured while
you add items.

## many metrics: benjamini hochberg

Check 20 metrics at 5 percent each and one will look special by luck.
Verdicts therefore run on Benjamini Hochberg adjusted q values, which control
the expected share of false discoveries across the whole metric family. Raw p
values are always shown beside the adjusted ones. The mathematical guarantee
that q is never below p is enforced exactly in code, because floating point
rounding alone can violate it by one unit in the last place.

## verdicts, gates, and exit codes

Each metric lands in one of four states: improvement (significant and at
least the minimum effect that matters, in the good direction), regression
(the mirror image), no detectable difference (either a real but trivial
difference, or a powered eval that found nothing), and underpowered (the
eval cannot answer the question, with a prescription for how many items
would). Metrics where lower is better are flipped before judging.

Two gate policies turn verdicts into an exit code. The default non
regression gate exits 0 unless a real regression exists (exit 1) or the
interval cannot rule out a regression of the minimum effect size (exit 2);
the rule out check uses a strict inequality, so identical variants pass
cleanly. The improvement gate exits 0 only when every metric improved. Data
and usage errors exit 3, so a broken file can never masquerade as a verdict.

## judge checks

Kappa measures judge versus human agreement after subtracting the agreement
two coin flippers would reach given the same label frequencies. Weighted
variants (linear, quadratic) treat ordered grades by their distance apart,
matching scikit learn's conventions exactly (verified in the golden tests).
When every label is identical kappa is undefined and reported as 0 with a
warning. The interval is a percentile bootstrap over clusters, built on the
fact that kappa depends only on the confusion matrix, so resampling clusters
means summing per cluster confusion matrices; the percentile method is the
standard choice for kappa, where studentizing has no stable variance
estimate.

Position bias comes from judging the same pairs twice with the answer order
swapped. A fair judge gives the same verdicts; the bias estimate is the
change in candidate win rate attributable to the seat, with the usual paired
machinery behind it, plus the flip rate and a McNemar test on the flipped
verdicts.

Stability repeats the same judging run and reports how often the judge
changes its mind on identical inputs, per item and overall, plus the judge's
own noise floor.

## the noise floor

Run the same eval twice with zero changes and the score still moves. From k
repeated runs, the floor is the run to run standard deviation of the per run
means times the t multiplier for k − 1 degrees of freedom: roughly where a
single fresh run would land 95 percent of the time. Runs must cover
identical example sets, otherwise the floor would mix run noise with
composition differences. Differences inside the floor are weather, not
climate.

## determinism and provenance

Every random step is seeded (default 42) and chunked deterministically:
identical inputs give byte identical reports. The json report embeds the
package version, the seed, and the sha256 of both input files, so any result
can be traced to exact data and reproduced.

## floating point honesty

Three real bugs were found by property based testing before release and are
now design decisions with pinned regression tests: identical valued
resamples whose rounded standard deviation is tiny but not zero (detected by
spread instead), variances that underflow to exact zero for values near
1e−230 (treated as degenerate), and Benjamini Hochberg adjustments that
rounding pushed one unit below their own p value (invariant now enforced).
One limit remains inherent: an interval endpoint is a difference of large
numbers, so an endpoint near zero carries cancellation noise proportional to
the magnitudes involved, in any implementation.

## known limitations

Cluster exchangeability does not cover drift shared by all clusters. Cluster
resampling needs roughly 10 clusters to be trustworthy. McNemar ignores
clustering. The mde is a normal approximation. The sign flip test assumes a
symmetric null and over rejects on small skewed samples. Binary metrics on
30 items cannot honestly carry a 95 percent promise (measured at about 87
percent in validation.md). The kappa interval is percentile, not
studentized. All of these are either warned about at runtime, measured in
validation.md, or both.

## references

Evan Miller, Adding Error Bars to Evals, arXiv:2411.00640. The MT Bench
sample in examples comes from lmsys/mt_bench_human_judgments (CC BY 4.0).
