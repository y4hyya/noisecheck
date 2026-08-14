# changelog

## 0.1.0

First release.

- paired comparisons with a studentized bootstrap interval, a sign flip permutation p value, and exact McNemar for pass or fail metrics
- cluster aware resampling for conversations and repeated runs, with intraclass correlation, design effect, and effective sample size
- power analysis: the smallest detectable effect and the items a target needs
- Benjamini Hochberg adjustment across metric families
- verdicts with two gate policies and honest exit codes: 0 ship, 1 stop, 2 cannot tell, 3 broken input
- judge checks: kappa agreement against human labels, position bias from order swapped runs, stability across repeats, and a promptfoo swap scaffold
- noise floor measurement over repeated identical runs
- readers for jsonl, csv, and promptfoo output, plus an import converter
- terminal, markdown, and byte deterministic json reports with sha256 provenance
- published coverage simulations, a methods page, and a paired versus unpaired case study
