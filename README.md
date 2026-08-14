# noisecheck

You test your AI twice with a quiz. First it scored 71. Second it scored 74.
Is it really better, or did you just get lucky?

noisecheck reads your eval results and gives you a straight answer: real improvement, real regression, or just noise. It also tells you the smallest change your eval can actually detect, and how many more test items you need when it cannot.

It never runs your evals and never calls any AI. It only does the math, on your files, on your machine, the same way every time.

## sixty seconds

```
uvx --from git+https://github.com/y4hyya/noisecheck noisecheck compare baseline.jsonl candidate.jsonl
```

Your files need one json line per answer:

```json
{"example_id": "q001", "variant": "baseline", "metric": "pass", "value": 1}
```

Already using promptfoo? Convert its output first:

```
noisecheck import results.json -o baseline.jsonl --variant my-model
```

## the trap this tool exists for

The repo ships a realistic example: 40 questions asked inside 8 conversations, two versions of an agent. Compare them the usual way and both metrics look like wins:

```
$ noisecheck compare examples/baseline.jsonl examples/candidate.jsonl

| metric       | verdict     | estimate | interval          | q      | mde     |
| latency_s    | improvement | +0.1165  | [0.1069, 0.127]   | 0.0002 | 0.01387 |
| task_success | improvement | +0.175   | [0.04758, 0.3515] | 0.0391 | 0.1978  |

gate non regression: pass (exit 0)
```

But questions inside one conversation go wrong together. Tell noisecheck about the conversations and it counts the evidence honestly:

```
$ noisecheck compare examples/baseline.jsonl examples/candidate.jsonl --cluster conversation_id

| metric       | verdict      | estimate | interval         | q       | mde     |
| latency_s    | improvement  | +0.1165  | [0.1023, 0.1295] | 0.0144  | 0.01569 |
| task_success | underpowered | +0.175   | [0.037, 0.3362]  | 0.06389 | 0.1653  |

task_success: this eval cannot detect differences below 0.1653, about 36 items would
```

Same data. The latency improvement survives, because it shows up inside every conversation. The task success "win" honestly becomes "cannot tell": 8 conversations are not enough independent evidence, and the tool says how many items would settle it.

This is not a matter of taste. [validation.md](docs/validation.md) simulates a thousand worlds per row where the truth is known: the naive analysis fires up to 32 percent false alarms on clustered data, while noisecheck stays at the promised 5.

## who grades the grader

If an AI judges your AI, check the judge before trusting the scores. On 80 real MT Bench battles where expert humans and GPT-4 judged the same model pairs:

```
$ noisecheck judge agreement examples/mtbench-sample.jsonl --human human --judge gpt4

winner: kappa 0.1513 [-0.02142, 0.3293], raw agreement 62% over 80 pairs
```

Raw agreement sounds fine at 62 percent. Kappa subtracts the agreement a coin flipping judge would get, and what is left barely clears zero. There are also `judge position` (does the verdict follow the seat instead of the answer?), `judge stability` (does the judge change its mind on identical inputs?), and `judge scaffold-swap` (a promptfoo template to produce the swapped runs).

## in your CI

Exit codes are the contract: 0 ship, 1 stop, 2 cannot tell, 3 your files are broken.

```yaml
- run: noisecheck compare baseline.jsonl candidate.jsonl --min-effect 0.02 --json report.json
```

Set `--min-effect` to the smallest difference you actually care about. Use `--gate improvement` when only proven wins may pass, and `--lower-is-better latency_s` for metrics where smaller is better. Reports embed the seed, the package version, and the sha256 of both input files, and identical inputs produce byte identical reports.

## what noisecheck does not do

- it does not run evals, call models, or need API keys
- no dashboard, no account, nothing leaves your machine
- it does not invent certainty: when the data cannot answer, the verdict is "underpowered" plus the number of items that would fix it

## the details

- [methods](docs/methods.md): every statistic, its assumptions, and where it breaks
- [validation](docs/validation.md): the 95 percent promise, measured, including the regimes where it falls short
- [case study](docs/case-study-paired.md): the same data giving two opposite answers, and why the most starred eval harness has this exact bug filed against it

Commands: `compare`, `power`, `floor`, `import`, `judge agreement`, `judge position`, `judge stability`, `judge scaffold-swap`. Each has `--help`.

The MT Bench sample comes from [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) (CC BY 4.0). MIT license.
