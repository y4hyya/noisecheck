# the same data, two opposite answers

Sixty questions. The candidate model is genuinely better by about 0.03 on
almost every single question. But the questions themselves differ wildly in
difficulty: scores range from near 0.1 to near 0.95. This is what real eval
data looks like.

## the wrong way

Compare the two piles of scores as if they were unrelated samples (Welch's
test, the unpaired approach):

```
estimate: 0.0255
p value: 0.443
interval: (-0.0402, 0.0912)
```

Nothing there. The difficulty spread between questions drowns the small,
consistent improvement. Verdict: no detectable difference.

## the right way

Pair each question with itself, subtract, and analyze the per question
deltas. The difficulty cancels out because each question is compared only
with itself:

```
$ noisecheck compare examples/case-baseline.jsonl examples/case-candidate.jsonl

| metric | verdict     | estimate | interval           | q         | mde      |
| score  | improvement | +0.02554 | [0.01983, 0.03105] | 9.999e-05 | 0.007812 |

score: a real improvement of about 0.02554
gate non regression: pass (exit 0)
```

Same data. Same estimate. Opposite answer, and the paired analysis is the
correct one: the improvement is real and the interval is tight.

## try it yourself

```
uv run python scripts/dev/make_case_study.py
uv run noisecheck compare examples/case-baseline.jsonl examples/case-candidate.jsonl
```

The Welch numbers come from the library on the identical files:

```python
from noisecheck import compare_unpaired, read_jsonl

baseline = [r.value for r in read_jsonl("examples/case-baseline.jsonl").records]
candidate = [r.value for r in read_jsonl("examples/case-candidate.jsonl").records]
print(compare_unpaired(baseline, candidate))
```

## why this page exists

This is not a strawman. lm-evaluation-harness, the most starred eval harness
in the field, has an open issue documenting that its model comparison script
applies an unpaired z test to paired data:
[EleutherAI/lm-evaluation-harness#3831](https://github.com/EleutherAI/lm-evaluation-harness/issues/3831).
On data shaped like this case study, that comparison reports nothing where a
real improvement exists. Pairing is not a refinement, it is the difference
between seeing your result and not seeing it.

noisecheck pairs by default and refuses to pair silently when the two sides
do not match ([methods](methods.md)).
