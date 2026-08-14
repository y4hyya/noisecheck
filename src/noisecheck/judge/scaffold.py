"""Emit a promptfoo config for judge swap experiments."""

from __future__ import annotations

SWAP_CONFIG = """description: pairwise judge with order swap, for noisecheck judge position
prompts:
  - label: original
    raw: |
      You are judging two answers to the same question.
      Question: {{question}}
      Answer A: {{answer_baseline}}
      Answer B: {{answer_candidate}}
      Reply with exactly one letter, A or B, naming the better answer.
  - label: swapped
    raw: |
      You are judging two answers to the same question.
      Question: {{question}}
      Answer A: {{answer_candidate}}
      Answer B: {{answer_baseline}}
      Reply with exactly one letter, A or B, naming the better answer.
providers:
  - id: openai:gpt-4.1-mini
tests: []
"""


def scaffold_swap_config() -> str:
    """The promptfoo config template that produces order swapped judge runs."""
    return SWAP_CONFIG
