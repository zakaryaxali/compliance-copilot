"""Run the three eval sets and emit metrics.

Usage:
    python src/evaluate.py --model ./checkpoints/final --eval-set all

Eval sets:
    held-out  - 100 synthetic test pairs (format + domain vocab)
    hard      - 30 hand-written edge cases (does it hedge?)
    baseline  - same prompts vs base Mistral and a frontier model

Metrics:
    - format adherence (regex over the three-part structure)
    - citation accuracy (does the cited section exist in the source standard?)
    - LLM-as-judge answer quality
"""

from __future__ import annotations

import argparse
from pathlib import Path

EVAL_SETS = {"held-out", "hard", "baseline", "all"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Path to fine-tuned model")
    p.add_argument("--eval-set", choices=sorted(EVAL_SETS), default="all")
    p.add_argument("--output", type=Path, default=Path("results/eval_results.md"))
    p.add_argument("--judge-model", default="claude-opus-4-7")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Evaluating {args.model} on '{args.eval_set}'...")
    raise NotImplementedError("Evaluation harness not yet implemented.")


if __name__ == "__main__":
    main()
