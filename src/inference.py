"""Interactive inference loop for the fine-tuned compliance-copilot.

Usage:
    python src/inference.py
    python src/inference.py --model ./checkpoints/final
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = Path("./checkpoints/final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading model from {args.model}...")
    raise NotImplementedError("Interactive inference loop not yet implemented.")


if __name__ == "__main__":
    main()
