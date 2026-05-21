"""Generate semi-synthetic Q&A pairs from compliance source documents.

Usage:
    python src/generate_dataset.py --source pci-dss --n 300

Pipeline:
    1. Load source standard (PCI-DSS / FATF / SOC 2) from data/sources/.
    2. Chunk into sections.
    3. For each section, ask a large model to generate engineer-style
       questions plus structured (short answer / citation / caveat) responses.
    4. Write JSONL to data/synthetic/<source>.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES = {"pci-dss", "fatf", "soc2", "all"}
DEFAULT_OUTPUT_DIR = Path("data/synthetic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, choices=sorted(SOURCES))
    p.add_argument("--n", type=int, default=300, help="Number of Q&A pairs to generate")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default="claude-opus-4-7", help="Generation model")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Dataset generation pipeline not yet implemented.")


if __name__ == "__main__":
    main()
