"""LoRA fine-tuning entry point.

Usage:
    python src/train.py --config configs/lora_default.yaml

Loads a base Mistral model with Unsloth, attaches LoRA adapters per the
config, and trains on the JSONL dataset referenced by the config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume-from", type=Path, default=None)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(f"Loaded config from {args.config}: base_model={config['base_model']}")
    raise NotImplementedError("LoRA training loop not yet implemented.")


if __name__ == "__main__":
    main()
