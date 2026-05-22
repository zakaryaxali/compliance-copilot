"""LoRA fine-tune base Mistral on synthetic Q&A pairs via mlx-lm.

Usage:
    python src/train.py \\
        --data data/synthetic/pci_qa.jsonl \\
        --config configs/lora_default.yaml \\
        --out checkpoints/v1

Pipeline:
  1. Load synthetic pairs.
  2. Convert each pair into the chat-format mlx-lm expects, folding the
     SYSTEM_PROMPT into the user message (Mistral's chat template rejects
     a standalone system role).
  3. Split into train/valid JSONL files in <out>/data/.
  4. Invoke mlx-lm's lora command with hyperparameters from the YAML config.
  5. Adapter weights land at <out>/adapter/.

Trained adapter feeds back into src/inference.py via --adapter-path for the
v1 eval.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import (  # noqa: E402
    MLX_MODEL_MAP,
    SYSTEM_PROMPT,
    load_env_file,
)

load_env_file()


def to_chat_format(pair: dict) -> dict:
    user_content = f"{SYSTEM_PROMPT}\n\nQuestion: {pair['question']}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": pair["answer"]},
        ]
    }


def prepare_data(
    pairs_path: Path, data_dir: Path, val_frac: float, seed: int
) -> tuple[int, int]:
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * val_frac)))
    val = pairs[:n_val]
    train = pairs[n_val:]
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("valid", val)):
        with (data_dir / f"{name}.jsonl").open("w") as f:
            for pair in subset:
                f.write(json.dumps(to_chat_format(pair)) + "\n")
    return len(train), len(val)


def resolve_model_id(base: str) -> str:
    return MLX_MODEL_MAP.get(base, base)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Synthetic pairs JSONL")
    p.add_argument(
        "--config", type=Path, default=Path("configs/lora_default.yaml")
    )
    p.add_argument("--out", type=Path, default=Path("checkpoints/v1"))
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare data + print the mlx-lm command without running it.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    data_dir = args.out / "data"
    adapter_dir = args.out / "adapter"
    n_train, n_val = prepare_data(args.data, data_dir, args.val_frac, args.seed)
    print(f"Prepared {n_train} train / {n_val} valid examples in {data_dir}")

    epochs = cfg["training"]["num_train_epochs"]
    batch = cfg["training"]["per_device_train_batch_size"]
    iters = max(1, math.ceil(n_train * epochs / batch))

    lora_params = {
        "rank": cfg["lora"]["r"],
        "scale": cfg["lora"]["alpha"] / cfg["lora"]["r"],
        "dropout": cfg["lora"]["dropout"],
    }

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", resolve_model_id(cfg["base_model"]),
        "--train",
        "--data", str(data_dir),
        "--fine-tune-type", "lora",
        "--num-layers", "16",
        "--batch-size", str(batch),
        "--iters", str(iters),
        "--learning-rate", str(cfg["training"]["learning_rate"]),
        "--adapter-path", str(adapter_dir),
        "--mask-prompt",
        "--max-seq-length", str(cfg["max_seq_length"]),
        "--seed", str(cfg["training"]["seed"]),
        "--save-every", "100",
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
    ]

    # mlx-lm reads lora_parameters from a YAML config file, not the CLI.
    # Write one alongside the data so the run is reproducible.
    run_config = args.out / "mlx_lora.yaml"
    with run_config.open("w") as f:
        yaml.safe_dump({"lora_parameters": lora_params}, f)
    cmd.extend(["-c", str(run_config)])

    print("\n$ " + " ".join(cmd) + "\n")
    if args.dry_run:
        return
    adapter_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    print(f"\nAdapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
