"""Run inference against Mistral to produce predictions for an eval set.

Usage:
    HF_TOKEN=hf_xxx python src/inference.py \\
        --eval-set data/eval/hard_cases.jsonl \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --out results/v0_predictions.jsonl

The Hugging Face Inference API is the only backend right now (no local
model download required). Get a token at:
    https://huggingface.co/settings/tokens

Input JSONL: one `{"id", "question", ...}` record per line.
Output JSONL: each input record plus
    {"answer", "model", "backend", "latency_seconds"}

Predictions feed directly into src/evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def load_env_file(path: Path = Path(".env")) -> None:
    """Minimal .env loader — no dependency. Existing env vars take precedence."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file()

SYSTEM_PROMPT = """\
You are a PCI-DSS 4.0.1 compliance assistant for fintech engineering teams.
Your role: give a direct, cited answer to a specific compliance question.

Respond in EXACTLY this three-part format, with these exact section labels:

**Short answer:** [1-2 sentences. State the answer directly. If the answer \
genuinely depends on context, say so here.]

**Citation:** [Cite the specific PCI-DSS 4.0.1 requirement number(s) — e.g., \
"Requirement 3.4.1" or "Requirements 3.5.1 and 3.5.1.1". Use real \
requirement IDs only.]

**Caveat:** [What could change the answer, or when to consult a QSA. If \
there is genuinely no caveat, write "none."]

Do not add a preamble, summary, or follow-up outside this three-part structure.\
"""


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def generate_hf_inference(
    question: str, model: str, max_tokens: int, temperature: float
) -> str:
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for the hf-inference backend. "
            "Install with `pip install huggingface_hub`."
        ) from exc
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN env var is required. Get a token at "
            "https://huggingface.co/settings/tokens."
        )
    client = InferenceClient(api_key=token)
    response = client.chat_completion(
        messages=build_messages(question),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


BACKENDS = {"hf-inference": generate_hf_inference}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-set", type=Path, required=True)
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument(
        "--out", type=Path, default=Path("results/v0_predictions.jsonl")
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--backend", default="hf-inference", choices=list(BACKENDS))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompts that would be sent without calling the model.",
    )
    return p.parse_args()


def load_questions(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    questions = load_questions(args.eval_set)
    print(f"Loaded {len(questions)} questions from {args.eval_set}")

    if args.dry_run:
        for q in questions:
            print(f"\n--- {q['id']} ---")
            for msg in build_messages(q["question"]):
                print(f"[{msg['role']}]")
                print(msg["content"])
        return

    generate = BACKENDS[args.backend]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict] = []

    for q in questions:
        t0 = time.time()
        try:
            answer = generate(
                q["question"], args.model, args.max_new_tokens, args.temperature
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR on {q['id']}: {exc}", file=sys.stderr)
            answer, err = "", str(exc)
        latency = round(time.time() - t0, 2)
        predictions.append(
            {
                **q,
                "answer": answer,
                "model": args.model,
                "backend": args.backend,
                "latency_seconds": latency,
                **({"error": err} if err else {}),
            }
        )
        status = "OK" if err is None else "ERR"
        print(f"  {q['id']:20s} {status:3s} {latency:5.1f}s")

    with args.out.open("w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"\nWrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
