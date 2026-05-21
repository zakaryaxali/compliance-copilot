"""Generate synthetic Q&A pairs from the indexed PCI-DSS 4.0.1 requirements.

Usage:
    python src/generate_dataset.py \\
        --index data/sources/pci_dss_4_0_1.jsonl \\
        --sample 10 --k 2 \\
        --out data/synthetic/pci_qa.jsonl

Pipeline:
  1. Load source requirements from the index.
  2. (Optional) sample N requirements for a fast first pass.
  3. For each requirement, prompt the generator model to produce K
     engineer-style questions whose right answer cites that requirement.
  4. The generator emits structured fields (question / short_answer /
     citation_text / caveat). We assemble the final 3-part answer in code
     so it always matches the regex used by Tier 1.

Default generator: Llama 3.3 70B Instruct via Together (chosen for cost
and to avoid self-distilling against the Mistral base we'll fine-tune).

Output JSONL: one Q&A pair per line, schema:
    {
      "question": str,
      "answer": str,                  # assembled 3-part format
      "source_requirement_id": str,   # e.g., "3.4.1"
      "source_requirement_title": str,
      "expected_citations": [str],
      "generator_model": str,
    }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Re-use the env loader from inference.py so we read TOGETHER_API_KEY/SSL vars.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import load_env_file  # noqa: E402

load_env_file()


DEFAULT_GENERATOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

GENERATOR_PROMPT_TEMPLATE = """\
You are creating training data for a PCI-DSS 4.0.1 compliance assistant.

Given one PCI-DSS 4.0.1 requirement, generate {k} realistic engineer-style \
questions whose correct answer would cite THIS specific requirement.

REQUIREMENT:
  ID: {req_id}
  Title: {req_title}
  Principal: Requirement {principal} — {principal_title}

Rules for the questions:
  - Sound like real engineering questions (concrete systems, code, infra)
  - Vary the scenarios (storage, network, access, key management, …)
  - Stay grounded so {req_id} is genuinely the right citation
  - Avoid generic / academic phrasing
  - Mix easy and hard ones (some "it depends", some direct)

For each question, write a structured answer in these fields:
  - short_answer: 1–2 sentences, direct. If the answer depends on context, say so.
  - citation_text: "Requirement {req_id} — <brief restatement of the rule>"
  - caveat: what could change the answer, or "none."

IMPORTANT:
  - Cite ONLY real PCI-DSS 4.0.1 requirement IDs. Never invent numbers.
  - Use the exact ID {req_id} in citation_text.

Return JSON with this exact schema:
{{
  "pairs": [
    {{
      "question": "...",
      "short_answer": "...",
      "citation_text": "Requirement {req_id} — ...",
      "caveat": "..."
    }},
    ...
  ]
}}
"""


def assemble_answer(short_answer: str, citation_text: str, caveat: str) -> str:
    return (
        f"**Short answer:** {short_answer.strip()}\n\n"
        f"**Citation:** {citation_text.strip()}\n\n"
        f"**Caveat:** {caveat.strip()}"
    )


def generate_pairs_for_requirement(
    req: dict, k: int, client, model: str
) -> list[dict]:
    prompt = GENERATOR_PROMPT_TEMPLATE.format(
        k=k,
        req_id=req["id"],
        req_title=req["title"],
        principal=req["principal"],
        principal_title=req["principal_title"],
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.6,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ! {req['id']}: non-JSON response, skipping", file=sys.stderr)
        return []
    pairs: list[dict] = []
    for item in data.get("pairs", []):
        try:
            pair = {
                "question": item["question"],
                "answer": assemble_answer(
                    item["short_answer"], item["citation_text"], item["caveat"]
                ),
                "source_requirement_id": req["id"],
                "source_requirement_title": req["title"],
                "expected_citations": [req["id"]],
                "generator_model": model,
            }
            pairs.append(pair)
        except KeyError as exc:
            print(
                f"  ! {req['id']}: pair missing field {exc}, skipping",
                file=sys.stderr,
            )
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample N requirements at random (0 = all)",
    )
    p.add_argument("--k", type=int, default=2, help="Pairs per requirement")
    p.add_argument(
        "--out", type=Path, default=Path("data/synthetic/pci_qa.jsonl")
    )
    p.add_argument("--model", default=DEFAULT_GENERATOR_MODEL)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set (check .env)")
    client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")

    with args.index.open() as f:
        requirements = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(requirements)} requirements from {args.index}")

    if args.sample > 0:
        random.Random(args.seed).shuffle(requirements)
        requirements = requirements[: args.sample]
        print(f"Sampled {len(requirements)} for generation")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    with args.out.open("w") as out_f:
        for i, req in enumerate(requirements, 1):
            t0 = time.time()
            pairs = generate_pairs_for_requirement(
                req, args.k, client, args.model
            )
            for pair in pairs:
                out_f.write(json.dumps(pair) + "\n")
            total_pairs += len(pairs)
            dt = time.time() - t0
            print(
                f"  [{i:3d}/{len(requirements)}] {req['id']:8s} "
                f"-> {len(pairs)} pairs in {dt:4.1f}s"
            )

    print(f"\nWrote {total_pairs} pairs to {args.out}")


if __name__ == "__main__":
    main()
