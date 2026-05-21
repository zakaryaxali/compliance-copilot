"""Score model answers against the compliance-copilot evaluation tiers.

Tier 1 (implemented): mechanical checks — 3-part structure, citation existence,
banned phrases. Pure scoring, no model inference.

Tier 2 (stub): in-scope coverage over the indexed source standards.
Tier 3 (stub): LLM-as-judge for inconclusive cases.

See `results/eval_methodology.md` for the full framework.

Usage:
    python src/evaluate.py \\
        --predictions data/eval/v0_predictions.jsonl \\
        --index data/sources/pci_dss_4_0_1.jsonl \\
        --out results/v0_tier1.jsonl

Predictions JSONL: one `{"id", "question", "answer"}` record per line.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Phrases the model must never produce. The compliance-copilot pitch is to
# give a direct, cited answer — generic LLM hedging defeats the whole point.
BANNED_PATTERNS = [
    r"\bas an ai\b",
    r"\bi cannot provide legal advice\b",
    r"\bi am not (a lawyer|qualified)\b",
    r"\bi'?m not (a lawyer|qualified)\b",
    r"\bplease consult (a |an |your )?(professional|lawyer|attorney)\b",
]
BANNED_RE = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)

# The three-part structure required by CLAUDE.md. Permissive about markdown
# (allow optional bold around the label), strict about presence and ordering.
SHORT_ANSWER_RE = re.compile(r"(?:\*\*)?\s*short answer\s*(?:\*\*)?\s*:", re.IGNORECASE)
CITATION_RE = re.compile(r"(?:\*\*)?\s*citation\s*(?:\*\*)?\s*:", re.IGNORECASE)
CAVEAT_RE = re.compile(r"(?:\*\*)?\s*caveat\s*(?:\*\*)?\s*:", re.IGNORECASE)

# Requirement IDs like 3.4, 3.4.1, 12.10.5. Two to four dot-separated numbers.
CITATION_ID_RE = re.compile(r"\b(\d{1,2}(?:\.\d{1,2}){1,3})\b")


@dataclass
class Tier1Score:
    format_pass: bool
    citation_exists_pass: bool
    banned_phrase_pass: bool
    tier1_pass: bool
    extracted_citations: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)


def parse_structure(answer: str) -> dict | None:
    """Return {short_answer, citation, caveat} or None if malformed."""
    sa = SHORT_ANSWER_RE.search(answer)
    ct = CITATION_RE.search(answer)
    cv = CAVEAT_RE.search(answer)
    if not (sa and ct and cv):
        return None
    if not (sa.start() < ct.start() < cv.start()):
        return None
    return {
        "short_answer": answer[sa.end() : ct.start()].strip(),
        "citation": answer[ct.end() : cv.start()].strip(),
        "caveat": answer[cv.end() :].strip(),
    }


def _is_plausible_requirement_id(req_id: str) -> bool:
    # Filters version strings like "4.0.1" — real PCI sub-requirements never
    # use 0 as a component (they go 1.1, 1.2, never 1.0).
    return all(int(p) > 0 for p in req_id.split("."))


def check_citation_exists(
    citation_text: str, valid_ids: set[str]
) -> tuple[list[str], list[str]]:
    """Return (extracted, missing). All extracted IDs must exist for a pass."""
    extracted = sorted(
        {
            req_id
            for req_id in CITATION_ID_RE.findall(citation_text)
            if _is_plausible_requirement_id(req_id)
        }
    )
    missing = [i for i in extracted if i not in valid_ids]
    return extracted, missing


def check_banned_phrases(answer: str) -> bool:
    return BANNED_RE.search(answer) is None


def score_tier1(answer: str, valid_ids: set[str]) -> Tier1Score:
    parts = parse_structure(answer)
    format_pass = parts is not None

    failed: list[str] = []
    extracted: list[str] = []
    missing: list[str] = []
    citation_pass = False

    if not format_pass:
        failed.append("format")
    else:
        extracted, missing = check_citation_exists(parts["citation"], valid_ids)
        if not extracted:
            failed.append("citation_no_id_found")
        elif missing:
            failed.append(f"citation_missing:{','.join(missing)}")
        else:
            citation_pass = True

    banned_pass = check_banned_phrases(answer)
    if not banned_pass:
        failed.append("banned_phrase")

    return Tier1Score(
        format_pass=format_pass,
        citation_exists_pass=citation_pass,
        banned_phrase_pass=banned_pass,
        tier1_pass=format_pass and citation_pass and banned_pass,
        extracted_citations=extracted,
        missing_citations=missing,
        failed_checks=failed,
    )


def load_index(path: Path) -> set[str]:
    with path.open() as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def summarize(scored: list[dict]) -> str:
    n = len(scored)
    if n == 0:
        return "no predictions scored"

    def pct(count: int) -> str:
        return f"{count}/{n}  ({100 * count / n:5.1f}%)"

    fmt = sum(1 for r in scored if r["scores"]["format_pass"])
    cit = sum(1 for r in scored if r["scores"]["citation_exists_pass"])
    ban = sum(1 for r in scored if r["scores"]["banned_phrase_pass"])
    overall = sum(1 for r in scored if r["scores"]["tier1_pass"])
    return (
        f"n = {n}\n"
        f"  format adherence:     {pct(fmt)}\n"
        f"  citation exists:      {pct(cit)}\n"
        f"  banned phrase clean:  {pct(ban)}\n"
        f"  Tier 1 overall:       {pct(overall)}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True, help="Source standards JSONL")
    p.add_argument(
        "--out", type=Path, default=Path("results/tier1_scored.jsonl")
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    valid_ids = load_index(args.index)
    preds = load_jsonl(args.predictions)

    scored = [
        {**rec, "scores": asdict(score_tier1(rec["answer"], valid_ids))}
        for rec in preds
    ]
    write_jsonl(scored, args.out)

    print(summarize(scored))
    print(f"\nDetails: {args.out}")


if __name__ == "__main__":
    main()
