"""Parse a PCI-DSS 4.0.1 PDF into a flat requirement index.

Usage:
    python src/build_source_index.py path/to/PCI-DSS-v4_0_1.pdf \\
        --out data/sources/pci_dss_4_0_1.jsonl

Output: one JSON object per line, e.g.
    {"id": "3.4.1", "title": "PAN is masked when displayed...", "parent": "3.4",
     "principal": "3", "principal_title": "Protect Stored Account Data",
     "source": "PCI-DSS-4.0.1"}

The index is consumed by:
  - Tier 1 citation-exists check in src/evaluate.py
  - Tier 2 coverage denominator (in-scope topic list)
  - Dataset generation grounding in src/generate_dataset.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pypdf

SOURCE_ID = "PCI-DSS-4.0.1"

# Principal requirements (1..12) from PCI-DSS 4.0.1 Table 1.
PRINCIPALS: dict[str, str] = {
    "1": "Install and Maintain Network Security Controls",
    "2": "Apply Secure Configurations to All System Components",
    "3": "Protect Stored Account Data",
    "4": "Protect Cardholder Data with Strong Cryptography During Transmission Over Open, Public Networks",
    "5": "Protect All Systems and Networks from Malicious Software",
    "6": "Develop and Maintain Secure Systems and Software",
    "7": "Restrict Access to System Components and Cardholder Data by Business Need to Know",
    "8": "Identify Users and Authenticate Access to System Components",
    "9": "Restrict Physical Access to Cardholder Data",
    "10": "Log and Monitor All Access to System Components and Cardholder Data",
    "11": "Test Security of Systems and Networks Regularly",
    "12": "Support Information Security with Organizational Policies and Programs",
}

# A requirement header looks like "3.4.1 PAN is masked when displayed (...)."
# The title may wrap across multiple PDF lines because of the column layout —
# we accumulate continuation lines until we hit a structural marker (bullet,
# next requirement ID, or a section header like "Customized Approach Objective").
REQUIREMENT_LINE_RE = re.compile(
    r"^(?P<id>\d{1,2}(?:\.\d{1,2}){1,3})\s+(?P<title>[A-Z].*)$"
)

# Lines that signal the title has ended.
TITLE_STOP_PREFIXES = (
    "•",
    "Customized Approach",
    "Applicability Notes",
    "Defined Approach Requirements",
    "Defined Approach Testing Procedures",
    "Note:",
    "Good Practice",
    "Purpose",
    "Definitions",
    "Examples",
    "Requirements and Testing",
    "(continued",
    "Payment Card Industry",
)

# Testing-procedure IDs look like 1.1.2.a / 12.3.1.b — they share the column with
# the requirement above them and we must not let them leak into the title.
TESTING_PROCEDURE_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){1,3}\.[a-z]\b")

# Page range to scan. Front matter and appendices have a different layout
# and don't contain normative requirements 1..12. Tuned against the
# June 2024 PCI-DSS 4.0.1 publication (397 pages).
BODY_START_PAGE = 40
BODY_END_PAGE = 333  # appendix A starts at page 334


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf", type=Path, help="Path to PCI-DSS 4.0.1 PDF")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/sources/pci_dss_4_0_1.jsonl"),
        help="Output JSONL path",
    )
    return p.parse_args()


def extract_text(pdf_path: Path, start: int, end: int) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    pages = reader.pages[start : min(end, len(reader.pages))]
    return "\n".join(page.extract_text() or "" for page in pages)


def clean_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", raw).strip()
    title = re.sub(r"\.{2,}.*$", "", title).strip()  # drop TOC dot-leaders if present
    return title


def is_title_stop(line: str) -> bool:
    if TESTING_PROCEDURE_RE.match(line):
        return True
    return any(line.startswith(p) for p in TITLE_STOP_PREFIXES)


def principal_of(req_id: str) -> str:
    return req_id.split(".", 1)[0]


def parent_of(req_id: str) -> str | None:
    parts = req_id.split(".")
    if len(parts) == 1:
        return None
    return ".".join(parts[:-1])


def parse_records(text: str) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    pending: dict | None = None
    title_parts: list[str] = []

    def finalize() -> None:
        nonlocal pending, title_parts
        if pending is None:
            return
        title = clean_title(" ".join(title_parts))
        if len(title) >= 8 and pending["id"] not in seen:
            pending["title"] = title
            seen[pending["id"]] = pending
        pending = None
        title_parts = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            finalize()
            continue
        match = REQUIREMENT_LINE_RE.match(line)
        if match and principal_of(match.group("id")) in PRINCIPALS:
            finalize()
            req_id = match.group("id")
            principal = principal_of(req_id)
            pending = {
                "id": req_id,
                "parent": parent_of(req_id),
                "principal": principal,
                "principal_title": PRINCIPALS[principal],
                "source": SOURCE_ID,
            }
            title_parts = [match.group("title")]
        elif pending is not None:
            if is_title_stop(line):
                finalize()
            else:
                title_parts.append(line)
    finalize()
    return seen


def build_index(pdf_path: Path) -> list[dict]:
    text = extract_text(pdf_path, BODY_START_PAGE, BODY_END_PAGE)
    seen = parse_records(text)
    return sorted(
        seen.values(),
        key=lambda r: tuple(int(p) for p in r["id"].split(".")),
    )


def write_jsonl(records: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    args = parse_args()
    records = build_index(args.pdf)
    write_jsonl(records, args.out)
    print(f"Wrote {len(records)} requirements to {args.out}")
    by_principal: dict[str, int] = {}
    for r in records:
        by_principal[r["principal"]] = by_principal.get(r["principal"], 0) + 1
    for p in sorted(by_principal, key=int):
        print(f"  Req {p}: {by_principal[p]:3d}  ({PRINCIPALS[p]})")


if __name__ == "__main__":
    main()
