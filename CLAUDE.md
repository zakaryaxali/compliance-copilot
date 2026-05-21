# CLAUDE.md

Project conventions for Claude (and human contributors) working in this repo.

## Output format is the product

Every model answer must follow this three-part structure:

- **Short answer:** one or two sentences, direct.
- **Citation:** specific section/requirement from the source standard.
- **Caveat:** what could change the answer, or when to escalate to a human.

Eval scripts regex-match this structure. Do not invent new fields, reorder, or drop any of the three. If a question genuinely has no caveat, write `Caveat: none.` — do not omit the line.

## Versions to pin

- **PCI-DSS 4.0.1** (not 4.0). Always 4.0.1 in answers, prompts, and synthetic data.
- **FATF 40 Recommendations** — November 2023 update.
- **SOC 2** — current AICPA Trust Services Criteria.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`.

## Citations are real or they are bugs

Never invent a section number. If you don't know the exact requirement, say "verify with the source standard" in the caveat instead of guessing. The eval has a citation-existence check specifically to catch hallucinated sections — fabricating one defeats the project's whole premise.

## Scope discipline

In scope: PCI-DSS 4.0.1, AML/KYC fundamentals (FATF, US BSA), SOC 2 Type II.

Out of scope — do not extend the dataset or model into these without an explicit ask:
- Jurisdiction-specific interpretations (MAS, EU PSD2, GDPR, UK FCA, etc.)
- Legal advice framing of any kind
- Real-time regulatory updates

## Project ethos

This is a portfolio/weekend project. Favor:
- Small, working stubs over framework scaffolding.
- Direct edits over abstractions for hypothetical future flexibility.
- Honest limitations in the README over optimistic claims.

Avoid adding orchestration frameworks, plugin systems, or config layers the current scope doesn't need.

## Code conventions

- Python 3.11+, argparse for CLIs, YAML for configs, JSONL for datasets.
- Each `src/*.py` script is independently runnable with `python src/<script>.py --help`.
- Training, eval, and inference must all read the same dataset schema.
