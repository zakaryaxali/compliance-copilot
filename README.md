# compliance-copilot

A fine-tuned small language model that answers fintech compliance questions (PCI-DSS, AML/KYC, SOC 2) for engineering teams, built on top of an open Mistral model with LoRA.

## Why this project

Compliance questions slow engineering teams down. An engineer building a new payment flow needs to know whether storing a specific field triggers PCI scope, what KYC tier applies to a given transaction type, or how a control maps to a SOC 2 criterion. The current options are: ping the compliance team (slow, expensive), read 200 pages of standards (slow), or guess (dangerous). General-purpose LLMs answer these questions confidently but are often wrong on specifics — they confuse PCI-DSS versions, hallucinate control numbers, and mix up regulatory regimes.

A small, domain-specific model fine-tuned on real compliance documentation gives engineering teams a fast, grounded first-pass answer with clear citations back to the source standards. It does not replace the compliance team — it filters the obvious 80% of questions so the compliance team can focus on the genuinely ambiguous 20%.

This project also serves as a demonstration of end-to-end LLM fine-tuning: dataset construction, LoRA training, evaluation against a non-fine-tuned baseline, and inference deployment. The choice of Mistral as the base model is deliberate — small enough to train and serve on consumer hardware, strong enough to follow instructions reliably, and openly licensed.

## What it does

Given a question like:

> "If I log the last 4 digits of a card PAN to our application logs, does that put the logging system in PCI scope?"

The model returns a structured answer:

- **Short answer:** No, the last 4 digits of the PAN (with the rest masked or truncated) are not in themselves cardholder data that triggers storage requirements under PCI-DSS 4.0.1.
- **Citation:** PCI-DSS 4.0.1 Requirement 3.4 — Render PAN unreadable when stored, and 3.5 — PAN masking when displayed (max first six / last four).
- **Caveat:** Logging the full PAN, even truncated alongside other identifiers that could reconstruct it, may still trigger scope. Confirm with your QSA.

The structured format is part of the fine-tuning objective — the model is trained to always separate answer from citation from caveat, so downstream tooling can parse the output reliably.

## Scope

**In scope for v1:**
- PCI-DSS 4.0.1 (payment card data)
- AML/KYC fundamentals (FATF recommendations, US BSA basics)
- SOC 2 Type II (Trust Services Criteria, common controls)

**Out of scope:**
- Jurisdiction-specific interpretations (MAS, EU PSD2, etc.) — future work
- Legal advice of any kind
- Real-time regulatory updates — the model is a snapshot

## Approach

### Base model

Mistral 7B Instruct v0.3, chosen for: open weights, strong instruction following, and small enough to fine-tune locally with LoRA on Apple Silicon (4-bit quantized: ~4 GB). Mistral specifically (rather than Llama or Qwen) because I want hands-on familiarity with their architecture and tooling.

### Fine-tuning method

LoRA (Low-Rank Adaptation) via [`mlx-lm`](https://github.com/ml-explore/mlx-lm) — Apple Silicon-native fine-tuning on top of the 4-bit quantized base. Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`. Rank 16, alpha 32, dropout 0.05, top 16 transformer layers tuned, prompt loss masked. Learning rate **1e-4** (the more aggressive 2e-4 diverged at iter ~30 on a 4-bit base — see `results/eval_results.md`).

Why MLX over Unsloth or HF Trainer: zero CUDA dependency, runs natively on the same Apple Silicon used for inference (so v0 and v1 use the same toolchain end-to-end, satisfying the methodology's apples-to-apples requirement).

### Dataset

614 question-answer pairs, generated semi-synthetically:

1. **Source documents:** Public PCI-DSS 4.0.1 standard (v1 covers PCI only — FATF and SOC 2 are next).
2. **Indexing:** `src/build_source_index.py` parses the official PDF into 307 sub-requirements (`{id, title, parent, principal}`). This index is also used at eval time for the Tier 1 citation-exists check.
3. **Generation:** Llama 3.3 70B Instruct via Together produces K=2 engineer-style Q&A pairs per indexed requirement, returning structured fields (`question`, `short_answer`, `citation_text`, `caveat`). We assemble the final 3-part answer in code so format adherence is guaranteed.
4. **Filtering:** none yet. The smoke run after a prompt iteration confirmed 100% Tier 1 pass on the generated pairs.

Known-flawed approach — training on LLM-generated data has a ceiling, and pairs are at the requirement-title level (no nuance about substantive application). The eval surfaces both: v1 fixed two v0 hallucinations but introduced a new one. See `results/eval_results.md` for the per-question diff.

Cost: ~$1.50 for the full 614-pair generation via Together.

### Evaluation

Framework: strict priority ordering (Quality > Cost > Latency) with v0 (base Mistral, no fine-tune) as a **fixed reference snapshot**. Three quality tiers, escalated only when previous is inconclusive:

1. **Tier 1 — Mechanical** (binary, inline every eval pass): format-adherence regex, citation-section-exists check against the parsed source index, banned-phrase blocklist. Implemented.
2. **Tier 2 — Coverage**: numerator/denominator over the explicit in-scope topic list. Designed, not yet implemented.
3. **Tier 3 — LLM-as-judge**: fires only when Tier 1 + 2 are inconclusive. Not yet implemented.

Eval set: 20 hand-written hard cases (`data/eval/hard_cases.draft.jsonl`), weighted toward PCI-DSS 4.0.1 changes that became mandatory in 2025-03-31 (Magecart scripts, MFA-into-CDE, removable-media anti-malware, multi-tenant pen-test facilitation). These are the questions base Mistral is most likely to hallucinate on, because they're version-specific.

Full methodology in `results/eval_methodology.md`.

## Repository structure

```
compliance-copilot/
├── README.md
├── CLAUDE.md                       # Project conventions (3-part format, scope, eval rules)
├── data/
│   ├── sources/                    # Source standards index (PDF stays out of repo)
│   ├── synthetic/                  # pci_qa.jsonl — 614 generated Q&A pairs
│   └── eval/                       # hard_cases.draft.jsonl — 20 hand-written hard cases
├── src/
│   ├── build_source_index.py       # Parse PCI-DSS 4.0.1 PDF into a flat requirement index
│   ├── generate_dataset.py         # Synthetic Q&A pipeline (Together / Llama 70B)
│   ├── train.py                    # mlx-lm LoRA fine-tuning
│   ├── evaluate.py                 # Tier 1 scorer (format, citation-exists, banned phrases)
│   └── inference.py                # MLX inference with optional --adapter-path
├── configs/
│   └── lora_default.yaml           # Training hyperparams (rank 16, alpha 32, lr 1e-4)
├── results/
│   ├── eval_methodology.md         # Priority ordering, fixed v0, 3-tier framework
│   ├── eval_results.md             # v0 vs v1 numbers + per-question diff
│   ├── v0_predictions.jsonl        # Raw base Mistral outputs
│   ├── v0_tier1.jsonl              # Scored
│   ├── v1_predictions.jsonl        # Raw fine-tuned outputs
│   └── v1_tier1.jsonl              # Scored
├── notebooks/
├── requirements.txt
└── LICENSE
```

## Requirements

### Hardware

- **Training and inference**: Apple Silicon (M-series). v0 + v1 ran end-to-end on an M4 Pro with 24 GB unified memory. The 4-bit quantized base is ~4 GB; rank-16 LoRA training holds peak memory under 18 GB.
- A CUDA path would also work via Hugging Face Transformers + PEFT (not currently wired), but the tooling here is MLX-first.

### Software (`requirements.txt`)

Key deps (see `requirements.txt` for full list):

- `mlx-lm` — Apple Silicon fine-tuning + inference. Mac-only.
- `pypdf` — Parse the PCI-DSS source PDF into a structured index.
- `openai` — Used as an OpenAI-compatible client against Together's API for dataset generation.
- `pyyaml` — Read the LoRA config.

### API keys

Only one is required to reproduce v1 end-to-end:

- `TOGETHER_API_KEY` — for synthetic dataset generation via Llama 3.3 70B. Set in `.env` (gitignored). `.env.example` ships in the repo as a template.

### Estimated cost

| Step | Cost |
|---|---|
| Build source index from PCI-DSS PDF | $0 (local) |
| Generate 614 synthetic Q&A pairs (Together) | ~$1.50 |
| LoRA fine-tune (~25 min on M4 Pro) | $0 (local) |
| Run inference for the 20-question eval | $0 (local) |
| **Total to reproduce v1** | **~$1.50** |

## Quick start

```bash
git clone https://github.com/zakaryaxali/compliance-copilot
cd compliance-copilot
python -m venv .venv && .venv/bin/pip install -r requirements.txt mlx-lm
cp .env.example .env  # then fill in TOGETHER_API_KEY

# 1. Build the PCI-DSS source index (download the PDF from PCI SSC first;
#    see data/sources/README.md)
.venv/bin/python src/build_source_index.py /path/to/PCI-DSS-v4_0_1.pdf \
    --out data/sources/pci_dss_4_0_1.jsonl

# 2. Run the v0 baseline (base Mistral, no fine-tune)
.venv/bin/python src/inference.py \
    --eval-set data/eval/hard_cases.draft.jsonl \
    --backend mlx \
    --out results/v0_predictions.jsonl
.venv/bin/python src/evaluate.py \
    --predictions results/v0_predictions.jsonl \
    --index data/sources/pci_dss_4_0_1.jsonl \
    --out results/v0_tier1.jsonl

# 3. Generate the synthetic dataset (~$1.50, ~25 min via Together / Llama 70B)
.venv/bin/python src/generate_dataset.py \
    --index data/sources/pci_dss_4_0_1.jsonl --k 2 \
    --out data/synthetic/pci_qa.jsonl

# 4. LoRA fine-tune (~25 min on M4 Pro)
.venv/bin/python src/train.py \
    --data data/synthetic/pci_qa.jsonl \
    --config configs/lora_default.yaml \
    --out checkpoints/v1

# 5. Run v1 with the adapter
.venv/bin/python src/inference.py \
    --eval-set data/eval/hard_cases.draft.jsonl \
    --backend mlx \
    --adapter-path checkpoints/v1/adapter \
    --out results/v1_predictions.jsonl
.venv/bin/python src/evaluate.py \
    --predictions results/v1_predictions.jsonl \
    --index data/sources/pci_dss_4_0_1.jsonl \
    --out results/v1_tier1.jsonl
```

## Results

20 hand-written hard-case eval (`data/eval/hard_cases.draft.jsonl`), weighted to PCI-DSS 4.0.1 changes.

| Tier 1 metric | v0 (base) | v1 (LoRA) | Δ |
|---|---|---|---|
| Format adherence | 20/20 (100%) | 20/20 (100%) | — |
| Citation exists | 18/20 (90%) | 19/20 (95%) | +1 |
| Banned phrase clean | 20/20 (100%) | 20/20 (100%) | — |
| **Overall** | **18/20 (90%)** | **19/20 (95%)** | **+1** |

**Per-question diff**:
- `hc-005` (PAN in URL logs) — v0 invented Req 6.6, **v1 fixed**.
- `hc-020` (multi-tenant pen-test) — v0 invented Req 11.3.4, **v1 fixed**.
- `hc-014` (SMS MFA acceptable?) — v0 passed, **v1 regressed** (invented Req 8.2.10).
- 17 others — unchanged, both pass.

**Honest read**: net +1 question, but the failure surface shifted rather than shrunk. Both v0 → v1 fixes are on PCI-DSS 4.0.1-specific questions (the headline target), so this is real signal. The hc-014 regression shows that K=2-pairs-per-requirement-title teaches "cite a real-looking ID" but not which ID is *correct* for nuanced questions. Per methodology this is a *hold* (don't-regress-any-P1-metric), not a *ship*.

Full numbers, per-question scoring, and reproduction in `results/eval_results.md`.

## Limitations

This is a weekend project, not a production system. Known limitations:

- **Synthetic training data has a ceiling.** The model can only be as good as the LLM that generated its training pairs. Real expert-labeled data would beat this.
- **Citation accuracy is shallow.** The model learns to *cite* sections, not to *understand* them. It will sometimes cite plausibly-named but nonexistent sections. The eval catches this; users should still verify.
- **No retrieval.** A RAG system over the same source documents would likely beat a fine-tuned model on accuracy. The point of doing fine-tuning here is to learn the workflow, not to claim it's the best architecture for this problem. A v2 would combine both: fine-tune for format and domain tone, RAG for ground-truth citations.
- **Not legal advice.** Obviously. The model is a productivity tool for engineers, not a substitute for qualified compliance counsel.

## Why a small fine-tuned model instead of just prompting a big one?

Three reasons:

1. **Cost and latency at scale.** If this were embedded in an internal developer tool answering hundreds of questions a day, a 7B local model is dramatically cheaper than per-token API calls to a frontier model.
2. **Format reliability.** Fine-tuning teaches the model to *always* return the structured three-part output. Prompting a frontier model is reliable most of the time but not always; fine-tuning narrows the gap considerably, which matters when downstream tooling parses the output. The eval section quantifies this.
3. **Data residency.** A locally-deployable model lets a compliance-sensitive organization keep compliance questions off third-party APIs. Slightly ironic but real.

## License

MIT. Source standards used in dataset generation are public; consult the original publishers (PCI SSC, FATF, AICPA) for redistribution terms before reusing the generated dataset commercially.
