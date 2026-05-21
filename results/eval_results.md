# Evaluation results

Per the methodology in `results/eval_methodology.md`. v0 is the fixed reference snapshot — base Mistral 7B Instruct v0.3, no fine-tuning. Versions in this table are not floating with upstream Mistral.

## Tier 1 results

| Metric | v0 | v1 | … |
|---|---|---|---|
| Format adherence | 20/20 (100.0%) | — | — |
| Citation exists (PCI-DSS 4.0.1 index) | 18/20 (90.0%) | — | — |
| Banned phrase clean | 20/20 (100.0%) | — | — |
| **Tier 1 overall** | **18/20 (90.0%)** | — | — |

## Latency and cost

| Metric | v0 |
|---|---|
| Inference backend | MLX local (Apple Silicon, M4 Pro 24 GB) |
| Model | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| Per-question latency (avg) | ~3.5 s |
| Per-question latency (range) | 2.2 s – 6.5 s |
| Total wall time (20 questions) | ~70 s |
| Cost per run | $0 (local) |

## v0 failure analysis

**Format adherence 100%** — the system prompt holds the 3-part structure reliably even on the base model. Format-only failure mode is not a useful fine-tuning target; the model already learns this from the prompt.

**Citation existence 90%** — two failures, both real hallucinations:

| Case | Hallucinated cite | Reality | Impact |
|---|---|---|---|
| hc-005 (PAN in URL logs) | Req 6.6 ("Store only hashed non-sensitive data elements in logs") | Req 6.6 does not exist in PCI-DSS 4.0.1 (4.0 reorganized away from it; payment-page rules moved to 6.4.x) | Fabricated ID + fabricated text |
| hc-020 (multi-tenant pen test) | Req 11.3.4 ("Service providers must not store SAD…") | Req 11.3.4 is about external vulnerability scans; the real answer is Req 11.4.7 (new in 4.0.1: providers MUST facilitate customer pen testing) | Wrong substantive answer + invented citation on a 4.0.1-specific change |

Both failures are on **PCI-DSS 4.0.1-specific or version-aware questions** — exactly the class the fine-tuning is designed to fix. The model is confidently wrong, not hedging. Banned-phrase check at 100% means the model isn't dodging via "consult a professional" hedges — it commits to wrong answers, which is the harder failure mode.

## What's still untested

- **Tier 2 coverage** — denominator/numerator over the indexed in-scope topic set. Not yet implemented.
- **Tier 3 LLM-as-judge** — only fires when Tiers 1+2 are inconclusive. Not yet implemented.
- **N=3 seed sweep** — v0 above is a single run. Per methodology, ship-candidate configs need ≥3 seeds. Base-model inference with temperature 0.2 is roughly deterministic, so seed variance is low here, but a follow-up rerun is needed before treating these numbers as a contract.
- **`expected_citations` correctness** — the per-question `expected_citations` field in `data/eval/hard_cases.draft.jsonl` is not yet used in scoring. Tier 2 / Tier 3 will consume it.

## Reproducing

```bash
# Once-only setup
python -m venv .venv && .venv/bin/pip install -r requirements.txt mlx-lm
python src/build_source_index.py /path/to/PCI-DSS-v4_0_1.pdf \
    --out data/sources/pci_dss_4_0_1.jsonl

# Run v0
python src/inference.py \
    --eval-set data/eval/hard_cases.draft.jsonl \
    --backend mlx \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --out results/v0_predictions.jsonl

# Score
python src/evaluate.py \
    --predictions results/v0_predictions.jsonl \
    --index data/sources/pci_dss_4_0_1.jsonl \
    --out results/v0_tier1.jsonl
```
