# Evaluation results

Per the methodology in `results/eval_methodology.md`. v0 is the fixed reference snapshot — base Mistral 7B Instruct v0.3, no fine-tuning. Versions in this table are not floating with upstream Mistral.

## Tier 1 results

| Metric | v0 | v1 | Δ |
|---|---|---|---|
| Format adherence | 20/20 (100.0%) | 20/20 (100.0%) | — |
| Citation exists (PCI-DSS 4.0.1 index) | 18/20 (90.0%) | 19/20 (95.0%) | +1 |
| Banned phrase clean | 20/20 (100.0%) | 20/20 (100.0%) | — |
| **Tier 1 overall** | **18/20 (90.0%)** | **19/20 (95.0%)** | **+1** |

## v0 → v1 per-question diff

| | v0 | v1 | Change |
|---|---|---|---|
| hc-005 (PAN in URL logs) | ✗ cited fake Req 6.6 | ✓ | **fixed** |
| hc-014 (SMS MFA acceptable?) | ✓ | ✗ cited fake Req 8.2.10 | **regression** |
| hc-020 (multi-tenant pen test) | ✗ cited fake Req 11.3.4 | ✓ | **fixed** |
| (other 17) | ✓ | ✓ | — |

Net +1, but the failure mode shifted rather than shrunk. v1 still hallucinates citation IDs — just on different questions than v0.

## Latency and cost

| Metric | v0 | v1 |
|---|---|---|
| Inference backend | MLX local (M4 Pro 24 GB) | MLX local (M4 Pro 24 GB) |
| Model | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | same + LoRA adapter (rank 16, α 32) |
| Per-question latency (avg) | ~3.5 s | ~3.5 s |
| Per-question latency (range) | 2.2 s – 6.5 s | 2.8 s – 17.3 s (first-call warm-up included) |
| Total wall time (20 questions) | ~70 s | ~80 s |
| Inference cost per run | $0 | $0 |

## Training cost (v1)

| | |
|---|---|
| Generator (Llama 3.3 70B via Together) | ~$1.50 for 614 synthetic Q&A pairs |
| Fine-tune (mlx-lm, local) | $0, ~25 min wall to iter 200 |
| Initial LR (2e-4) | Diverged at iter 30 — killed, retried |
| Effective LR (1e-4) | Trained cleanly. Best val loss at iter 200 (0.791, vs starting 1.707) |
| Stop reason | Val loss plateaued (iter 150: 0.790, iter 200: 0.791) while train kept descending — overfit signal |

## Verdict

Per the methodology:
- **P1 Quality**: passed at the *aggregate* level (+1 question, no regression on banned-phrase or format), but there is a per-question regression on hc-014. By the strict reading ("don't regress *any* P1 metric"), this is a hold, not a ship.
- **P2 Cost**: ~$1.50 one-time + $0 ongoing, neutral vs v0 ($0).
- **P3 Latency**: unchanged at ~3.5 s/q.

**Honest take**: the fine-tune shuffled the failure surface more than it shrunk it. The two failures it fixed (hc-005, hc-020) were both on PCI-DSS 4.0.1-specific questions, which is the headline target of the project — so this is real signal. But hc-014 is a regression and the same root cause (citation hallucination) is unaddressed.

## What v1 learned (and didn't)

**Learned** — from 614 synthetic pairs anchored on real requirement IDs:
- "Always cite a real-looking requirement ID" → fixed the totally-invented Req 6.6 in hc-005
- "Stay within the 12 principal requirement space" → no more wild numbers like 99.9.9

**Didn't learn** — because the synthetic dataset was K=2 pairs at the requirement-title level:
- The *substantive content* of each requirement well enough to pick the right ID for nuanced questions
- The distinction between similar-looking IDs (8.2.x vs 8.4.x) for MFA-flavored questions

## What's still untested

- **Tier 2 coverage** — would catch hc-014 not as "ID hallucinated" but as "cited the wrong real ID." Not yet implemented; would tighten the v1 verdict.
- **Tier 3 LLM-as-judge** — designed but not invoked.
- **N=3 seed sweep** — v1 above is a single training run. Per methodology, ship-candidate configs need ≥3 seeds. The training has a fixed seed (42); rerunning with seeds 7 and 13 is the next pass.
- **Held-out synthetic test set** — only the 20 hand-written hard cases were used for evaluation. A larger held-out from the synthetic dataset would reduce noise.

## Reproducing

```bash
# Once-only setup
python -m venv .venv && .venv/bin/pip install -r requirements.txt mlx-lm
python src/build_source_index.py /path/to/PCI-DSS-v4_0_1.pdf \
    --out data/sources/pci_dss_4_0_1.jsonl

# v0
python src/inference.py \
    --eval-set data/eval/hard_cases.draft.jsonl \
    --backend mlx \
    --out results/v0_predictions.jsonl
python src/evaluate.py \
    --predictions results/v0_predictions.jsonl \
    --index data/sources/pci_dss_4_0_1.jsonl \
    --out results/v0_tier1.jsonl

# v1 — generate dataset, fine-tune, eval
python src/generate_dataset.py \
    --index data/sources/pci_dss_4_0_1.jsonl --k 2 \
    --out data/synthetic/pci_qa.jsonl                    # ~$1.50 via Together
python src/train.py \
    --data data/synthetic/pci_qa.jsonl \
    --config configs/lora_default.yaml \
    --out checkpoints/v1                                  # ~25 min on M4 Pro
python src/inference.py \
    --eval-set data/eval/hard_cases.draft.jsonl \
    --backend mlx \
    --adapter-path checkpoints/v1/adapter \
    --out results/v1_predictions.jsonl
python src/evaluate.py \
    --predictions results/v1_predictions.jsonl \
    --index data/sources/pci_dss_4_0_1.jsonl \
    --out results/v1_tier1.jsonl
```
