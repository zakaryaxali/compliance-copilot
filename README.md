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

- **Short answer:** No, the last 4 digits are not considered sensitive cardholder data under PCI-DSS 4.0.1.
- **Citation:** PCI-DSS 4.0.1 Requirement 3.3 — Sensitive Authentication Data.
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

Mistral 7B Instruct v0.3, chosen for: open weights, strong instruction following, small enough to fine-tune with LoRA on a single A100 or two consumer GPUs, and architectural alignment with the company whose role I'm targeting.

### Fine-tuning method

LoRA (Low-Rank Adaptation) using Unsloth for ~2x training speed and lower memory. Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`. Rank 16, alpha 32, dropout 0.05. These are conservative defaults — rank 16 is enough to learn structured-output formatting and domain vocabulary without overwriting the base model's general reasoning.

### Dataset

Roughly 500-1000 question-answer pairs, generated semi-synthetically:

1. **Source documents:** Public PCI-DSS 4.0.1 standard, FATF 40 Recommendations, AICPA SOC 2 Trust Services Criteria.
2. **Question generation:** A larger model (Claude or GPT-4) generates realistic engineer-style questions from each source section.
3. **Answer generation:** The same larger model writes structured answers (short answer / citation / caveat format) grounded in the source.
4. **Human review:** A sample (~10%) is manually reviewed for accuracy before training. Bad pairs are filtered.

This is a known-flawed approach — training on LLM-generated data has limits — and the README documents it as such. The point is to show the workflow, not to ship a production compliance tool.

### Evaluation

Three eval sets:

1. **Held-out synthetic test set** (100 pairs): measures whether the model learned the structured-output format and domain vocabulary.
2. **Hand-written hard cases** (30 questions): real-feeling edge cases — questions where the right answer is "it depends" or "consult your QSA." Measures whether the model hedges appropriately instead of confidently hallucinating.
3. **Baseline comparison:** same prompts run against (a) base Mistral 7B Instruct without fine-tuning, and (b) a frontier model. The fine-tuned model should beat base Mistral on format adherence and domain accuracy, and approach the frontier model on the easier questions.

Metrics: format adherence (regex-checkable — does it return the three-part structure?), citation accuracy (does the cited section actually exist in the source standard?), and a small LLM-as-judge eval for answer quality.

## Repository structure

```
compliance-copilot/
├── README.md
├── data/
│   ├── sources/              # Public standards (PCI, FATF, SOC 2)
│   ├── synthetic/            # Generated Q&A pairs
│   └── eval/                 # Held-out test sets
├── src/
│   ├── generate_dataset.py   # Synthetic data pipeline
│   ├── train.py              # LoRA fine-tuning
│   ├── evaluate.py           # All three eval sets
│   └── inference.py          # Loading + generation
├── notebooks/
│   └── exploration.ipynb     # EDA and prompt iteration
├── results/
│   ├── eval_results.md       # Numbers + analysis
│   └── example_outputs.md    # Side-by-side base vs fine-tuned
├── requirements.txt
└── LICENSE
```

## Requirements

### Hardware

- Training: 1x A100 40GB (Modal, Lambda Labs, or RunPod) for ~2-3 hours, or 1x consumer GPU with 24GB+ VRAM (e.g., 3090/4090) with smaller batch size.
- Inference: any machine that can run Mistral 7B in 4-bit (~6GB VRAM) — fine on a MacBook M-series with MLX or a single consumer GPU.

### Software (`requirements.txt`)

```
# Core ML stack
torch>=2.1.0
transformers>=4.40.0
datasets>=2.18.0
accelerate>=0.29.0
peft>=0.10.0
bitsandbytes>=0.43.0

# Fine-tuning speedup
unsloth @ git+https://github.com/unslothai/unsloth.git

# Dataset generation
anthropic>=0.25.0
openai>=1.20.0
tqdm>=4.66.0

# Evaluation
scikit-learn>=1.4.0
pandas>=2.2.0

# Document parsing for source standards
pypdf>=4.0.0
beautifulsoup4>=4.12.0

# Inference + serving (optional)
vllm>=0.4.0
fastapi>=0.110.0
uvicorn>=0.29.0
```

### API keys

- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — for synthetic dataset generation and LLM-as-judge evaluation. Both are optional if you bring your own data.

### Estimated cost

- Compute: $5-15 on Modal/RunPod for one full training run.
- Dataset generation: ~$2-5 in Claude or OpenAI API calls for 1000 Q&A pairs.

## Quick start

```bash
git clone https://github.com/zakaryaxali/compliance-copilot
cd compliance-copilot
pip install -r requirements.txt

# 1. Generate the dataset (or skip and use the included sample)
python src/generate_dataset.py --source pci-dss --n 300

# 2. Train
python src/train.py --config configs/lora_default.yaml

# 3. Evaluate
python src/evaluate.py --model ./checkpoints/final --eval-set all

# 4. Try it interactively
python src/inference.py
```

## Results

*To be filled in after the first training run. Will include: format adherence %, citation accuracy %, side-by-side outputs vs base Mistral, and honest notes on failure modes.*

## Limitations

This is a weekend project, not a production system. Known limitations:

- **Synthetic training data has a ceiling.** The model can only be as good as the LLM that generated its training pairs. Real expert-labeled data would beat this.
- **Citation accuracy is shallow.** The model learns to *cite* sections, not to *understand* them. It will sometimes cite plausibly-named but nonexistent sections. The eval catches this; users should still verify.
- **No retrieval.** A RAG system over the same source documents would likely beat a fine-tuned model on accuracy. The point of doing fine-tuning here is to learn the workflow, not to claim it's the best architecture for this problem. A v2 would combine both: fine-tune for format and domain tone, RAG for ground-truth citations.
- **Not legal advice.** Obviously. The model is a productivity tool for engineers, not a substitute for qualified compliance counsel.

## Why a small fine-tuned model instead of just prompting a big one?

Three reasons:

1. **Cost and latency at scale.** If this were embedded in an internal developer tool answering hundreds of questions a day, a 7B local model is dramatically cheaper than per-token API calls to a frontier model.
2. **Format reliability.** Fine-tuning teaches the model to *always* return the structured three-part output. Prompting a frontier model gets you ~95% reliability; fine-tuning gets you closer to 99%, which matters when downstream tooling parses the output.
3. **Data residency.** A locally-deployable model lets a compliance-sensitive organization keep compliance questions off third-party APIs. Slightly ironic but real.

## License

MIT. Source standards used in dataset generation are public; consult the original publishers (PCI SSC, FATF, AICPA) for redistribution terms before reusing the generated dataset commercially.
