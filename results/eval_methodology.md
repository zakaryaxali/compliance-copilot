# Evaluation methodology

How performance of compliance-copilot is measured. The goal is to make every "we improved X" claim defensible — and every regression visible — before it ships.

## Priorities (strict ordering)

| | Priority | Rule |
|---|---|---|
| P1 | **Quality** | **don't regress** — overrides cost and latency |
| P2 | Cost | reduce (training $ per version + inference $/query) |
| P3 | Latency | reduce (time-to-first-token, tokens/sec) |

A change that improves cost or latency **but regresses any P1 metric** is rejected. A change that improves quality at the cost of money or latency may ship — depending on the size of the move.

This ordering is the reason the measurement system exists. Without it, "cheaper" could mean "silently stopped citing real sections."

## v0 — the fixed reference snapshot

**v0 = base Mistral 7B Instruct v0.3, no fine-tuning, scored against all three eval sets at a frozen commit on a frozen day.**

v0 is fixed, not floating with upstream. "Are we still ahead of current Mistral?" is a separate measurement campaign (call it `upstream-HEAD`), not a redefinition of v0.

## How quality is measured — three tiers

Escalate to the next tier only when the previous one is inconclusive.

### Tier 1 — Mechanical (inline, every eval pass)

All binary pass/fail. Cheap. Runs on every generated answer.

| Gate | Floor | Check |
|---|---|---|
| 3-part structure (short answer / citation / caveat) | 100% | regex |
| Cited section exists in source standard | 100% | lookup against indexed PCI-DSS / FATF / SOC 2 |
| JSON-parseable output (when invoked with `--json`) | 100% | `json.loads` |
| No banned phrases (e.g., "as an AI", "I cannot provide legal advice") | 100% | regex blocklist |

### Tier 2 — Coverage

Numerator / denominator over an explicit in-scope topic list.

```
denominator = in-scope topics (PCI-DSS 4.0.1 requirements + FATF rec. + SOC 2 TSC)
              MINUS explicit OUT_OF_SCOPE exclusions (jurisdiction-specific, legal-advice framing, real-time regulatory)
numerator   = topics where the model produces a valid+cited answer
coverage    = numerator / denominator
```

The `OUT_OF_SCOPE` list lives in code (`src/evaluate.py`) and is editorial — reviewable in a PR. Without it the score would drop and stay low for reasons unrelated to fine-tuning quality.

**v0's coverage is the non-regression floor.** Every subsequent version must hold or improve it.

### Tier 3 — LLM-as-judge

Only fires when Tier 1 + Tier 2 are inconclusive. Frontier model scores the held-out and hard-case eval sets on a fixed rubric (answer quality, citation correctness *beyond* existence, hedging appropriateness on "it depends" cases).

Designed but not the primary signal. If we ever rely on Tier 3 to break a tie, the result needs N≥3 judge runs with different seeds.

## Per-version results table

Every version (v0, v1, v2, …) gets a row in `results/eval_results.md`:

| Metric | v0 | v1 | … |
|---|---|---|---|
| Training cost (one-time) | n/a | $X | $Y |
| Training wall time | n/a | … | … |
| Inference $/query (4-bit) | $a | $b | $c |
| Time-to-first-token | … | … | … |
| Tier 1 — format adherence | … | … | … |
| Tier 1 — citation exists | … | … | … |
| Tier 2 — coverage | floor | held / +Δ | … |
| Tier 3 — judge (if invoked) | … | … | … |
| **Verdict** | reference | ship / hold / reject | … |

Failed experiments stay in the table. A clean ship-only record hides the lessons that made the wins possible.

## What this does NOT prove (caveats)

These are honest weaknesses readers should know before drawing conclusions:

1. **n=1 is not a result.** LoRA training is noisy (init, data shuffle, optimizer state). Each ship-candidate config is trained ≥3 times with different seeds before a verdict.
2. **Synthetic-data ceiling.** The model can only be as good as the larger model that generated its training pairs. A real expert-labeled dataset would beat this.
3. **Citation existence ≠ citation correctness.** Tier 1 catches fabricated section numbers; it does not catch correctly-cited-but-wrong-for-the-question sections. Tier 3 partially closes this gap.
4. **No real-engineer A/B.** The eval set is synthetic + hand-written. Behavior on actual on-call questions is untested.
5. **`OUT_OF_SCOPE` is editorial.** Documented, reviewable in a PR — but a choice. Different choices give different denominators.

## Why this framework

Adapted from the C100 "Improving Landing Zone Agents" optimisation campaign — the priority-ordering + multi-tier quality measurement is the same shape, with eval sets and metrics swapped for an LLM fine-tuning context.

The framework forces a specific question for every proposed change: *which tier does this move, and at what cost to the tiers above it?* That ordering is the entire reason the measurement system exists.
