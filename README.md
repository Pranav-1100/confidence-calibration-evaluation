# Calibrated Enough to Know, Not Calibrated to Act

**Relevant-looking evidence makes LLM agents commit to the unknowable.**

Pranav Aggarwal — Independent Researcher — pranavaggarwal1100@gmail.com

Paper: [`PAPER_DRAFT.md`](PAPER_DRAFT.md) · [`PAPER_DRAFT.pdf`](PAPER_DRAFT.pdf)

## Summary

An LLM agent with a search tool is asked a provably unpredictable question (will a
stock's price be higher in 10 trading days?) and must ANSWER, CALL_TOOL, or DECLINE.
On 40 outcome-balanced, post-training-cutoff cases (chance = 50% by construction):

| Condition | Commitment (ANSWER) |
|---|---|
| L0 — bare question | 6.5% |
| L1 — two prices | 14.8% |
| **L2 — full technical panel (non-predictive)** | **54.0%** |
| L2′ — same panel, *different* stock | 3.5% |

- Commitment shift L2−L0 = **+48pp** (case-clustered 95% CI [+44, +51]), across 12 frontier models.
- The committed calls are **worse than uninformative**: Brier 0.282 vs 0.250 for always-"50%" (gap CI [+0.008, +0.056]); models herd (90% within-case agreement) on momentum-shaped signals that score 48% themselves.
- The collapse at L2′ shows the trigger is **relevance**, not the presence of data.
- The judgment exists: told to classify knowability first, models label the question
  irreducible **91%** of the time and then commit **0.4%**. The triage instruction cuts
  commitment 54%→10% at zero cost on answerable questions; a matched-length placebo
  prompt yields 48% — and backfires for some models.

Limitations & future work: [`LIMITATIONS_AND_FUTURE_WORK.md`](LIMITATIONS_AND_FUTURE_WORK.md).
Pre-registration with honest timestamps: [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Repository map

```
PAPER_DRAFT.md / .pdf          the paper
PREREGISTRATION.md             predictions & decision rules, timestamped
LIMITATIONS_AND_FUTURE_WORK.md full limitations discussion + follow-up agenda
writeups/                      per-experiment writeups + experiments index
figures/                       all paper figures (PNG + vector PDF)
scripts/                       experiment runners (TypeScript) + analyzers (Python)
data/                          frozen raw results + case files (JSON)
```

## Reproducing

Requirements: Node 18+ (`npm i`), Python 3.10+. API keys via environment
(`HACKCLUB_API_KEY`, `NVIDIA_API_KEY`) in a `.env` file — never committed.

```bash
# definitive agentic gradient (L0/L1/L2 + L2' via LEVELS env)
npx ts-node -r dotenv/config scripts/run_agentic_gradient.ts data/knowability_postcutoff.json out.json all 2.5 1
python3 scripts/analyze_agentic_gradient.py data/agentic_postcutoff.json

# mitigation pair (MITIGATION=1 = triage; MITIGATION=placebo = placebo arm)
MITIGATION=1 LEVELS=0,2 npx ts-node -r dotenv/config scripts/run_agentic_gradient.ts data/knowability_postcutoff.json out_mitigated.json all 1.5
python3 scripts/analyze_mitigation.py data/agentic_postcutoff.json data/agentic_mitigated.json data/discrimination_all.json data/discrimination_mitigated.json

# figures + paper PDF
python3 -m venv .venv && .venv/bin/pip install matplotlib markdown
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/md_to_pdf.py
```

Analyzers are deterministic (seeded bootstrap); every number in the paper reproduces
from the JSONs in `data/`. Total cost of all experiments: ≈ $10 (~2,900 paid model calls).

## Citation

```bibtex
@misc{aggarwal2026calibrated,
  title  = {Calibrated Enough to Know, Not Calibrated to Act:
            Relevant-Looking Evidence Makes LLM Agents Commit to the Unknowable},
  author = {Aggarwal, Pranav},
  year   = {2026},
  note   = {Preprint},
}
```
