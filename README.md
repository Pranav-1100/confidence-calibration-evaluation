# Calibrated Enough to Know, Not Calibrated to Act

**Fabricated Evidence Makes LLM Agents Commit to the Unknowable**

Pranav Aggarwal, independent researcher · [arXiv:2608.27167](https://arxiv.org/abs/2608.27167) · [DOI 10.5281/zenodo.22043517](https://doi.org/10.5281/zenodo.22043517) · ORCID [0009-0005-1243-0520](https://orcid.org/0009-0005-1243-0520)

**[Read on arXiv](https://arxiv.org/abs/2608.27167)** · **[Read the paper (PDF)](paper.pdf)** · [Markdown source](paper.md)

---

## What this is

LLM agents deployed behind dashboards and market feeds are assumed to make better decisions with
more context. For questions that are irreducibly uncertain, that assumption inverts.

Across 12 frontier models, commitment to a directional call on a provably unpredictable question
rises **6.5% → 54.0%** as an authoritative-looking indicator panel is added. Fabricating the entire
panel, so that nothing the model can see is true except the question, produces **36.8%** commitment
against **37.6%** for real market data. The trigger is the presentation, not the information.

The failure sits at the act/don't-act gate rather than in stated belief: the same models answer
matched *answerable* questions essentially perfectly, and their stated probabilities are
anti-predictive of outcomes (AUROC **0.346**). The gate can be trained into a 3B model with
synthetic data about dice and coins, and it holds exactly when the response format leaves the model
room to reason.

## Repository layout

| path | contents |
|---|---|
| `paper.md`, `paper.pdf` | the current paper (v2) |
| `figures/` | the six figures it uses, PNG + PDF |
| `data/` | every cached model output behind every number, plus the evaluation case sets |
| `scripts/` | v1 experiment runners and analysis; `scripts/v2/` builds the v2 figures and PDF |
| `PREREGISTRATION.md` | the pre-registration for the diagnostic study, written before the confirmatory run |
| `v1/` | the earlier paper this one extends, with its own figures |
| `_archive/` | superseded figures, kept rather than deleted |

Training code, checkpoint evaluation scripts and the Kaggle notebooks live in the `RL_env/`
directory of the working repository; the cached generations they produced are in `data/` here.

## Reproducing the numbers

Every number in the paper is recomputable from `data/` without an API call or a GPU hour, because
the release contains raw cached generations rather than summary statistics.

```bash
# the numbers (standard library only)
python3 scripts/v2/v2_stats.py                     # transfer domains, original 40 cases, dose-response
python3 scripts/v2/aggregate_seeds.py data         # per-run breakdown for all eleven trained checkpoints
python3 scripts/v2/calibration_corp.py             # §5 CORP decomposition, AUROC by level, Brier Skill Score
python3 scripts/v2/gate_and_replication.py         # §5 separability (AUROC 0.887), §3 48-event replication
python3 scripts/v2/compute_T_metric.py data/raw_generations.json

# the figures and the PDF
python3 -m venv .venv && .venv/bin/pip install markdown matplotlib
.venv/bin/python scripts/v2/make_v2_figures.py     # rebuild all six figures from data/
.venv/bin/python scripts/v2/build_pdf.py           # rebuild paper.pdf from paper.md
```

## What is in the data

12-model runs across four domains; four scrambled-display constructions including a fully
fabricated arm; a 48-event replication of the scrambled-display result; the elicited-edge separability run; a
dose-response run over panel density; the original equity study; eleven trained checkpoints'
generations under two evaluation framings; and a tense-balanced control set built to break the one
confound that could have explained the training result away.

Two cautions are documented in `data/README_v2_data.md` and matter if you re-analyse: `sealedYes`
is a placeholder on the sports and weather unknowable arms, which have no resolved outcomes; and
two models leave a large fraction of their responses unparseable, so their discrimination scores
are lower bounds.

## Corrections (23 September 2026)

A full recomputation of the paper against `data/` found and fixed the following. No headline result
changed.

- The trained model's zero over-abstention holds under the natural framing; under the
  reasoning-suppressing framing both recipes also decline some answerable items (at most 45.8% of
  answerable weather items, in one run). §2, §6.3 and §8 now say so.
- Appendix E lists all eleven checkpoints rather than four, and Appendix C all eleven rather than
  five (the two parsers agree on eight of them, not three).
- §8 ranges now cover all eleven checkpoints; the ablation runs are named *ablation s0-s3*
  throughout; a stale sentence in §6.0 was removed.
- Wording made exact: the calibration bands in §5 are 55-65% and 35-45%; the dose-response edge is
  averaged over the rows that report it; the separability analysis uses 344 labeled rows of 528.
- The data behind the §5 separability result and the §3 48-event replication are now released,
  with the scripts that reproduce them.

## Citing

```
Aggarwal, P. (2026). Calibrated Enough to Know, Not Calibrated to Act:
Fabricated Evidence Makes LLM Agents Commit to the Unknowable.
arXiv:2608.27167. DOI 10.5281/zenodo.22043517
```

arXiv: [arxiv.org/abs/2608.27167](https://arxiv.org/abs/2608.27167) · Zenodo (this version): [10.5281/zenodo.22043517](https://doi.org/10.5281/zenodo.22043517) · Zenodo (v1): [10.5281/zenodo.21325375](https://doi.org/10.5281/zenodo.21325375)

## License

MIT for code; see `LICENSE`. The paper and figures are the author's work.
