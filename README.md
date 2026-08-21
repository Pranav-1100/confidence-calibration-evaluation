# Calibrated Enough to Know, Not Calibrated to Act

**Fabricated Evidence Makes LLM Agents Commit to the Unknowable**

Pranav Aggarwal, independent researcher · [DOI 10.5281/zenodo.21325375](https://doi.org/10.5281/zenodo.21325375) · ORCID [0009-0005-1243-0520](https://orcid.org/0009-0005-1243-0520)

**[Read the paper (PDF)](paper.pdf)** · [Markdown source](paper.md)

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
python3 -m venv .venv && .venv/bin/pip install markdown matplotlib
.venv/bin/python scripts/v2/make_v2_figures.py     # rebuild all six figures from data/
.venv/bin/python scripts/v2/build_pdf.py           # rebuild paper.pdf from paper.md
```

## What is in the data

12-model runs across four domains; four scrambled-display constructions including a fully
fabricated arm; a dose-response run over panel density; the original equity study; eleven trained
checkpoints' generations under two evaluation framings; and a tense-balanced control set built to
break the one confound that could have explained the training result away.

Two cautions are documented in `data/README_v2_data.md` and matter if you re-analyse: `sealedYes`
is a placeholder on the sports and weather unknowable arms, which have no resolved outcomes; and
two models leave a large fraction of their responses unparseable, so their discrimination scores
are lower bounds.

## Citing

Cite the Zenodo record, which resolves to the latest version:

```
Aggarwal, P. (2026). Calibrated Enough to Know, Not Calibrated to Act:
Fabricated Evidence Makes LLM Agents Commit to the Unknowable.
DOI 10.5281/zenodo.21325375
```

## License

MIT for code; see `LICENSE`. The paper and figures are the author's work.
