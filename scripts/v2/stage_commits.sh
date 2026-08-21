#!/usr/bin/env bash
# ==============================================================================
# Stage the v2 release as a sequence of logical commits.
#
# These are real commits with today's dates. What they narrate is the ORDER the
# work actually happened in, which is the thing worth showing: the data came
# first, the analysis found problems in it, the problems were fixed by running
# more experiments, and the paper came last. A reader who opens this history
# sees a research process rather than a single dump.
#
# Run from the repo root:  bash scripts/v2/stage_commits.sh
# Review with `git log --stat` before pushing. Nothing here pushes.
# ==============================================================================
set -e
cd "$(git rev-parse --show-toplevel)"

c () { git commit -q -m "$1" -m "$2" 2>/dev/null && echo "  ✓ $1" || echo "  · nothing staged for: $1"; }

echo "staging v2 release..."

# ---------- 1. the diagnostic data, in the order it was collected ----------
git add -f data/agentic_postcutoff.json data/agentic_mitigated.json data/agentic_placebo.json 2>/dev/null || true
c "data: 12-model equity run, evidence gradient and triage/placebo arms" \
  "Cached generations for the confirmatory run: four evidence levels, 12 frontier
models, outcomes sealed at construction and resolved after the fact. Includes the
triage instruction and the matched-length placebo used to rule out generic prompt
pressure."

git add -f data/frontier_transfer_baseline.json data/crypto_*.json data/sports_*.json data/weather_*.json 2>/dev/null || true
c "data: four-domain transfer run with matched answerable controls" \
  "Extends the effect beyond equities to crypto, sports and ten-day precipitation.
Each domain carries a matched answerable arm so a blanket decliner cannot score
well, which is what makes the discrimination number meaningful."

git add -f data/flagship_cases.json data/scrambled_12models.json 2>/dev/null || true
c "data: scrambled-display control, 12 models x 24 events" \
  "The causal manipulation. Six technical fields replaced with the same asset's
values from an earlier date; symbol, dates, price header and regime tag left real."

git add -f data/flagship_scramfull_cases.json data/scramfull_12models.json 2>/dev/null || true
c "data: fully fabricated panel arm" \
  "Closes the loophole left by the partial scramble: here the header and regime tag
are fabricated too, so nothing the model can see is true except the question.
Commitment 36.8% against 37.6% for real market data."

git add -f data/v2_dose_paid.json data/v2_flagship*.json 2>/dev/null || true
c "data: dose-response over panel density, plus two earlier paid constructions" \
  "Commitment scales with the number of indicators displayed: 0 / 2 / 4 / 7
indicators give 0.0% / 5.4% / 32.1% / 50.0%, 112 decisions per density."

# ---------- 2. training, and what it broke ----------
git add -f data/knowability_train.json data/labeled_train.json data/dpo_pairs.json data/knowability_postcutoff.json 2>/dev/null || true
c "data: synthetic training set and preference pairs" \
  "540 cases about dice, coins, jars and timers. No stocks or crypto; the 24 sports
items are disclosed and deleted in the ablation below."

git add -f data/raw_generations.json data/nse_generations.json 2>/dev/null || true
c "data: first trained checkpoint, transfer and original-benchmark evaluations" \
  "Commitment falls to 0.0% at every evidence level on the original 40 cases, under
a prompt reproduced verbatim from the published study."

git add -f data/knowability_eval.json data/knowability_eval_generations.json data/knowability_eval_seed3.json 2>/dev/null || true
c "data: tense-balanced control set and its evaluations" \
  "Every transfer set separates unknowable from answerable by grammatical tense, so
a 'decline anything future-tense' rule would score perfectly while understanding
nothing. This set inverts that: answerable-future and unknowable-present items.
The trained model does the opposite of the shortcut."

git add -f data/raw_generations_nosports_s*.json 2>/dev/null || true
c "data: sports-ablation retrain, four seeds" \
  "Retrained with the 24 overlapping sports items deleted. Sports discrimination
survives in all four runs (+100/+83/+92/+96), so all three transfer domains can be
read as held out. One of these seeds also produces the paper's strongest negative
result and is reported rather than dropped."

git add -f data/raw_generations_seed*.json 2>/dev/null || true
c "data: main-recipe seeds 1-5" \
  "Six independent training runs. Three were added after the first draft precisely
because three runs is not a spread, and they widened it: crypto's low end moved from
+88 to +62. The bound that matters holds across all seven checkpoints."

# ---------- 3. analysis and figures ----------
git add -f scripts/v2/make_v2_figures.py scripts/v2/*.py 2>/dev/null || true
git add -f figures/ 2>/dev/null || true
c "analysis: figure pipeline, rebuilt from cached data" \
  "Every figure is generated from the released JSON rather than from typed-in
numbers. An earlier version of figure 3 hardcoded a line at zero that the data does
not support; it is now computed."

git add -f data/README_v2_data.md PREREGISTRATION.md 2>/dev/null || true
c "docs: data README and pre-registration" \
  "Documents two hazards for anyone re-analysing: sealedYes is a placeholder on the
sports and weather unknowable arms, which have no resolved outcomes; and two models
leave a large share of responses unparseable, so their scores are lower bounds."

# ---------- 4. the paper ----------
git add -f paper.md paper.pdf 2>/dev/null || true
c "paper: v2" \
  "Fabricated Evidence Makes LLM Agents Commit to the Unknowable. Extends the
published v1 with three further domains, the fabricated-panel control, a CORP
calibration decomposition replacing bin-dependent ECE, a training intervention, and
a map of where that intervention breaks."

git add -f v1/ _archive/ 2>/dev/null || true
c "repo: preserve v1 and superseded figures" \
  "v1 stays in the repository because the DOI resolves to this lineage. Superseded
figures are archived rather than deleted."

git add -A
c "repo: README, citation metadata, layout" \
  "Promotes v2 to the repository root so the current paper is the first thing a
reader sees. CITATION.cff updated to v2.0.0."

echo
echo "done. review with:  git log --oneline --stat | head -60"
echo "then:               git tag -a v2.0.0 -m 'v2: fabricated evidence' && git push origin main --tags"
