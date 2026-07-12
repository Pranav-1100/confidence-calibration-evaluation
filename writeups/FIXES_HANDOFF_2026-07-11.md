# Fixes applied 2026-07-11 — handoff notes

Context: preparing the agentic-gradient result for a paper / Anthropic-fellowship artifact.
A deep audit of the raw data found four validity problems; all are now fixed in code.
The definitive run is on `data/knowability_postcutoff.json` (40 cases, 20 UP / 20 DOWN,
all as-of dates Feb–Apr 2026 = after every model's Jan-2026 cutoff).

## Fix 1 — Balanced, post-cutoff case set (DONE earlier today, run in flight)
**Problem:** the old paid run used `slice(0,12)` of the n=25 set; those 12 cases were 10/12 UP
(83%). The "earned check" claimed accuracy "stays near chance (50%)" — but chance on that
sample is 83%: always-guess-UP scores 80% on the same rows where models scored 63%. A reviewer
kills the earned-check section instantly. Also 2020-24 dates are inside pretraining.
**Fix:** `scripts/generate_postcutoff_cases.ts` builds an outcome-balanced (20/20) set from
post-cutoff dates only (pool was 91 up / 89 down ≈ coin-flip, which itself validates the
"unknowable" premise). Runner cap removed (`MAX_CASES` env, default = all cases).

## Fix 2 — Analyzer statistics upgraded (`analyze_agentic_gradient.py`)
**Problem (a):** bootstrap resampled individual rows, but rows within a case are correlated
(12 cases only) → anti-conservative CIs. **(b):** 90% CIs; reviewers expect 95%.
**Fix:** cluster bootstrap resampling CASES, 95% CI. On the old data the headline survives:
acting L2−L0 = +42pp, 95% CI [+35, +49] clustered — report it this way.

## Fix 3 — ANSWER / CALL_TOOL decomposition (analyzer)
**Problem:** "acting-rate" pooled ANSWER + CALL_TOOL, but a skeptic can argue CALL_TOOL is
defensible (checking earnings/news is what real analysts do). Also gpt-5.4's "Reckless
100% acting at L0" is actually 100% CALL_TOOL with reasonable check-the-news reasoning —
"Reckless" was an overclaim; call it "tool-first" (do NOT keep the Reckless label).
**Fix:** analyzer now reports ANSWER% / CALL_TOOL% / DECLINE% per level plus a separate
ANSWER-only clustered CI. On old data the rise is almost all DECLINE→ANSWER
(ANSWER 0.7% → 16% → 46%; CALL_TOOL flat ~15-19%): ANSWER-only shift = +45pp, 95% CI
[+38, +52]. **The sharper headline: evidence induces COMMITMENT, not tool-hunger** —
and it survives the CALL_TOOL objection entirely.

## Fix 4 — "Today is {asOfDate}" prompt anchor (`run_agentic_gradient.ts`)
**Problem:** the prompt never stated the present date, but L2 evidence shows an as-of date
months in the PAST relative to the real run date. The tool is described as returning info
"up to today" → for a past as-of date the "future" outcome is already searchable, making
CALL_TOOL objectively CORRECT and poisoning the DECLINE-is-correct scoring.
**Fix:** prompt now opens with `Today is {asOfDate}.` at ALL levels (matched across
conditions, so evidence remains the only varying factor). NOTE: the postcutoff run
launched tonight predates this fix — treat it as the no-date-anchor variant; the definitive
run should be rerun with the anchor (or at minimum reported with this caveat + the
ANSWER-only metric, which the confound doesn't touch).

## Fix 5 — L2′ irrelevant-evidence pragmatics control (runner level 3)
**Problem:** "Data available:" pragmatically implies the client wants the data used
(Gricean relevance / demand characteristics). Is the effect *illusion of predictability*
or just *"they gave me data so they want analysis"* compliance? Biggest remaining
validity threat to the causal claim.
**Fix:** new level 3 = same "Data available:" framing but ANOTHER stock's full technicals
(next case with a different symbol, cyclic — matched format/length, provably irrelevant).
Run via `LEVELS=3` (merge with the main run's L0/L2 by caseId/model). Verdict logic
(printed by analyzer): L3≈L0 & L2>L3 → domain-relevant seduction confirmed;
L3≈L2 → pragmatics/demand-compliance drives it (reframe, still publishable).

## Fix 6 — copies synced
`RL_env/scripts/run_agentic_gradient.ts` still had the old `slice(0,12)` cap; both the
runner and analyzer are now synced from `research/scripts/` (the canonical copies).

## MITIGATION BUILD (2026-07-11, built + typechecked + analyzer self-tested; NOT yet run)
Pre-registered design (PREREGISTRATION.md §Mitigation). `MITIGATION=1` env on BOTH runners adds
the SAME byte-identical epistemic-triage SYSTEM prompt (procedure-only: classify
COMPUTABLE/LOOKUPABLE/IRREDUCIBLE, act only on 1-2 — deliberately NO coaching about
evidence/markets, that would be circular). Models also emit `CATEGORY: <1|2|3>` (parsed,
stored) so failures can be LOCALIZED: evidence corrupting the classification vs
classify-3-but-commit-anyway (instruction-following failure). Both runners refuse a
MITIGATION run whose outFile lacks "mitig" (protects resume from mixing arms).
Discrimination runner also gained resume logic (same pattern as agentic).
Discrimination prompt keeps its baseline "Today is 10 July 2026" anchor ON PURPOSE —
the paired comparison needs the mitigated arm identical to the baseline except the system prompt.
Analyzer: `analyze_mitigation.py <agentic_base> <agentic_mit> <disc_base> <disc_mit>` —
trade-off table + paired case-clustered 95% CIs (Δcommitment L2, Δ(L2−L0) shift,
Δfalse-abstain on KNOWN+EPISTEMIC) + triage-localization readout. Self-tested with
baseline-as-both-arms (all deltas 0, graceful degradation confirmed).

### Mitigation run commands (smoke FIRST, per standing rule)
```
# 1. FREE smoke (~$0, 4 cases x free roster x L0/L2) — read transcripts before paying:
MITIGATION=1 LEVELS=0,2 MAX_CASES=4 npx ts-node -r dotenv/config scripts/run_agentic_gradient.ts data/knowability_postcutoff.json data/agentic_mitigated_smoke.json free 0.05
MITIGATION=1 npx ts-node -r dotenv/config scripts/run_discrimination_experiment.ts data/discrimination_cases.json data/discrimination_mitigated_smoke.json test 0.05
# CHECK in smoke output: CATEGORY parsed non-null; DECLINE reasoning cites classification;
# KNOWN/EPISTEMIC still answered in disc smoke; no format breakage on any model family.

# 2. Paid pair (sequential, never parallel):
MITIGATION=1 LEVELS=0,2 npx ts-node -r dotenv/config scripts/run_agentic_gradient.ts data/knowability_postcutoff.json data/agentic_mitigated.json all 1.5     # 960 cells ~ $1.1
MITIGATION=1 npx ts-node -r dotenv/config scripts/run_discrimination_experiment.ts data/discrimination_cases.json data/discrimination_mitigated.json all 0.8  # 528 cells ~ $0.6

# 3. Analysis:
python3 scripts/analyze_mitigation.py data/agentic_postcutoff.json data/agentic_mitigated.json data/discrimination_all.json data/discrimination_mitigated.json
```

## Run queue after the in-flight postcutoff run finishes (~$0.0009/call)
1. Rerun postcutoff WITH the today-anchor: 40×12×3 = 1440 calls ≈ $1.3 (definitive main result)
2. L2′ control: `LEVELS=3` → 480 calls ≈ $0.45
3. Mitigation (epistemic-triage system prompt), L0/L2: 960 calls ≈ $0.9
4. Mitigation overcorrection check on discrimination set (44×12): ≈ $0.5
5. Optional: second aleatoric domain gradient ≈ $0.5–1; cascade experiment ≈ $1–2
Never run two paid runs in parallel (shared HackClub 450/30min limiter).
