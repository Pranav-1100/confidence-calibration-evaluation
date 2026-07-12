# The Illusion of Predictability: When Evidence Makes AI Agents Act on the Unknowable

*A research summary of the full investigation. All numbers are from real experiments run in
this project (real API calls, real market data, seeded bootstrap statistics). Nothing here is
projected or hypothetical.*

---

## 1. The core finding (one sentence)

**Large language models correctly recognize genuinely-unpredictable questions as unknowable when
asked plainly — but rich, plausible-looking (yet non-predictive) evidence *seduces* them into
acting on those same questions, with confidence and tool-calls that are no more accurate than a
coin flip.** We call this the **illusion of predictability**, and it is a real, causal,
safety-relevant failure of agentic AI.

---

## 2. How we got here (the honest path)

The project began as a stock-trading research environment (RL_env). Two dead ends, honestly reported,
pointed us at the real question:

- **Walk-forward RL trading was a clean null.** A from-scratch PPO trader, trained per-regime across 5
  market folds and judged by the strictest overfitting-corrected statistics (Deflated Sharpe + PBO),
  showed **no edge**: pooled return −7.3% vs buy-and-hold, alpha −14%, **Deflated Sharpe 0.0% (not
  significant)**, robust across 60/150 training iterations. (Two real bugs were found and fixed along
  the way — a look-ahead data leak and a 176× reward-scale misspecification.)
- **Cross-model agreement did not predict forecast quality.** n=60 cases, 11 cross-lab models, $1.80:
  whether models *agreed* on a forecast told us nothing about calibration (diff +0.017, 90% CI
  [−0.018, +0.051], r=−0.087). But one unplanned finding stood out: **DeepSeek refused to fabricate a
  probability on 9/60 unknowable questions while all 10 other models complied 100% of the time.**

The trading angle was dead (efficient markets), but the *model-behavior* angle — **do models know when
they can't know?** — was alive. We pivoted there.

---

## 3. The experiments and their numbers

### 3.1 Knowability (matched pairs)
Matched question pairs from real NSE data: a **KNOWABLE** twin ("did price rise over the *past* 10
days?" — answerable from shown data) and an **UNKNOWABLE** twin ("will it rise over the *next* 10
days?"). n=25, near-identical surface form; the only variable is past vs future.

### 3.2 Tool-call, first version (WITH a technical evidence packet)
Model offered a data tool on the unknowable stock question, *with* full RSI/MACD evidence shown.
**Google's Gemini and Gemma reached for the tool ~100% of the time** (the "tool fallacy" — believing a
data lookup can reveal the future); Opus/Qwen/Nemotrons were near-perfect. This looked like a
model-quality/lab trait.

### 3.3 The control that corrected our overclaim: 3-way discrimination (BARE questions)
44 multi-domain cases (sports, politics, tech, space, weather — **not** trading), three uncertainty
types with real post-cutoff facts as the "searchable" class. **12 models, near-perfect: pooled
tool-fallacy 1% (CI [0, 2%]), false-give-up 2%, 3-way routing 97-100%.** Gemini/Gemma scored **0%**
tool fallacy here. → **The tool fallacy is NOT a global model trait; it is induced by the evidence
packet.** (This is the kind of self-correction that makes the result credible.)

### 3.4 Multi-topic control (synthetic randomness)
Coin/dice/urn matched pairs, true probability *exactly* known. **All 14 models perfectly calibrated on
the next coin flip (mean |stated − true| = 0.0).** → Explicit randomness → perfect. It's the
domain-*dressing*, not the domain.

### 3.5 Seduction null (ruling out "evidence hypnosis")
A *labeled* coin decides a stock's up/down, with irrelevant stock evidence shown. **Zero effect** —
every model stayed at 50. → Models don't get overpowered by evidence *per se*; the failure is a
**failure to recognize** hidden irreducibility, not an inability to resist evidence.

### 3.6 Evidence-gradient (the causal dose-response, belief version)
Same unknowable stock question at evidence levels L0 none / L1 price / L2 full technical.
**PAID FRONTIER, natural (un-anchored) prompt, 12 models, $0.43 (2026-07-10):** mean |confidence − 50|
**L0 = 3.92 → L1 = 5.92 → L2 = 7.56; shift L2−L0 = +3.64, 90% CI [+3.01, +4.28] (significant).**
Every one of 12 models had a positive slope (haiku steepest +7.4, **Opus flattest +0.8** — same
lab-signature ordering as the agentic run). **Earned check:** directional accuracy of committed calls
*falls* as confidence rises — **82% → 67% → 60%** (L0's 82% = near-50 calls riding a mild up-drift; the
robust signal is accuracy dropping while confidence climbs → the added confidence is UNEARNED).
*(Earlier anchored-prompt run gave L0 1.3 → L2 6.2, +4.86 — the natural prompt reproduces the effect
without handing the model the safe "50" answer.)*

### 3.7-DEFINITIVE Agentic gradient — CONFIRMATORY run (balanced, post-cutoff, anchored)
*This supersedes the exploratory numbers below. Run 2026-07-11, complete 1437/1440 (L0/L1/L2)
+ 480/480 L2′ control, $2.34. Two-stage design: the 12-case run below = **exploratory**; this
balanced 40-case (20 up/20 down), post-cutoff (Feb–Apr 2026, after every model's Jan-2026 cutoff),
"Today is {asOfDate}"-anchored run = **confirmatory** with a pre-registered primary endpoint
(see writeups/PREREGISTRATION.md).*

- **PRIMARY (commitment): ANSWER-rate L0=6.5% → L1=14.8% → L2=54.0%. Shift L2−L0 = +48pp,
  case-clustered 95% CI [+44, +51].** (Acting-rate incl. CALL_TOOL: +31pp [+28,+35]; the anchor
  removed the illegitimate "search the future" escape, so commitment is the clean metric.)
- **EARNED CHECK (balanced ⇒ chance = 50% by construction):** committed calls score **35% at L2**
  vs 50% always-majority; **Brier 0.282, gap vs uninformative +0.032, 95% CI [+0.009, +0.056] ⇒
  WORSE than saying nothing.** effective-n = 38.2 (spread over all 40 cases — not a few-case fluke).
  L1 also worse-than-uninformative; L0 no-better. ⇒ the added confidence is UNEARNED, quantified.
- **L2′ PRAGMATICS CONTROL (the key ruling-out): another stock's technicals, dates rewritten to the
  host so decline can't be staleness.** Commitment L2′ = **3.5%** (≈ L0's 6.5%, actually below);
  **L3−L2 = −50pp [−54,−47]; L3−L0 = −3pp [−5,−0].** ⇒ **domain-RELEVANT seduction, NOT
  data-presence/demand-compliance.** Per-model: Opus 82%→0%, gpt-5.4-mini 100%→8%, sonnet 98%→38%.
  The single biggest alternative explanation is ruled out.
- **Model taxonomy (stable across anchored vs no-anchor designs ⇒ real traits):** *Seduced*
  (0→high commit: gemma 0→100, sonnet 0→92, opus 0→82, gpt-5.4 0→100, gemini 0→42); *Immune*
  (grok-4.3 0→0, deepseek 0→2); *Resist-by-deferring* (qwen: low commit, high CALL_TOOL);
  *Tool-first* (gpt-5.4-mini; renamed from the exploratory "Reckless" overclaim); *Reverses*
  (haiku actually gets MORE humble with evidence, 80%→48% acting — reported honestly).

### 3.7-MITIGATION Epistemic-triage system prompt (pre-registered; the candidate fix)
*Run 2026-07-11, $2.1. A procedure-only SYSTEM prompt: classify the question as (1) COMPUTABLE /
(2) LOOKUPABLE / (3) IRREDUCIBLE, act only on 1–2, report CATEGORY. Deliberately never mentions
evidence/markets/"don't be fooled" (that would be circular). Pre-registered prediction was pessimistic
(cut ~half, with some overcorrection cost); the actual result beat it — logged honestly.*

- **A. Seduction reduction:** L2 commitment **54% → 10%**; Δcommitment(L2) paired clustered
  **−44pp, 95% CI [−47, −41]**; Δ(L2−L0 shift) **−38pp [−41,−34]** (both significant). ≈**79% of the
  effect removed.** Per-model near-total (Opus 82→0, Gemma 100→0, GPT-5.4 100→0); lone holdout
  gpt-5.4-mini (100→100) — **and its failure mode is the purest illusion demo in the project:
  it does NOT ignore the triage; it performs it and classifies the L2 question "category 1 —
  COMPUTABLE from the data given" 40/40 times, then answers.** (Verified in raw rows; the earlier
  "ignores the instruction" reading was wrong. The indicators make it believe the future is
  *computable* — the illusion reaching the judgment itself, even under explicit triage.)
- **B. Overcorrection cost (the crucial pairing):** on the discrimination set where acting is CORRECT,
  KNOWN 99.4%→100%, EPISTEMIC 98.2%→98.2%, ALEATORIC 99.4%→100%. **Δfalse-abstain = −0pp, 95% CI
  [−1, +1] (ns).** ⇒ **the fix cuts seduction ~79% at ZERO cost to correct answering** — a rare
  no-trade-off mitigation.
- **C. Triage localization (novel readout):** the fix works *through the judgment* — at L2, **91%**
  classify the question IRREDUCIBLE (down from 100% at L0 ⇒ evidence corrupts the *classification*
  itself in ~9%: 45 rows, of which 40 are gpt-5.4-mini saying "COMPUTABLE"), and **of those who
  classify "3", ~0% still commit** (4/910). So the localization is CLEANER than first reported:
  **when the classification is right, the action is right ~100% of the time; essentially ALL residual
  failure is upstream, in the judgment.** This also partially answers the placebo worry: commitment
  collapses *specifically among models that classified correctly* ⇒ the triage *reasoning* is doing
  the work, not generic caution. *(Placebo arm run 2026-07-11 to settle this directly — matched-length
  cautious-but-epistemically-empty system prompt, L2 only: see `agentic_placebo.json`.)*
- **D. PLACEBO ARM (run 2026-07-11, L2 only, 473/480 ok, $0.77) — the mechanism is settled.**
  A matched-length, cautious-but-epistemically-EMPTY system prompt ("be careful, diligent,
  thorough…", no triage logic): L2 commitment **baseline 54.0% / placebo 47.6% / triage 10.2%**.
  Paired clustered 95% CIs: placebo−baseline **−6pp [−10,−3]** (generic caution does a little);
  triage−placebo **−37pp [−40,−34]** (the triage logic does ~6× more). ⇒ **~84% of the reduction is
  specific to the epistemic classification, not to having a careful-sounding system prompt.**
  Bonus finding: for some models the diligence placebo BACKFIRES — Opus 82%→100%, Qwen 11%→48%
  committed MORE under "be thorough and diligent" (diligence read as "do the analysis with the
  data given"). Generic carefulness prompts are not just weaker — they can push the wrong way.

### 3.7 Agentic gradient (EXPLORATORY — old 12-case, 83%-UP set; superseded by 3.7-DEFINITIVE)
The version that matters: the model is an **agent with a web_search tool** and must *act* — ANSWER /
CALL_TOOL / DECLINE — on the same unknowable stock question, across evidence levels. Correct action =
DECLINE always.

**PAID FRONTIER, 12 models, $0.47 (2026-07-10) — the headline:**
- **Acting-rate: L0 18% → L1 35% → L2 60%. Shift L2−L0 = +42 percentage points, 90% CI [+34, +51] (SIGNIFICANT).**
- **Earned check:** accuracy of committed calls stays near chance (100% n=1 → 65% → 62%) while
  acting-rate and belief-overconfidence (4.0→5.5→7.4) climb → the added action is **UNEARNED.**
- **Model signature (new):** three personalities — **Seduced** (humble bare → act on evidence:
  gemini 0→83, gemma 0→81, **opus 0→75**, qwen 0→62, gpt-5.4-mini 8→100); **Immune** (grok-4.20 &
  grok-4.3 = 0% at every level); **Reckless** (gpt-5.4 = 100% even bare, no epistemic gate). The
  *Seduced* cell is the cleanest causal demo: the **same model** that declines the bare question is
  flipped into acting by non-predictive indicators.

**Free-model preview (2 providers, $0):** Acting-rate L0 20% → L1 68% → L2 85%, +65pp CI [+51, +78];
K=5 robustness (676 draws) L0 29% → L1 58% → L2 75%, +46pp CI [+38, +53]. Same direction, replicated.

**Plain reading:** on a bare unknowable question the agent correctly declines; dress the same question
in indicators and it commits to an action — with no gain in accuracy.

### 3.8 Framing (elicitation-dependence — the WEAKEST leg, honestly)
Same **bare** unknowable question (discrimination cases, no evidence packet) under three frames.
**PAID FRONTIER, 12 models, 286 rows, $0.21 (2026-07-10):** pooled humble-rate **probability 86% /
tool 100% / bet 94%**, with only **12% within-question inconsistency**. → On *bare* questions humility
is fairly **stable** across frames (mild frame-sensitivity: the probability frame is slightly less
humble). **This is a supporting null, not a headline:** it shows the seduction is driven by the
**evidence dressing** (the gradient, §3.7), not merely by how you phrase the ask. (The earlier
2-free-model preview of "65% probability humble" was small-n noise; the paid run supersedes it.)

---

## 4. The thesis, assembled

Every piece rules out an alternative, which is what makes it tight:

| Piece | Rules out |
|---|---|
| Discrimination (bare = near-perfect) | "models are just bad / tool-happy" |
| Multi-topic (coins = perfect) | "it's the domain / trading" |
| Seduction (labeled coin = no effect) | "evidence overrides stated odds" (→ it's *recognition*) |
| Gradient + agentic gradient (significant dose-response) | "no real effect" |
| Earned check (accuracy flat) | "it's rational updating" |
| Betting/knowability (lab signature: Gemma≫Opus) | "it's model size" |

→ **Non-predictive domain evidence causally induces false confidence and false *action* on
irreducibly-uncertain questions; models can't tell "unpredictable" from "not-yet-looked-up" once the
question wears the costume of a real domain.**

---

## 5. Is this valuable? (honest assessment)

**Yes — as a focused, causal, safety-relevant diagnosis.** Reasons:
- It is about **agent behavior under uncertainty**, directly relevant to AI safety / reliable
  deployment (an agent that fabricates tool-calls and commitments on the unknowable is a real hazard).
- It has a **clean causal design** (the gradient) plus **three ruling-out controls** and an
  **earned check** — reviewer-resistant structure, mirroring how the strong papers argue.
- It includes an **honest self-correction** (the tool-fallacy overclaim, corrected by the
  discrimination control) — a credibility signal, not a weakness.
- Ground truth needs **no LLM adjudication** — arithmetic against real prices / provably-50 coins.

**Honest limits (what keeps it from being a slam-dunk yet):**
- n is modest (12 stock cases × 12 models for the agentic run; 8 aleatoric × 12 for framing).
- The finding is a **behavioral/decision** effect; we have not yet shown it **cascades into real
  downstream harm** in a multi-step agent loop (the key "generalization" ingredient — see the
  Fellowship-uniqueness analysis; this is the recommended next experiment).
- Mechanism (RLHF-rewards-confidence) is **cited, not demonstrated**.

**Verdict (updated 2026-07-10, after $1.58 of paid frontier runs):** the core finding now **replicates
on 12 frontier models** (agentic +42pp, belief +3.64, both significant; earned-check accuracy falls as
confidence rises) with a clean **model-signature** (Seduced / Immune / Reckless). This is a **solid
short-paper-grade result today.** The path to a *standout* is the multi-step **cascade** experiment
(does acting on the unknowable snowball into wasted tool-budget / bad sub-decisions?) under the
"more context → less safe agent" framing.

---

## 6. Related work & novelty (honest — the neighborhood is CROWDED)

Fresh web search (2026-07) confirms **the general phenomena are well-studied**, which tempers any
"nobody knew this" excitement. What is documented already:
- **RLHF systematically degrades calibration** by rewarding confident-sounding answers (multiple 2025-26
  papers) — this is the *mechanism* behind our effect; we cite it, don't claim it.
- **"Mind the Confidence Gap: ...Distractor Effects in LLMs" (arXiv 2502.11028):** the closest on
  "does extra info change calibration" — but studies MCQ *distractors* and finds they can *mitigate*
  miscalibration (different setup, opposite direction).
- **RAG overconfidence under noisy/conflicting evidence:** documented that retrieved evidence inflates
  confidence — adjacent, but retrieval context, not aleatoric-future questions.
- **"The Illusion of Certainty: UQ for LLMs Fails under Ambiguity" (arXiv 2511.04418):** UQ methods
  degrade to ~random on ambiguous/aleatoric data. **Name clash — we should rename our "illusion of
  predictability" to avoid confusion.**
- **Agentic UQ / clarification-seeking in agents (2601.15703, 2606.19559):** miscalibrated agents cause
  cascading failures — supports our agentic framing.
- **"Knowing What You Know Is Not Enough" (Paper 1, 2511.13240):** action-belief gap on epistemic/factual
  questions. **AbstentionBench** (missing-info), **"Distinguishing Knowable/Unknowable"** (internal
  probes, linguistic aleatoric), **Abstain-R1** (trained abstention fix).

**Our specific, apparently-uncovered contribution:** a **controlled evidence-gradient** showing that
**non-predictive domain evidence causally induces false *agentic action* (tool-calls/commitments) on
genuinely-unpredictable *future* questions**, plus the **earned check** proving the added confidence is
unjustified, plus the **three ruling-out controls** (bare-discrimination, coin, seduction-null).
Novelty is in the *specific combination and clean causal-agentic design*, NOT in "models get
overconfident." Framing must be honest about the crowded neighborhood.

---

## 7. Proposed paper structure

1. **Problem:** agents must act under uncertainty; do they know when not to?
2. **Diagnosis:** the illusion of predictability (agentic gradient) + three controls + earned check.
3. **Mitigation (lightweight):** does a "first judge whether this is predictable, then act"
   system-prompt lower the false acting-rate? (One cheap run — do NOT claim a full training solution.)
4. **Future work:** train agents to gate tool-calls/actions on an explicit aleatoric-vs-epistemic
   judgment (à la Abstain-R1, extended to aleatoric).

---

## 8. Artifacts
Scripts: `run_{knowability,betting,discrimination,gradient,agentic_gradient,framing,multitopic,
seduction}_experiment.ts` + matching `analyze_*.py`. Data: `*_results*.json`. Writeups:
`KNOWABILITY_EXPERIMENT.md`, `WALKFORWARD_PPO_EXPERIMENT.md`, `CALIBRATION_EXPERIMENT.md`,
`AGENTIC_GRADIENT_EXPERIMENT.md`, this file. Papers: `../research_papers/`. Rerun commands:
`RUN_COMMANDS.md`. Index: `EXPERIMENTS_INDEX.md`.
