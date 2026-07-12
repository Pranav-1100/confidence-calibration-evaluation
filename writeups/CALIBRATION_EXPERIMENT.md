# Calibration Experiment — Results

**Research question:** does cross-model agreement/disagreement on a forecast correlate with forecast *calibration quality* (proper scoring, across many cases) — not "was any single call right," which is not a valid question for a stochastic market outcome (see conversation for the full methodological correction that led here: a market outcome is one noisy draw, not a verifiable fact, so per-case correctness is invalid — only calibration across many cases is a valid test).

Two runs exist: an initial n=10 pilot (4 Claude-family models, via the `Agent` tool, free) and a scaled n=60 real run (11 cross-lab models, via 3 real API providers, real cost $1.80). **The n=60 run is the headline result** — it has real statistical power (bootstrap confidence intervals, not eyeballed differences) and genuine cross-lab model diversity. The n=10 pilot is kept below as the preliminary run that validated the pipeline and shaped the design.

---

## Main result (n=60, cross-lab, bootstrap-tested)

**Design:** 60 real historical (symbol, as-of-date) points, 12 sampled from each of the 5 `WalkForward.ts` regime folds (2020 COVID crash+recovery, 2021 bull, 2022 bear/correction, 2023 recovery, 2024 bull), spanning 27 distinct symbols, with a minimum 45-day spacing enforced between cases sharing the same symbol to reduce the serial-correlation risk financial-backtest literature warns overlapping windows introduce. Evidence packets (RSI, EMA20/50, MACD histogram, Bollinger position, ATR%, volume ratio, 5d/20d returns, regime classification) built from real cached NSE data via existing, trusted code (`indicatorService`, `RegimeDetector`) — no look-ahead. Forecasting target: **P(price higher in 10 trading days)**.

**Models (11, three real API providers, genuine cross-lab diversity):**
- Paid, via Hack Club AI proxy: Claude Opus 4.8, Sonnet 5, Haiku 4.5, Fable 5 (Anthropic); GPT-5.4 (OpenAI); Gemini 3.5 Flash (Google); Grok 4.3 (xAI); DeepSeek V3.2; Qwen 3.7 Plus (Alibaba)
- Free, via dedicated per-key quotas (NOT a shared/saturated community pool): Qwen3-Next-80B via NVIDIA Build; Gemma 4 31B via OpenRouter direct

**Real cost:** $1.797 for 660 calls (60 cases × 11 models). 648/660 succeeded.

### Per-model Brier scores (bootstrap 90% CI)

| Model | N | Brier | 90% CI | Directional accuracy |
|---|---|---|---|---|
| Gemini 3.5 Flash | 60 | **0.2300** | [0.211, 0.250] | 37/60 |
| Fable | 60 | 0.2366 | [0.220, 0.253] | 36/60 |
| Haiku | 60 | 0.2367 | [0.208, 0.267] | 37/60 |
| Sonnet | 60 | 0.2381 | [0.224, 0.253] | 35/60 |
| Qwen 3.7 Plus | 60 | 0.2387 | [0.222, 0.257] | 36/60 |
| Grok 4.3 | 60 | 0.2395 | [0.221, 0.259] | 37/60 |
| GPT-5.4 | 60 | 0.2397 | [0.220, 0.260] | 37/60 |
| Opus | 60 | 0.2398 | [0.229, 0.251] | 35/60 |
| Gemma 4 31B (free) | 48* | 0.2408 | [0.214, 0.268] | 31/48 |
| Qwen3-Next-80B (free) | 60 | 0.2435 | [0.219, 0.268] | 35/60 |
| DeepSeek V3.2 | 51** | 0.2650 | [0.241, 0.289] | 24/51 |

*12/60 calls failed on the free tier (upstream rate limits) despite retries — expected free-tier flakiness, not a pipeline bug.
**9/60 DeepSeek responses had no numeric probability at all — see "DeepSeek's abstentions" below, this is not an error, it's a real behavioral finding.

Naive baselines for reference: always-guess-50% = 0.2500; always-guess-the-sample-base-rate (55% of cases were actually UP) = 0.2475.

### Model vs. naive-50% baseline

**10 of 11 models beat the naive 50% baseline directionally** (only DeepSeek did not) — a meaningfully different picture from the n=10 pilot, where every model was worse than baseline. **However, none of the 11 differences were statistically significant** at a 90% bootstrap CI (every CI straddles zero). Gemini 3.5 Flash and Opus came closest to significance (CIs [-0.040, 0.000] and [-0.021, 0.001] respectively — right at the edge).

### The core question: does agreement predict calibration?

Split into a high-agreement bucket (30 cases with the tightest model spread) and low-agreement bucket (30 widest-spread cases):

- High-agreement bucket: mean Brier **0.2463**
- Low-agreement bucket: mean Brier **0.2293**
- Difference: **+0.0170, 90% CI [-0.018, +0.051] → not significant (null result)**
- Pearson correlation (spread vs. squared error) across all 60 cases: **r = -0.087** (near zero)

**Still a clean null result, now with real statistical power behind it** (bootstrap CI, n=30 per bucket, not n=5 as in the pilot). Disagreement does not predict forecast quality in this domain, at this scale. If anything the point estimate leans the *opposite* direction from DiscoUQ's QA-domain finding (low agreement did slightly better, not worse) — but the CI comfortably includes zero, so this should be read as "no relationship detected," not "disagreement is anti-informative."

### Does "bigger model = better calibrated" replicate (Prophet Arena's finding)?

**No, and this update matters**: at n=10, Opus (arguably the most capable model in that pilot) had the best Brier score, which I flagged at the time as a "small, honest partial replication" of Prophet Arena's finding — with an explicit caveat that n=10 could easily be noise. At n=60, that ordering did not hold: **Gemini 3.5 Flash (a fast/smaller-tier model) is best**, and Opus is mid-pack (8th of 11). This is exactly the kind of correction larger samples are supposed to produce, and it's worth stating plainly: the earlier "replication" was very likely a coincidence of small n, not a real effect.

### DeepSeek's abstentions — an unplanned, genuinely interesting finding

In 9 of 60 cases (15%), DeepSeek V3.2 did not return a numeric probability at all. Reading the actual responses, this was not a parsing failure or API error — DeepSeek explicitly reasoned that it should not produce a calibrated number from the given evidence, e.g.: *"The provided data is insufficient to generate a meaningful probability estimate for a 10-day price change... the evidence only allows for qualitative directional assessment, not a calibrated numerical probability."* Every other model complied with the instruction 100% of the time (0 abstentions across 660 calls). This is a real, measurable behavioral difference in epistemic humility across models — one model was willing to say "I shouldn't guess" under instructions that technically demanded a guess, the other ten were not. This ties directly to the calibration-failure literature found earlier (verbal overconfidence induced by instruction-following/alignment training) and is worth a follow-up on its own.

### Per-symbol sanity check (independence concern)

Brier scores varied genuinely across the 27 symbols represented (from 0.125 on ASIANPAINT to 0.385 on TCS), with no single symbol dominating the sample (max 5 cases for any one symbol, out of 60) — the topline null result is not an artifact of one or two symbols/regimes.

---

## Preliminary pilot (n=10, historical — superseded by the n=60 run above)

10 cases, 4 Claude-family models only (Opus/Sonnet/Haiku/Fable via the free `Agent` tool). All 4 models scored *worse* than the 50% baseline (Brier 0.265–0.305) and all 4 landed on an identical 3/10 directional hit rate — a striking but small-sample result. The agreement-vs-calibration test was also null here (0.284 vs 0.280), but with only 5 cases per bucket the significance-check now correctly flags this as **too small a sample for a reliable verdict** (a safeguard added after this pilot: bootstrap CIs can trivially "look significant" at very low n due to degenerate resampling, and this pilot's numbers should not be read as confirmed by themselves).

## Honest overall interpretation

1. **The core research question has a clean, now well-powered answer: no.** Cross-model disagreement does not predict forecast calibration quality in short-horizon single-stock technical forecasting, at n=60 with real bootstrap testing. This is consistent with the domain being closer to informationally efficient than the QA/factual domains where disagreement-as-signal has been shown to work (DiscoUQ).
2. **The picture on raw forecasting skill improved from the pilot but is still not statistically significant.** Going from n=10 (universally worse than baseline) to n=60 (10/11 models directionally better than baseline, none significantly) is itself informative — it shows the pilot's negative result was partly small-sample pessimism, but 60 cases still isn't enough to confidently claim real skill either. Both directions of overclaiming are avoided here on purpose.
3. **The model-size-ordering finding did not replicate**, and that non-replication is reported as prominently as the earlier partial-replication was — this is the correct, symmetric way to treat a small-sample coincidence once more data arrives.
4. **DeepSeek's abstention behavior is a genuine, unplanned finding** worth a dedicated follow-up, separate from the calibration question this experiment was designed to answer.
5. **Known limitations, unchanged from the pilot**: real symbol/date shown to every model (possible pretraining leakage, not measured here); technicals-only evidence (a deliberate ablation, not the full context the actual product's Prime agent would use); 60 cases is still a research pilot by Tetlock-scale standards, not a large-sample calibration study.

## What would make this a stronger result

Scale n further (the pipeline — `generate_forecast_cases.ts` + `run_forecast_grading.ts` + `analyze_forecast_results.py` — already supports this at near-zero marginal engineering cost, real cost was ~$0.03/case across 11 models) and specifically follow up on the DeepSeek abstention behavior as its own question: does abstention rate correlate with genuinely harder/more ambiguous cases, or is it a fixed behavioral trait independent of case difficulty?
