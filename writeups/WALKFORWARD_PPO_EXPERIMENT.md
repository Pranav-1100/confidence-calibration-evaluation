# Walk-Forward PPO with Overfitting-Corrected Significance (DSR + PBO)

**Research question:** does a gradient-RL trading policy (PPO), trained fresh on each of five distinct market-regime folds and tested strictly out-of-sample, produce a *real* edge over passive buy-and-hold — where "real" is judged by the two strictest overfitting-corrected tests in the quantitative-finance literature: the **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) and the **Probability of Backtest Overfitting** via CSCV (Bailey, Borwein, López de Prado & Zhu 2015)?

**Answer: No — a clean, robust null.** Across two training budgets, PPO showed no overfitting-corrected edge over buy-and-hold (Deflated Sharpe 0.0%, not significant), and in fact underperformed both passive and a random policy out-of-sample.

## Why this specific combination is worth reporting

The closest honest precedent — arXiv 2512.12924 (Dec 2025), which stood out precisely for reporting a null (Sharpe 0.33, p=0.34) in a field full of inflated Sharpe claims — did **not** use the Deflated Sharpe Ratio or PBO (only t-tests / bootstrap / permutation tests), and its "reinforcement learning" was an epsilon-greedy bandit selecting among 5 fixed hypotheses, **not a trained trading policy**. The combination of (a) an actual gradient-RL policy, (b) 5-regime walk-forward, and (c) both DSR *and* PBO applied to the result is genuinely under-served. This experiment fills that gap with a correctly-reported number that needs **zero LLM judgment** — market P&L against real historical NSE prices is arithmetic, not adjudication.

## Method (proper walk-forward — no leakage)

- **5 regime folds** (from `WalkForward.ts`): 2020 COVID crash+recovery, 2021 bull, 2022 correction/bear, 2023 recovery, 2024 bull. Each fold has its own training window strictly *before* its test window.
- For each fold, a **fresh PPO** is trained on **only that fold's training window**, then evaluated on **only that fold's test window** — not one globally-trained checkpoint scored five times (which would be a weaker claim).
- The full **benchmark agent zoo** (BUYHOLD, RANDOM, RULE, MOMENTUM, MEANREV, MACD, BREAKOUT, CLAUDE) is run on the identical test windows, giving a 9-strategy comparison set.
- Per-(fold,symbol) returns are pooled into one aligned matrix (40 observations = 5 folds × 8 symbols), then **DSR** is applied to PPO's pooled returns (deflated by the 9 strategies searched) and **PBO** to the full observation×strategy matrix.
- Reproducible: the entire run (PPO init, episode sampling, everything) is seeded via a mulberry32 override of `Math.random` (seed 42).
- Reward misspecification bug fixed earlier this project (`dsrWeight` 50→1) is in effect, so PPO trains on a correctly-scaled reward.

## Results

Two training budgets were run to rule out the obvious "you just undertrained it" objection. **Both give the same verdict:**

| Config | PPO pooled return | Buy & Hold | **Alpha** | PPO Sharpe | **Deflated Sharpe** | PBO |
|---|---|---|---|---|---|---|
| 60 iters/fold | −9.64% | +6.82% | **−16.47%** | −0.671 | **0.0% (✗ not sig)** | 37% |
| 150 iters/fold | −7.29% | +6.82% | **−14.11%** | −0.513 | **0.0% (✗ not sig)** | 37% |

More training moved PPO modestly toward matching passive (in the COVID fold, 150-iter alpha was +0.12%, essentially flat vs B&H) but did **not** change any headline conclusion.

**Strategy leaderboard (pooled mean return, 150-iter run):**

| Rank | Strategy | Return | Sharpe |
|---|---|---|---|
| 1 | MACD | +10.01% | 0.560 |
| 2 | **BUYHOLD** | +6.82% | 0.295 |
| 3 | CLAUDE | +4.40% | 0.291 |
| 4 | BREAKOUT | +4.11% | 0.277 |
| 5 | MOMENTUM | +3.40% | 0.251 |
| 6 | MEANREV | +0.32% | 0.027 |
| 7 | RULE | 0.00% | 0.000 |
| 8 | RANDOM | −0.23% | −0.015 |
| 9 | **PPO** | **−7.29%** | **−0.513** |

PPO finished **last of nine**, behind even a random policy, and underperformed buy-and-hold in **all 5 regimes**.

## Honest interpretation

1. **The headline is a clean, robust null.** A from-scratch gradient-RL trading policy shows no overfitting-corrected edge over passive buy-and-hold across regimes (DSR 0.0%). This is the scientifically valuable, honestly-reported outcome — the entire point of the pivot.
2. **PPO didn't just fail to add value — it transferred *negatively* across regime boundaries.** Finishing below RANDOM out-of-sample, with negative Sharpe in every fold, is consistent with the RL policy overfitting to its training regime and then applying a systematically counterproductive bias (directional and/or cost-churning) once the regime shifts. This "negative transfer under regime shift" is itself a real, if unsurprising, observation and aligns with the literature's standing warnings about RL overfitting in non-stationary markets.
3. **Classic rule-based signals and passive both beat it.** MACD was the only strategy to beat buy-and-hold, and even that edge is not claimed as significant here (it wasn't the DSR candidate; a full multiple-testing correction on MACD would be the natural follow-up).
4. **Honest caveats, disclosed not hidden:**
   - **DSR nTrials = 9** counts only the final strategy comparison, **not** the hyperparameter search (LR, reward-weight, iteration counts) conducted earlier this project. The true deflation is therefore *at least* this strong, likely stronger — i.e., the null is conservative.
   - **Single seed** per configuration. A fuller version would average over multiple seeds; given how decisively negative the result is, seed variance is very unlikely to flip it, but this is stated rather than glossed.
   - **8 symbols, daily bars.** The one place any RL edge appeared this project was the small, unreplicated minute-bar sample; this daily walk-forward does not test that.
   - PPO is a **from-scratch implementation** (manual backprop + Adam, no PyTorch), so this is a statement about *this* PPO under *this* protocol, not about deep-RL trading in full generality.

## What would strengthen this into a paper

- Multi-seed averaging (5–10 seeds/fold) with per-fold confidence bands.
- Apply the same DSR/PBO correction to MACD (the only B&H-beating strategy) to test whether *its* edge survives multiple-testing correction, or is itself a search artifact.
- Extend to the minute-bar data (the one regime where an edge previously, if weakly, appeared) to test whether the null holds at higher frequency.
- Report the honest hyperparameter-search count and fold it into the DSR trial count for a fully conservative deflation.

## Reproduce

```
npx ts-node -r dotenv/config scripts/walkforward_ppo.ts 8 150 0.001 1D 42
```

Raw results: `walkforward_ppo_result_1D_60iter.json`, `walkforward_ppo_result_1D_150iter.json`.
