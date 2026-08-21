# v2 data manifest

Every number in the v2 paper is re-derivable from these files with the scripts in
`../scripts/`. No GPU or API access is needed for the analysis - all model outputs are cached.

## Model outputs (cached generations / API results)

| file | n | what |
|---|---|---|
| `frontier_transfer_baseline.json` | 3,456 | 12 frontier models x 288 cases on crypto/sports/weather, under the frontier prompt. Includes the API's per-call `cost` field. |
| `nse_generations.json` | 480 | Trained 3B (SFT-2 and SFT-2+DPO) on v1's own 40 NSE cases, v1's prompt reproduced verbatim, levels L0/L1/L2/L2' plus the knowable arm. |
| `raw_generations.json` | 1,152 | SFT-2 and SFT-2+DPO on the transfer domains, both held-out framings. |
| `raw_generations_seed1.json` | 576 | SFT-2 seed 1, same eval. |
| `raw_generations_seed2.json` | 576 | SFT-2 seed 2, same eval. |

Seed 0's generations were not cached (that run predates the caching change), so seed-spread
statistics cover seeds 1 and 2 plus the seed-0 checkpoint re-scored in `raw_generations.json`.

## Evaluation cases

| file | n | what |
|---|---|---|
| `{crypto,sports,weather}_seduction_eval.json` | 72 each | Unknowable questions at L0/L1/L2. Correct action is DECLINE. |
| `{crypto,sports,weather}_answerable_eval.json` | 24 each | **Matched answerable controls**: same rich L2 panel, but a question resolvable from the data shown (balanced threshold comparisons, 36 YES / 36 NO overall, ground truth recomputed from the panel text). Without this arm a model that declines everything scores perfectly. Built by `../scripts/build_transfer_answerable.py`. |

## Training data

| file | n | what |
|---|---|---|
| `knowability_train.json` | 540 | Synthetic knowability cases (dice, coins, jars, timers, calendars). Tense x label balanced. Half the unknowable cases carry a rich non-predictive panel, paired with matched cases whose panel genuinely resolves the question. No stocks, crypto, sports or weather. |
| `knowability_eval.json` | 240 | Held-out synthetic eval, 6 arms. |

## Inherited from v1 (already in `../../data/`)

- `knowability_postcutoff.json` - the 40 NSE cases
- `agentic_postcutoff.json` - the published 12-model frontier baseline (L0 6.5 / L1 14.8 /
  L2 54.0 / L2' 3.5). Verified: those figures reproduce under the **any-ANSWER** definition,
  not the confident-only variant (which gives 7.5% at L2). The v2 numbers use the same
  any-ANSWER definition so the comparison is exact.

## Reproducing

```bash
python3 ../scripts/v2_stats.py          # every headline number + 95% bootstrap CIs
python3 ../scripts/aggregate_seeds.py   # seed spread, mean +/- sd
python3 ../scripts/compute_T_metric.py  # TruthRL T vs the pre-declared +42 baseline
```

## Parsing note

Decisions are read from the model's decision line. A **semantic** parser accepts `RESPONSE:`
(and a few other prefixes) as synonyms for `DECISION:`, because models frequently emit the
former; a **strict** parser accepts only `DECISION:`. Both are reported side by side
throughout, the same pair is applied to every model and every cell, and no cell was re-parsed
selectively. The strict/semantic delta reached +100pp on individual cells, so this is disclosed
rather than silently corrected.

## Caution: `sealedYes` on the sports and weather unknowable arms

Those two domains have **no resolved outcomes**. In the source case files
(`sports_seduction_eval.json`, `weather_seduction_eval.json`) `sealedYes` is `null`. The runner
coerces it with `bool(...)`, so in `frontier_transfer_baseline.json` it is stored as `false` on all
864 rows of each. **That `false` means "unknown", not "the outcome was NO."**

No result in the paper is affected: those two domains are scored only on commitment rate, which does
not read `sealedYes`, and the Brier / CORP analysis uses the equity run alone. But any new analysis
that groups or scores by `sealedYes` on sports or weather would silently be scoring against a
placeholder. Filter to `domain == "crypto"` (unknowable arm) or to `arm == "know"` (all three
domains, which do carry real `goldAnswer` values, 12 YES / 12 NO per domain) before scoring accuracy.
