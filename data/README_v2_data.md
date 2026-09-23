# v2 data manifest

Every number in the v2 paper is re-derivable from these files with the scripts in
`../scripts/v2/`. No GPU or API access is needed for the analysis - all model outputs are cached.

## Model outputs (cached generations / API results)

| file | n | what |
|---|---|---|
| `frontier_transfer_baseline.json` | 3,456 | 12 frontier models x 288 cases on crypto/sports/weather, under the frontier prompt. Includes the API's per-call `cost` field. |
| `nse_generations.json` | 480 | Trained 3B (SFT-2 and SFT-2+DPO) on v1's own 40 NSE cases, v1's prompt reproduced verbatim, levels L0/L1/L2/L2' plus the knowable arm. |
| `raw_generations.json` | 1,152 | Main run (`SFT-2`, seed 0) and its preference stage (`SFT-2+DPO`) on the transfer domains, both held-out framings. |
| `raw_generations_seed{1..5}.json` | 576 each | Main 540-case recipe, seeds 1 to 5, same eval. |
| `raw_generations_nosports_s{0..3}.json` | 576 each | Ablation recipe (516 cases, the 24 sports training items removed), four seeds, same eval. |
| `knowability_eval_generations.json` | 960 | Tense-balanced control (§6.6) for the main run and the preference stage, both framings. |
| `knowability_eval_seed3.json` | 480 | Tense-balanced control for *main-s3*, a further main-recipe run whose transfer generations were not retained. |
| `v2_gate_free.json` + `labeled_balanced.json` | 528 / 88 | Six open-weight models on a mixed answerable/unknowable set: the §5 elicited-edge separability (AUROC 0.887). |
| `powered_run.json` + `powered_cases.json` | 616 / cases | The §3 48-event replication (scrambled vs real, seven responsive models). |

Together the transfer files hold eleven checkpoints: seven from the main recipe (main run,
preference stage, seeds 1 to 5) and four from the ablation.

## Evaluation cases

| file | n | what |
|---|---|---|
| `{crypto,sports,weather}_seduction_eval.json` | 72 each | Unknowable questions at L0/L1/L2. Correct action is DECLINE. |
| `{crypto,sports,weather}_answerable_eval.json` | 24 each | **Matched answerable controls**: same rich L2 panel, but a question resolvable from the data shown (balanced threshold comparisons, 36 YES / 36 NO overall, ground truth recomputed from the panel text). Without this arm a model that declines everything scores perfectly. Built by `../scripts/v2/build_transfer_answerable.py`. |

## Training data

| file | n | what |
|---|---|---|
| `knowability_train.json` | 540 | Synthetic knowability cases (dice, coins, jars, timers, calendars). Tense x label balanced. Half the unknowable cases carry a rich non-predictive panel, paired with matched cases whose panel genuinely resolves the question. No stocks, crypto, sports or weather. |
| `knowability_eval.json` | 240 | Held-out synthetic eval, 6 arms. |

## Inherited from v1 (also in this folder)

- `knowability_postcutoff.json` - the 40 NSE cases
- `agentic_postcutoff.json` - the published 12-model frontier baseline (L0 6.5 / L1 14.8 /
  L2 54.0 / L2' 3.5). Verified: those figures reproduce under the **any-ANSWER** definition,
  not the confident-only variant (which gives 7.5% at L2). The v2 numbers use the same
  any-ANSWER definition so the comparison is exact.

## Reproducing

```bash
# run from the repository root
python3 scripts/v2/v2_stats.py                # transfer, original-case and dose-response numbers + 95% bootstrap CIs
python3 scripts/v2/aggregate_seeds.py data    # per-run action breakdown across all eleven checkpoints
python3 scripts/v2/compute_T_metric.py data/raw_generations.json   # TruthRL T vs the pre-declared +42 baseline
python3 scripts/v2/calibration_corp.py        # §5 CORP decomposition, AUROC by level, Brier Skill Score
python3 scripts/v2/gate_and_replication.py    # §5 separability (AUROC 0.887) and the §3 48-event replication
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
