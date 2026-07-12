# Experiments Index — Model Rosters & Result Files

Central index of the confidence/knowability/action-belief experiments: which models,
which tiers, and where every result + analysis is stored. (Governance calibration is a
separate parked 2nd track — see memory `project-governance-calibration`.)

## Model roster by tier

### FREE (NVIDIA build endpoint — verified working 2026-07-08; volatile, re-probe before big runs)
| Model | Notes |
|---|---|
| `mistralai/mistral-nemotron` | reliable, fast (~300ms) |
| `nvidia/nemotron-3-super-120b-a12b` | reliable, fast (~400ms) |
| `nvidia/nemotron-3-ultra-550b-a55b` | STRONG reasoner (Intelligence Index 47.7), flaky (~6s, 503s) |
| `mistralai/mistral-medium-3.5-128b` | capable, slow (~7.7s) |
| `meta/llama-3.1-70b-instruct` | mostly reliable (3/4) |

(Dead/excluded on 2026-07-08: qwen3-next, llama-3.3-70b, nemotron-70b/nano/super-49b; `nemotron-3.5-content-safety` is a MODERATION CLASSIFIER not a reasoner. OpenRouter free = unusable, daily-capped.)

### SMALL / CHEAP PAID (Hack Club, non-pro, all <$0.0002/call)
| Model | ~cost/call |
|---|---|
| `google/gemma-4-31b-it` | $0.000003 |
| `meta-llama/llama-3.3-70b-instruct` | $0.000003 |
| `google/gemini-3.5-flash` | $0.000012 |
| `anthropic/claude-haiku-4.5` | $0.000029 |
| `openai/gpt-5.4-mini` | small |
| `x-ai/grok-4.20` | $0.0001 |

(Deprecated/invalid: grok-4-fast, grok-3-mini, gemini-3.5-flash-lite.)

### LARGE / FRONTIER PAID (Hack Club — for the capstone frontier run)
| Model |
|---|
| `anthropic/claude-opus-4.8` |
| `anthropic/claude-sonnet-5` |
| `openai/gpt-5.4` |
| `x-ai/grok-4.3` |
| `google/gemini-3.5-flash` (Google's flagship non-image here; pro variants via id if needed) |
| `deepseek/deepseek-v3.2` |
| `qwen/qwen3.7-plus` |

## Result files (all under `RL_env/`)

### Knowability experiment (words: does stated confidence drop for unknowable?)
- Cases: `knowability_cases_n25.json` (n=25 matched pairs), `knowability_cases_smoke.json` (n=5)
- Raw results: `knowability_results_n25.json` (11 models, $2.28)
- Analysis script: `scripts/analyze_knowability.py`
- Writeup: `KNOWABILITY_EXPERIMENT.md`

### Betting / action-belief experiment (action: does stake track knowability?)
- Reuses `knowability_cases_n25.json`
- Raw results: `betting_results_free_n25.json` (5 free), `betting_results_cheap_n25.json` (9 cheap, $0.32), `betting_results_rerun_n25.json` (haiku+ultra, fixed tokens)
- Analysis script: `scripts/analyze_betting.py`
- Runner: `scripts/run_betting_experiment.ts` (rosters: test / cheap / rerun / full)

### Supporting
- Walk-forward PPO + DSR/PBO (trading null): `WALKFORWARD_PPO_EXPERIMENT.md`, `walkforward_ppo_result_1D_*.json`
- Calibration (agreement) experiment: `CALIBRATION_EXPERIMENT.md`, `grading_results_n60.json`
- Free-model probe: `scripts/probe_free_models.ts`
- Papers: `../research_papers/` (10 behavioral) + `../research_papers/governance/` (governance + calibration)
