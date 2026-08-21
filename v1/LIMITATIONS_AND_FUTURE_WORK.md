# Limitations & Future Work (repo companion to the paper)

*(Moved out of the main paper per author decision 2026-07-11. NOTE: a condensed
limitations paragraph must be re-imported into the paper before venue submission —
checklist requirement at NeurIPS/ICLR-family venues.)*

## Limitations

1. **One domain, one window, one market** for the confirmatory gradient (NSE equities,
   Feb–Apr 2026). Multi-domain evidence exists only at the bare-question level
   (44-case discrimination set: sports/politics/tech/space/weather) and for explicit
   randomness (coins/dice/urns). A second-domain gradient is the natural replication.
2. **Below-chance accuracy is window-specific.** Committed calls scored 35% at L2, but
   with 90% cross-model herding the effective unit is the case (12/40 modal-correct),
   and the momentum recipe models follow scored 48% in this window. We claim only
   "no better than — here worse than — uninformative."
3. **API serving configuration not controlled** for hosted models (quantization and
   serving details unknown). Model IDs and run dates are pinned in the data files.
4. **Elicitation format.** One prompt family (A/B/C decision format, "Today is
   {date}" anchor). The exploratory-vs-confirmatory comparison shows L0 *tool-seeking*
   shifts with design for 2–3 models, even though within-design commitment gradients
   replicate; per-model baselines are prompt-sensitive.
5. **L2′ mismatch salience.** The irrelevant-evidence control shows a visibly different
   symbol; declining could partly be a "wrong data" reflex rather than a knowability
   judgment. A subtler control — fabricated data attributed to the SAME symbol — would
   test whether merely *looking* relevant suffices to seduce.
6. **12 models, K=1** in the confirmatory run (K=5 robustness exists only from the
   free-model replication).
7. **Two-stage history disclosed:** the exploratory stage used a 12-case, 83%-up-skewed
   set predating the date-anchor fix; the Brier wording rule was fixed after an 85%
   partial (labeled post-hoc-but-constrained in PREREGISTRATION.md).
8. **Mitigation is a probe, not a solution.** System prompts are brittle (one model's
   judgment stays corrupted; deployers don't always control the system prompt); the
   79% reduction demonstrates the judgment is accessible, not that prompting suffices.

## Future work

- **Cascade experiment (consequence-level harm):** does one evidence-induced commitment
  snowball in a multi-step agent loop — wasted tool budget, dependent sub-decisions
  inheriting the false directional prior? Pre-registered endpoint sketch in
  PREREGISTRATION.md. This is the flagship follow-up (fellowship proposal).
- **Training-time gating:** RL that rewards consulting the (existing) knowability
  judgment before acting — Abstain-R1 extended to aleatoric uncertainty, using
  triage-elicitation as the reward-signal source. Motivated directly by the
  localization result (judgment present 91%, consulted ~0% by default).
- **Second aleatoric domain** (weather/sports gradient with non-predictive stats).
- **Subtler relevance controls** (same-symbol fabricated data; partially-relevant data).
- **Placebo taxonomy:** why does "be thorough and diligent" *backfire* for some models
  (Opus 82→100%, Qwen 11→48%)? Hypothesis: diligence is parsed as "use the provided
  data," i.e., an implicit demand amplifier.
- **Multi-seed / K>1 confirmatory replication** and additional providers.

## Framing material removed from the paper (v0.5) — kept for talks/blog/repo

**The Oskamp opening (was the paper's first paragraph).** In 1965, Stuart Oskamp gave
clinical psychologists a case file in four installments, asking the same diagnostic
questions after each. As the file grew, their confidence rose from 33% to 53% while
accuracy stayed near 26–28%. Slovic & Corrigan (1973) found the same in horse-race
handicappers given more variables. Information that *looks* relevant inflates
confidence without informing it — one of the most replicated findings in human
judgment research; this project shows LLM agents inherit it at the action level.

**The human-parallel discussion point.** The failure signature matches the human one
(confidence up, accuracy flat, driven by surface-relevant cues — the illusion of
validity), with one difference in the models' favor: a single instruction reconnects
judgment to action, something human debiasing rarely achieves. Removed from the paper
body per author preference; the Oskamp/Slovic/Tversky–Kahneman citations remain in the
paper's Related Work (human judgment lineage), which is the conventional placement.
