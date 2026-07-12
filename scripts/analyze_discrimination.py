#!/usr/bin/env python3
"""
Analysis for the 3-way uncertainty-discrimination experiment (improved Leg 2).

Full confusion matrix per model: for each true uncertainty type (KNOWN / EPISTEMIC
/ ALEATORIC), what decision did the model make? The correct routing is:
  KNOWN -> ANSWER, EPISTEMIC -> CALL_TOOL, ALEATORIC -> CANNOT_RESOLVE.

The KEY new metric this design enables — DISCRIMINATION — rules out the "tool-happy"
alternative explanation: a model that calls the tool on ALEATORIC (tool fallacy)
must be shown to correctly NOT do so elsewhere. We report:
  - per-type routing accuracy,
  - the TOOL-FALLACY rate (CALL_TOOL on ALEATORIC),
  - the FALSE-GIVE-UP rate (CANNOT_RESOLVE on EPISTEMIC = treating searchable as unknowable),
  - overall 3-way accuracy.
Optionally compares a baseline run vs a +nudge run (does the nudge fix failures?).

Usage: python3 analyze_discrimination.py <results.json> [nudge_results.json]
"""
import json, sys, random
from collections import defaultdict
random.seed(42)

CORRECT = {"KNOWN": "ANSWER", "EPISTEMIC": "CALL_TOOL", "ALEATORIC": "CANNOT_RESOLVE"}

def boot_ci(bits, nb=3000, alpha=0.10):
    """90% bootstrap CI for the mean of a 0/1 list (a rate)."""
    n = len(bits)
    if n == 0: return (float("nan"), float("nan"), float("nan"))
    ms = sorted(sum(bits[random.randrange(n)] for _ in range(n)) / n for _ in range(nb))
    return (ms[int(alpha/2*nb)], sum(bits)/n, ms[int((1-alpha/2)*nb)-1])

def load(fn):
    return [r for r in json.load(open(fn)) if not r.get("error") and r.get("decision")]

def matrix(ok):
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # model->type->decision
    for r in ok: by[r["model"]][r["type"]][r["decision"]] += 1
    return by

def report(ok, title):
    by = matrix(ok)
    models = sorted(by.keys())
    print("=" * 96)
    print(f"{title} — {len(models)} models, {len(ok)} decisions")
    print("=" * 96)
    print(f"{'Model':<20}| {'KNOWN->ANS':>11} | {'EPIST->TOOL':>12} {'(falseGiveUp)':>13} | {'ALEA->CANT':>11} {'(toolFallacy)':>14} | {'3way acc':>9}")
    print("-" * 96)
    agg = defaultdict(lambda: defaultdict(int))
    for m in models:
        def pct(t):
            d = by[m][t]; n = sum(d.values()) or 1
            return d.get(CORRECT[t], 0), n
        ka, kn = pct("KNOWN")
        ea, en = pct("EPISTEMIC")
        egu = by[m]["EPISTEMIC"].get("CANNOT_RESOLVE", 0)  # searchable treated as unknowable
        aa, an = pct("ALEATORIC")
        atf = by[m]["ALEATORIC"].get("CALL_TOOL", 0)       # tool fallacy
        tot_c = ka + ea + aa; tot_n = kn + en + an
        for t in CORRECT: agg[t]["correct"] += by[m][t].get(CORRECT[t], 0); agg[t]["n"] += sum(by[m][t].values())
        agg["ALEATORIC"]["toolfallacy"] += atf
        agg["EPISTEMIC"]["falsegiveup"] += egu
        print(f"{m:<20}| {ka*100//kn:>9}%  | {ea*100//en:>10}%  {egu:>11}   | {aa*100//an:>9}%  {atf:>12}   | {tot_c*100//tot_n:>7}%")
    print("\n  falseGiveUp = EPISTEMIC(searchable) wrongly marked CANNOT_RESOLVE.  toolFallacy = ALEATORIC(unknowable) wrongly marked CALL_TOOL.")
    print("  A model with BOTH low toolFallacy AND low falseGiveUp genuinely discriminates uncertainty TYPE (rules out 'tool-happy').")
    # pooled
    print("\n  POOLED routing accuracy:")
    for t in ("KNOWN", "EPISTEMIC", "ALEATORIC"):
        a = agg[t]; extra = ""
        if t == "ALEATORIC": extra = f"  | tool-fallacy {a['toolfallacy']}/{a['n']} = {a['toolfallacy']*100//max(1,a['n'])}%"
        if t == "EPISTEMIC": extra = f"  | false-give-up {a['falsegiveup']}/{a['n']} = {a['falsegiveup']*100//max(1,a['n'])}%"
        print(f"    {t:<11} {a['correct']}/{a['n']} = {a['correct']*100//max(1,a['n'])}% correct{extra}")
    # bootstrap CIs on the two headline failure rates
    tf = [1 if r["decision"] == "CALL_TOOL" else 0 for r in ok if r["type"] == "ALEATORIC"]
    fg = [1 if r["decision"] == "CANNOT_RESOLVE" else 0 for r in ok if r["type"] == "EPISTEMIC"]
    lo, m_, hi = boot_ci(tf); print(f"\n  Tool-fallacy rate 90% CI: {m_*100:.0f}% [{lo*100:.0f}%, {hi*100:.0f}%]  (n={len(tf)})")
    lo, m_, hi = boot_ci(fg); print(f"  False-give-up rate 90% CI: {m_*100:.0f}% [{lo*100:.0f}%, {hi*100:.0f}%]  (n={len(fg)})")
    return by

def main():
    ok = load(sys.argv[1] if len(sys.argv) > 1 else "discrimination_all.json")
    by_base = report(ok, "DISCRIMINATION — BASELINE")
    if len(sys.argv) > 2:
        ok2 = load(sys.argv[2])
        print("\n\n")
        by_nudge = report(ok2, "DISCRIMINATION — WITH NUDGE (does the PA1-style nudge fix failures?)")
        # delta on the two failure modes
        print("\n" + "=" * 60)
        print("NUDGE EFFECT on failures (baseline -> nudge), pooled:")
        def rate(by, t, wrong):
            c = n = 0
            for m in by:
                for dec, k in by[m][t].items(): n += k
                c += by[m][t].get(wrong, 0)
            return c, n
        for t, wrong, name in [("ALEATORIC","CALL_TOOL","tool-fallacy"),("EPISTEMIC","CANNOT_RESOLVE","false-give-up")]:
            cb, nb = rate(by_base, t, wrong); cn, nn = rate(by_nudge, t, wrong)
            print(f"  {name:<14} {cb*100//max(1,nb)}%  ->  {cn*100//max(1,nn)}%   ({'FIXED by prompt' if cn*100//max(1,nn) < cb*100//max(1,nb) - 5 else 'largely unchanged = trained-in'})")

if __name__ == "__main__":
    main()
