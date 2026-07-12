#!/usr/bin/env python3
"""
Analysis for the controlled framing experiment.

Same unknowable question, same model, 3 frames (P=probability, T=tool, B=bet).
A model whose humility is a STABLE TRAIT is humble in all 3 frames or none.
ELICITATION-DEPENDENCE = humble in SOME frames but not others on the same question.

Per model we report the humble-rate in each frame and the FRAME-INCONSISTENCY:
fraction of questions where the model's humility differs across frames.

Usage: python3 analyze_framing.py <framing_results.json>
"""
import json, sys
from collections import defaultdict

def main():
    r = [x for x in json.load(open(sys.argv[1] if len(sys.argv) > 1 else "framing_results.json")) if not x.get("error") and x.get("humble") is not None]
    # Exclude KNOWN-PROBABILITY aleatoric cases (e.g. a fair die: p=1/6 is computable,
    # so stating the probability is CORRECT, not overconfident — including it mislabels
    # the probability frame as "non-humble"). Marked knownProbability in the case set.
    import os
    cf = os.path.join(os.path.dirname(__file__), "..", "data", "discrimination_cases.json")
    excluded = set()
    if os.path.exists(cf):
        excluded = {c["id"] for c in json.load(open(cf)).get("cases", []) if c.get("knownProbability")}
    if excluded:
        before = len(r); r = [x for x in r if x["caseId"] not in excluded]
        print(f"(excluded known-probability cases {sorted(excluded)}: {before-len(r)} rows dropped — probability is computable there, so a stated probability is correct not overconfident)")
    models = sorted(set(x["model"] for x in r))
    print("=" * 84)
    print(f"CONTROLLED FRAMING — {len(models)} models. Questions are ALL unknowable => humble is correct in every frame.")
    print("=" * 84)
    print(f"{'Model':<20}{'P humble%':>10}{'T humble%':>10}{'B humble%':>10}{'inconsistent%':>15}")
    print("-" * 84)
    for m in models:
        byq = defaultdict(dict)  # caseId -> frame -> humble
        for x in r:
            if x["model"] == m: byq[x["caseId"]][x["frame"]] = x["humble"]
        def frate(fr):
            v = [x["humble"] for x in r if x["model"] == m and x["frame"] == fr]
            return sum(v) * 100 // len(v) if v else -1
        # inconsistency: questions with all 3 frames present where humility is not uniform
        incon = tot = 0
        for cid, fm in byq.items():
            if len(fm) == 3:
                tot += 1
                if len(set(fm.values())) > 1: incon += 1
        print(f"{m:<20}{frate('P'):>9}%{frate('T'):>9}%{frate('B'):>9}%{(incon*100//max(1,tot)):>14}%")
    print("\n  P=probability frame, T=tool frame, B=bet frame. humble% = recognized the unknowable & didn't commit.")
    print("  inconsistent% = same question, humility DIFFERS across frames => elicitation-dependence (the key result).")
    # pooled per-frame
    print("\n  POOLED humble-rate by frame (if these differ, humility is frame-dependent across the board):")
    for fr, name in [("P", "probability"), ("T", "tool"), ("B", "bet")]:
        v = [x["humble"] for x in r if x["frame"] == fr]
        print(f"    {name:<12} {sum(v)*100//max(1,len(v))}%  (n={len(v)})")

if __name__ == "__main__":
    main()
