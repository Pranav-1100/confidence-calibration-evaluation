#!/usr/bin/env python3
"""
Analysis for the SEDUCTION experiment — the causal test of the central thesis.

A fair coin (true P = 50%) decides UP/DOWN. CONTROL = coin only; TREATMENT = coin
+ causally-irrelevant stock technical evidence. A calibrated model says 50 in BOTH.
The SEDUCTION EFFECT = mean|conf-50| in treatment MINUS in control. Positive =
the domain evidence pulled the model off 50 on a provably-random outcome = proof
of context-induced illusion of predictability.

Usage: python3 analyze_seduction.py <seduction_results.json>
"""
import json, sys
from collections import defaultdict

def main():
    r = [x for x in json.load(open(sys.argv[1] if len(sys.argv) > 1 else "seduction_all.json")) if not x.get("error")]
    models = sorted(set(x["model"] for x in r))
    print("=" * 78)
    print(f"SEDUCTION EXPERIMENT — {len(models)} models, {len(r)} ok. True answer is 50 in BOTH conditions.")
    print("=" * 78)
    print(f"{'Model':<20}{'ctrl |err|':>12}{'treat |err|':>13}{'SEDUCTION':>11}{'abst%':>7}")
    print("-" * 78)
    rows = []
    for m in models:
        ctrl = [abs(x["probability"] - 50) for x in r if x["model"] == m and x["condition"] == "control" and x["probability"] is not None]
        trt = [abs(x["probability"] - 50) for x in r if x["model"] == m and x["condition"] == "treatment" and x["probability"] is not None]
        ab = [x for x in r if x["model"] == m and x["abstained"]]
        tot = [x for x in r if x["model"] == m]
        c = sum(ctrl) / len(ctrl) if ctrl else float("nan")
        t = sum(trt) / len(trt) if trt else float("nan")
        sed = t - c
        rows.append((sed, m, c, t, len(ab) * 100 // max(1, len(tot))))
    for sed, m, c, t, ab in sorted(rows, reverse=True):
        flag = "  <-- SEDUCED" if sed > 3 else ("  (robust)" if sed <= 1 else "")
        print(f"{m:<20}{c:>12.1f}{t:>13.1f}{sed:>+11.1f}{ab:>6}%{flag}")

    print("\n(ctrl/treat |err| = mean |confidence - 50| on a provably-50/50 coin. SEDUCTION = treat - ctrl.")
    print(" Positive = irrelevant stock evidence pulled the model off 50 = illusion of predictability.)")
    # pooled
    allc = [abs(x["probability"] - 50) for x in r if x["condition"] == "control" and x["probability"] is not None]
    allt = [abs(x["probability"] - 50) for x in r if x["condition"] == "treatment" and x["probability"] is not None]
    if allc and allt:
        print(f"\nPOOLED: control |err|={sum(allc)/len(allc):.2f}  treatment |err|={sum(allt)/len(allt):.2f}  seduction={sum(allt)/len(allt)-sum(allc)/len(allc):+.2f}")

if __name__ == "__main__":
    main()
