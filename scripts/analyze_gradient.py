#!/usr/bin/env python3
"""
Analysis for the evidence-gradient experiment.

Same implicit unknowable question ("will SYMBOL be higher in 10 days?") at 3
evidence levels: L0 none, L1 price, L2 full technical. True answer is ~50 at
every level. If mean |conf-50| GROWS L0->L2, rich domain evidence is inducing
false confidence about a genuinely-unpredictable event (illusion of predictability),
isolated as a dose-response to evidence amount.

Usage: python3 analyze_gradient.py <gradient_results.json>
"""
import json, sys, random
from collections import defaultdict
random.seed(42)

def boot_diff(a, b, nb=3000, alpha=0.10):
    na, nb_ = len(a), len(b)
    if not na or not nb_: return (float("nan"),)*3
    ds = sorted(sum(a[random.randrange(na)] for _ in range(na))/na - sum(b[random.randrange(nb_)] for _ in range(nb_))/nb_ for _ in range(nb))
    return (ds[int(alpha/2*nb)], sum(a)/na - sum(b)/nb_, ds[int((1-alpha/2)*nb)-1])

def main():
    r = [x for x in json.load(open(sys.argv[1] if len(sys.argv) > 1 else "gradient_all.json")) if not x.get("error")]
    models = sorted(set(x["model"] for x in r))
    print("=" * 78)
    print(f"EVIDENCE-GRADIENT — {len(models)} models. True answer ~50 at every level.")
    print("mean |conf-50| by evidence level (L0 none / L1 price / L2 full technical).")
    print("Rising L0->L2 = evidence induces false confidence on an unknowable event.")
    print("=" * 78)
    print(f"{'Model':<20}{'L0':>7}{'L1':>7}{'L2':>7}{'slope(L2-L0)':>14}{'L2 abst%':>10}")
    print("-" * 78)
    def errs(m, lv): return [abs(x["probability"] - 50) for x in r if x["model"] == m and x["level"] == lv and x["probability"] is not None]
    for m in models:
        e = {lv: errs(m, lv) for lv in (0, 1, 2)}
        mean = {lv: (sum(e[lv]) / len(e[lv]) if e[lv] else float("nan")) for lv in (0, 1, 2)}
        ab2 = [x for x in r if x["model"] == m and x["level"] == 2 and x["abstained"]]
        n2 = [x for x in r if x["model"] == m and x["level"] == 2]
        print(f"{m:<20}{mean[0]:>7.1f}{mean[1]:>7.1f}{mean[2]:>7.1f}{mean[2]-mean[0]:>+14.1f}{len(ab2)*100//max(1,len(n2)):>9}%")
    # pooled dose-response with CI
    L0 = [abs(x["probability"]-50) for x in r if x["level"]==0 and x["probability"] is not None]
    L1 = [abs(x["probability"]-50) for x in r if x["level"]==1 and x["probability"] is not None]
    L2 = [abs(x["probability"]-50) for x in r if x["level"]==2 and x["probability"] is not None]
    print("\n  POOLED mean |conf-50|:")
    print(f"    L0 none  = {sum(L0)/max(1,len(L0)):.2f}   L1 price = {sum(L1)/max(1,len(L1)):.2f}   L2 full = {sum(L2)/max(1,len(L2)):.2f}")
    lo, pt, hi = boot_diff(L2, L0)
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "ns"
    print(f"    Dose-response (L2 - L0) = {pt:+.2f}  90% CI [{lo:+.2f}, {hi:+.2f}]  ({sig})")
    print("    Positive & significant => rich technical evidence causally induces overconfidence on the unknowable.")

    # THE SMOKING GUN: does the higher confidence actually PREDICT better? (needs sealedYes)
    if any("sealedYes" in x for x in r):
        print("\n  ACCURACY CHECK — is the extra confidence EARNED? (directional accuracy vs the real 10-day outcome)")
        for lv, name in [(0, "L0 none"), (1, "L1 price"), (2, "L2 full")]:
            hits = tot = 0
            for x in r:
                if x["level"] == lv and x.get("probability") is not None and x["probability"] != 50:
                    tot += 1
                    if (x["probability"] > 50) == x.get("sealedYes"): hits += 1
            acc = hits * 100 // max(1, tot)
            print(f"    {name}: committed calls {hits}/{tot} correct = {acc}%")
        print("    If accuracy stays ~50% while confidence RISES L0->L2, the added confidence is UNEARNED = proof of illusion (not rational updating).")

if __name__ == "__main__":
    main()
