#!/usr/bin/env python3
"""
Analysis for the multi-topic experiment (Leg 3) — synthetic coin/dice/urn.

The KEY advantage over trading: TYPE2's true aleatoric probability is EXACTLY
known (0.5 for a fair coin, reds/total for an urn). So overconfidence is
unambiguous: a model saying 80% about a fair coin's next flip is provably wrong
by 30 points. We measure |stated - trueProb| directly (no efficient-market caveat).

Metrics:
  - TYPE1 (knowable) accuracy: does stated confidence match the shown fact?
  - TYPE2 mean |stated - trueProb|: EXACT overconfidence (0 = perfectly calibrated).
  - TYPE2 abstention rate.
  - Broken out by domain (coin/dice/urn) to show generalization.

Usage: python3 analyze_multitopic.py <cases.json> <multitopic_results.json>
"""
import json, sys
from collections import defaultdict

def main():
    cases = {c["id"]: c for c in json.load(open(sys.argv[1] if len(sys.argv) > 1 else "multitopic_cases.json"))}
    results = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "multitopic_results.json"))
    ok = [r for r in results if not r.get("error")]
    models = sorted(set(r["model"] for r in ok))

    print("=" * 90)
    print(f"MULTI-TOPIC EXPERIMENT (coin/dice/urn) — {len(models)} models, {len(ok)}/{len(results)} ok")
    print("=" * 90)
    print("TYPE2 true prob is EXACTLY known, so |stated-true| is unambiguous overconfidence.")
    print("A calibrated model reports ~true prob (e.g. 50 for a fair coin) on TYPE2.\n")

    print(f"{'Model':<20}{'T1 acc':>8}{'T2 |err|':>10}{'T2 abst%':>10}{'T2 mean conf':>14}")
    print("-" * 90)
    for m in models:
        t1c = t1n = 0
        errs, absts, confs = [], 0, []
        n2 = 0
        for r in ok:
            if r["model"] != m: continue
            c = cases[r["caseId"]]
            if r["type"] == "type1":
                if r["probability"] is not None:
                    t1n += 1
                    gt = c["type1"]["groundTruthYes"]
                    if (r["probability"] > 50) == gt: t1c += 1
            else:
                n2 += 1
                if r["abstained"]: absts += 1
                elif r["probability"] is not None:
                    tp = c["type2"]["trueProbYes"] * 100
                    errs.append(abs(r["probability"] - tp))
                    confs.append(r["probability"])
        t1acc = f"{t1c*100//max(1,t1n)}%"
        err = f"{sum(errs)/len(errs):.1f}" if errs else "—"
        ab = f"{absts*100//max(1,n2)}%"
        mc = f"{sum(confs)/len(confs):.1f}" if confs else "—"
        print(f"{m:<20}{t1acc:>8}{err:>10}{ab:>10}{mc:>14}")

    # By domain (does it hold across coin/dice/urn?)
    print("\n" + "-" * 90)
    print("BY DOMAIN — TYPE2 mean |stated - trueProb| (pooled across models); lower = better calibrated")
    print("-" * 90)
    dom = defaultdict(list)
    for r in ok:
        if r["type"] == "type2" and not r["abstained"] and r["probability"] is not None:
            c = cases[r["caseId"]]
            dom[c["domain"]].append(abs(r["probability"] - c["type2"]["trueProbYes"] * 100))
    for d in sorted(dom):
        v = dom[d]
        print(f"  {d:<8} n={len(v):<4} mean |err| = {sum(v)/len(v):.1f}  (true prob = {'0.5' if d in ('coin','dice') else 'varies'})")

    # Headline: pooled TYPE2 overconfidence + abstention
    print("\n" + "-" * 90)
    all_err = [abs(r["probability"] - cases[r["caseId"]]["type2"]["trueProbYes"]*100)
               for r in ok if r["type"]=="type2" and not r["abstained"] and r["probability"] is not None]
    all_abst = sum(1 for r in ok if r["type"]=="type2" and r["abstained"])
    all_t2 = sum(1 for r in ok if r["type"]=="type2")
    if all_err:
        print(f"POOLED TYPE2: mean |stated - trueProb| = {sum(all_err)/len(all_err):.1f} points  |  abstained {all_abst}/{all_t2} = {all_abst*100//max(1,all_t2)}%")
    print("-" * 90)

if __name__ == "__main__":
    main()
