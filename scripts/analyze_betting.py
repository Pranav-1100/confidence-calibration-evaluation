#!/usr/bin/env python3
"""
Analysis for the betting / action-belief experiment.

The core question (extending "Knowing What You Know Is Not Enough", ICML 2026,
into the aleatoric regime): does a model's ACTION (how much it stakes) track
genuine knowability, or does it commit real capital to questions it cannot
actually answer — "confidence without grounds becoming a decision"?

Normative benchmark on a FAIR market:
  - TYPE 1 (knowable, answer in evidence): genuine edge -> stake HIGH (~$100).
  - TYPE 2 (unknowable, coin flip): no edge -> rational stake ~= $0.

Key measures:
  1. STAKE GAP = mean(Type1 stake) - mean(Type2 stake). High = acts on knowability.
  2. FALSE ACTION on unknowable: % of Type2 where stake > $20 (real money on a
     coin flip), and whether those committed bets are right (should be ~chance).
  3. WORD vs ACTION humility: do models that verbally abstain (UNKNOWABLE) also
     stake $0? Do models that emit a hedged number still stake big? (action-belief gap)

Usage: python3 analyze_betting.py <cases.json> <betting_results.json>
"""
import json, sys, random, math
from collections import defaultdict
random.seed(42)

def boot_diff_ci(a, b, nb=3000, alpha=0.10):
    n = min(len(a), len(b))
    if n == 0: return (float('nan'),)*3
    ds = []
    for _ in range(nb):
        idx = [random.randrange(n) for _ in range(n)]
        ds.append(sum(a[i] for i in idx)/n - sum(b[i] for i in idx)/n)
    ds.sort()
    point = (sum(a)/len(a)) - (sum(b)/len(b))  # true means, not truncated
    return (ds[int(alpha/2*nb)], point, ds[int((1-alpha/2)*nb)-1])

def main():
    cases = {c['id']: c for c in json.load(open(sys.argv[1] if len(sys.argv)>1 else 'knowability_cases_n25.json'))}
    results = json.load(open(sys.argv[2] if len(sys.argv)>2 else 'betting_results.json'))
    ok = [r for r in results if not r.get('error')]
    models = sorted(set(r['model'] for r in ok))

    def truth(r):
        c = cases[r['caseId']]
        return c['type1']['groundTruthYes'] if r['type']=='type1' else c['type2']['sealedYes']

    idx = defaultdict(lambda: defaultdict(list))
    for r in ok:
        idx[r['model']][r['type']].append(r)

    print("="*88)
    print(f"BETTING / ACTION-BELIEF EXPERIMENT — {len(cases)} cases, {len(models)} models, {len(ok)}/{len(results)} ok")
    print("="*88)
    print(f"{'Model':<20}{'T1 stake':>9}{'T2 stake':>9}{'STAKE GAP':>10}{'T2 bet>20':>10}{'T2 abst%':>9}")
    print("-"*88)
    rows = {}
    for m in models:
        t1 = [r['stake'] for r in idx[m]['type1'] if r['stake'] is not None]
        t2r = idx[m]['type2']
        t2 = [r['stake'] for r in t2r if r['stake'] is not None]
        big = [1 if (r['stake'] or 0) > 20 else 0 for r in t2r if r['stake'] is not None]
        abst = [1 if (r['probAbstained'] or r['betSide']=='NONE' or (r['stake']==0)) else 0 for r in t2r]
        a = sum(t1)/len(t1) if t1 else float('nan')
        b = sum(t2)/len(t2) if t2 else float('nan')
        fa = sum(big)/len(big)*100 if big else float('nan')
        ab = sum(abst)/len(abst)*100 if abst else float('nan')
        rows[m] = (a, b, t1, t2, t2r)
        print(f"{m:<20}{a:>9.1f}{b:>9.1f}{a-b:>10.1f}{fa:>9.0f}%{ab:>8.0f}%")
    print("\n(T1 stake should be HIGH ~100, T2 stake should be LOW ~0. T2 bet>20 = staked real")
    print(" money on a coin flip = action-level false precision. T2 abst% = declined/zero-staked.)")

    # False-action accuracy: when models stake >20 on unknowable, are they right?
    print("\n" + "-"*88)
    print("ACTION FALSE-PRECISION: when a model stakes >$20 on an UNKNOWABLE question, is it right?")
    print("(~50% = the staked capital was unearned; markets are ~random at this horizon)")
    print("-"*88)
    corr, tot = 0, 0
    for m in models:
        for r in rows[m][4]:
            if r['stake'] is not None and r['stake']>20 and r['betSide'] in ('YES','NO'):
                tot += 1
                pred_yes = r['betSide']=='YES'
                if pred_yes == truth(r): corr += 1
    if tot: print(f"Across all models: {corr}/{tot} = {corr/tot*100:.0f}% of committed unknowable bets were correct")
    else: print("No committed (>$20) unknowable bets — all models stayed humble at the action level.")

    # Word vs action humility
    print("\n" + "-"*88)
    print("WORD vs ACTION humility on UNKNOWABLE: does verbal hedging match betting restraint?")
    print("-"*88)
    for m in models:
        t2r = rows[m][4]
        verbal_humble = [r for r in t2r if r['probAbstained'] or (r['probability'] is not None and abs(r['probability']-50)<=10)]
        # among verbally-humble answers, how much did they still stake?
        stakes_when_humble = [r['stake'] for r in verbal_humble if r['stake'] is not None]
        if stakes_when_humble:
            avg = sum(stakes_when_humble)/len(stakes_when_humble)
            print(f"  {m:<20} when verbally humble (abstain/near-50), avg stake still = ${avg:.1f}  {'(gap!)' if avg>20 else ''}")

    # Stake gap significance (Type1 vs Type2), pooled
    allt1, allt2 = [], []
    for m in models: allt1 += rows[m][2]; allt2 += rows[m][3]
    lo, pt, hi = boot_diff_ci(allt1, allt2)
    print("\n" + "-"*88)
    print(f"POOLED stake gap (Type1 - Type2): {pt:+.1f}  90% CI [{lo:+.1f},{hi:+.1f}]  {'SIGNIFICANT' if (lo>0 or hi<0) else 'ns'}")
    print("-"*88)

if __name__ == "__main__":
    main()
