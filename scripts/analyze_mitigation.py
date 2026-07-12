#!/usr/bin/env python3
"""
Mitigation trade-off analysis (pre-registered: PREREGISTRATION.md §Mitigation).

The epistemic-triage system prompt is only a WIN if it (a) cuts evidence-induced
commitment on the unknowable AND (b) does not cause false-declining where acting
is correct. Per the pre-registration, BOTH numbers are always reported — the
trade-off curve is the result, not either number alone.

Inputs (4 result files):
  1. agentic BASELINE   (e.g. agentic_postcutoff.json      — no system prompt)
  2. agentic MITIGATED  (e.g. agentic_mitigated.json       — MITIGATION=1, LEVELS=0,2)
  3. discrimination BASELINE  (e.g. discrimination_all.json)
  4. discrimination MITIGATED (e.g. discrimination_mitigated.json)

Reports:
  A. Seduction reduction — commitment (ANSWER%) per level, baseline vs mitigated;
     PAIRED case-clustered 95% CI on Δcommitment(L2) and on Δ(L2−L0 shift).
  B. Overcorrection — per uncertainty type (KNOWN/EPISTEMIC/ALEATORIC): correct-
     decision rate baseline vs mitigated; paired clustered CI on the false-abstain
     rate (CANNOT_RESOLVE where acting is correct) over KNOWN+EPISTEMIC cases.
  C. Triage localization (mitigated agentic only) — every gradient question is
     category-(3) by construction, so: % classified 3 by level (does evidence
     corrupt the CLASSIFICATION?) and, among rows classified 3, % that still
     committed (classify-right-act-wrong = instruction-following failure).

Usage: python3 analyze_mitigation.py <agentic_base> <agentic_mit> <disc_base> <disc_mit>
"""
import json, sys, random
from collections import defaultdict
random.seed(42)

def load(p):
    return [x for x in json.load(open(p)) if not x.get("error") and x.get("decision")]

def paired_cluster_ci(base_rows, mit_rows, val, nb=3000, alpha=0.05):
    """Paired case-clustered bootstrap on mean(val|mit) - mean(val|base).
    Pairs by caseId; resamples the shared case list. Returns (lo, pt, hi) or None."""
    b = defaultdict(list); m = defaultdict(list)
    for x in base_rows: b[x["caseId"]].append(val(x))
    for x in mit_rows: m[x["caseId"]].append(val(x))
    cids = sorted(set(b) & set(m))
    if not cids: return None
    def diff(cs):
        bv = [v for c in cs for v in b[c]]; mv = [v for c in cs for v in m[c]]
        return sum(mv)/len(mv) - sum(bv)/len(bv)
    pt = diff(cids)
    ds = sorted(diff([cids[random.randrange(len(cids))] for _ in cids]) for _ in range(nb))
    return ds[int(alpha/2*nb)], pt, ds[int((1-alpha/2)*nb)-1]

def fmt_ci(cb, unit="pp", scale=100):
    l, pt, h = cb
    sig = "SIGNIFICANT" if (l > 0 or h < 0) else "ns"
    return f"{pt*scale:+.0f}{unit}  95% CI [{l*scale:+.0f}, {h*scale:+.0f}]  ({sig})"

def main():
    if len(sys.argv) != 5:
        print(__doc__); sys.exit(1)
    ab, am, db, dm = (load(p) for p in sys.argv[1:5])

    # ---------- A. seduction reduction (agentic) ----------
    print("=" * 92)
    print("A. SEDUCTION REDUCTION — commitment (ANSWER%) on the unknowable gradient")
    print("=" * 92)
    levels = sorted(set(x["level"] for x in am) & set(x["level"] for x in ab))
    is_ans = lambda x: 1 if x["decision"] == "ANSWER" else 0
    is_act = lambda x: 1 if x["decision"] in ("ANSWER", "CALL_TOOL") else 0
    models = sorted(set(x["model"] for x in am))
    print(f"{'Model':<20}" + "".join(f"{'L%d base'%lv:>10}{'L%d mit'%lv:>9}" for lv in levels))
    for mo in models:
        row = ""
        for lv in levels:
            bb = [x for x in ab if x["model"] == mo and x["level"] == lv]
            mm = [x for x in am if x["model"] == mo and x["level"] == lv]
            row += f"{(sum(map(is_ans,bb))*100//len(bb)) if bb else '-':>9}%{(sum(map(is_ans,mm))*100//len(mm)) if mm else '-':>8}%"
        print(f"{mo:<20}{row}")
    for lv in levels:
        bb = [x for x in ab if x["level"] == lv]; mm = [x for x in am if x["level"] == lv]
        print(f"\n  pooled L{lv}: commitment base {sum(map(is_ans,bb))*100/len(bb):.0f}% -> mitigated {sum(map(is_ans,mm))*100/len(mm):.0f}%"
              f"   (acting {sum(map(is_act,bb))*100/len(bb):.0f}% -> {sum(map(is_act,mm))*100/len(mm):.0f}%)")
    if 2 in levels:
        cb = paired_cluster_ci([x for x in ab if x["level"] == 2], [x for x in am if x["level"] == 2], is_ans)
        print(f"\n  Δcommitment at L2 (mit − base), paired clustered: {fmt_ci(cb)}")
    if 0 in levels and 2 in levels:
        # Δ of the shift: (mitL2−mitL0) − (baseL2−baseL0), clustered over shared cases
        b2 = defaultdict(list); b0 = defaultdict(list); m2 = defaultdict(list); m0 = defaultdict(list)
        for x in ab: (b2 if x["level"] == 2 else b0 if x["level"] == 0 else defaultdict(list))[x["caseId"]].append(is_ans(x))
        for x in am: (m2 if x["level"] == 2 else m0 if x["level"] == 0 else defaultdict(list))[x["caseId"]].append(is_ans(x))
        cids = sorted(set(b2) & set(b0) & set(m2) & set(m0))
        def shiftdiff(cs):
            f = lambda d: (lambda v: sum(v)/len(v))([u for c in cs for u in d[c]])
            return (f(m2) - f(m0)) - (f(b2) - f(b0))
        pt = shiftdiff(cids)
        ds = sorted(shiftdiff([cids[random.randrange(len(cids))] for _ in cids]) for _ in range(3000))
        print(f"  Δ(L2−L0 shift) (mit − base), paired clustered: {fmt_ci((ds[75], pt, ds[2924]))}")
        print("  (negative = the triage prompt reduced evidence-induced commitment)")

    # ---------- B. overcorrection (discrimination) ----------
    print("\n" + "=" * 92)
    print("B. OVERCORRECTION — discrimination set (acting is CORRECT for KNOWN/EPISTEMIC)")
    print("=" * 92)
    correct = {"KNOWN": "ANSWER", "EPISTEMIC": "CALL_TOOL", "ALEATORIC": "CANNOT_RESOLVE"}
    for t, cd in correct.items():
        bb = [x for x in db if x["type"] == t]; mm = [x for x in dm if x["type"] == t]
        if not bb or not mm: print(f"  {t:<10} (missing rows — skipped)"); continue
        cb_rate = sum(1 for x in bb if x["decision"] == cd)*100/len(bb)
        cm_rate = sum(1 for x in mm if x["decision"] == cd)*100/len(mm)
        print(f"  {t:<10} correct({cd}) base {cb_rate:5.1f}% -> mitigated {cm_rate:5.1f}%")
    ans_ok = [x for x in db if x["type"] in ("KNOWN", "EPISTEMIC")]
    ans_ok_m = [x for x in dm if x["type"] in ("KNOWN", "EPISTEMIC")]
    if ans_ok and ans_ok_m:
        false_abst = lambda x: 1 if x["decision"] == "CANNOT_RESOLVE" else 0
        cb = paired_cluster_ci(ans_ok, ans_ok_m, false_abst)
        print(f"\n  Δfalse-abstain on answerable (KNOWN+EPISTEMIC), paired clustered: {fmt_ci(cb)}")
        print("  (positive = the triage prompt causes false-declining where acting is correct = the COST)")

    # ---------- C. triage localization (mitigated agentic) ----------
    print("\n" + "=" * 92)
    print("C. TRIAGE LOCALIZATION — where does the mitigation fail (if it fails)?")
    print("=" * 92)
    has_cat = [x for x in am if x.get("category") is not None]
    print(f"  CATEGORY parsed on {len(has_cat)}/{len(am)} mitigated agentic rows (all questions are category-3 by construction)")
    for lv in sorted(set(x["level"] for x in has_cat)):
        d = [x for x in has_cat if x["level"] == lv]
        c3 = sum(1 for x in d if x["category"] == 3)
        print(f"    L{lv}: classified IRREDUCIBLE(3): {c3*100/len(d):.0f}%   "
              f"(classification corrupted by evidence if this falls L0->L2)")
    said3 = [x for x in has_cat if x["category"] == 3]
    if said3:
        commit3 = sum(1 for x in said3 if x["decision"] == "ANSWER")
        print(f"    classified 3 but STILL committed: {commit3}/{len(said3)} = {commit3*100/len(said3):.0f}%   "
              f"(classify-right-act-wrong = instruction-following failure, not judgment failure)")

if __name__ == "__main__":
    main()
