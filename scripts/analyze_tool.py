#!/usr/bin/env python3
"""
Analysis for the tool-call / "no tool can help" experiment (Leg 2).

Core question: on an UNKNOWABLE (future) question, does the model recognize that
NO data tool can resolve it (CANNOT_RESOLVE) — or does it commit the "tool fallacy"
(CALL_TOOL, thinking a data lookup reveals the future) or fabricate (ANSWER)?

Normative:
  - TYPE 1 (knowable/past): ANSWER or CALL_TOOL both fine (it IS resolvable).
  - TYPE 2 (unknowable/future): CANNOT_RESOLVE is the ONLY correct decision.

Usage: python3 analyze_tool.py <cases.json> <tool_results.json>
"""
import json, sys
from collections import defaultdict

def main():
    results = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "tool_results.json"))
    ok = [r for r in results if not r.get("error") and r.get("decision")]
    models = sorted(set(r["model"] for r in ok))

    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # model->type->decision->count
    for r in ok:
        idx[r["model"]][r["type"]][r["decision"]] += 1

    print("=" * 92)
    print(f"TOOL-CALL EXPERIMENT — {len(models)} models, {len(ok)}/{len(results)} ok")
    print("=" * 92)
    print("On UNKNOWABLE (future) questions, CANNOT_RESOLVE is correct; CALL_TOOL = 'tool fallacy'")
    print("(believing a data lookup reveals the future); ANSWER = fabrication.\n")
    print(f"{'Model':<20} | {'TYPE1 (knowable)':<32} | {'TYPE2 (UNKNOWABLE)':<36}")
    print(f"{'':<20} | {'ANSWER  CALL_TOOL  CANT':<32} | {'ANSWER  CALL_TOOL  CANT(correct)':<36}")
    print("-" * 92)
    for m in models:
        def row(t):
            d = idx[m][t]; n = sum(d.values()) or 1
            return d.get("ANSWER",0), d.get("CALL_TOOL",0), d.get("CANNOT_RESOLVE",0), n
        a1,c1,r1,n1 = row("type1")
        a2,c2,r2,n2 = row("type2")
        print(f"{m:<20} | {a1:>6}  {c1:>9}  {r1:>4}   ({n1:>2})       | {a2:>6}  {c2:>9}  {r2:>4}={r2*100//max(1,n2):>3}% correct  ({n2})")

    # Aggregate: the headline — on unknowable, what fraction correctly say CANNOT_RESOLVE?
    print("\n" + "-" * 92)
    print("HEADLINE — on UNKNOWABLE questions, decision distribution (pooled across models):")
    print("-" * 92)
    agg = defaultdict(int)
    for r in ok:
        if r["type"] == "type2": agg[r["decision"]] += 1
    tot = sum(agg.values()) or 1
    for dec in ("CANNOT_RESOLVE", "CALL_TOOL", "ANSWER"):
        lab = {"CANNOT_RESOLVE":"CANNOT_RESOLVE (correct — no tool helps)","CALL_TOOL":"CALL_TOOL (tool fallacy)","ANSWER":"ANSWER (fabricates)"}[dec]
        print(f"  {lab:<45} {agg.get(dec,0):>4} / {tot}  = {agg.get(dec,0)*100//tot}%")

    # Contrast: TYPE1 should almost never be CANNOT_RESOLVE (that'd be over-refusal)
    t1_cant = sum(1 for r in ok if r["type"]=="type1" and r["decision"]=="CANNOT_RESOLVE")
    t1_tot = sum(1 for r in ok if r["type"]=="type1") or 1
    print(f"\n  Sanity: TYPE1 (knowable) wrongly marked CANNOT_RESOLVE: {t1_cant}/{t1_tot} = {t1_cant*100//t1_tot}% (should be ~0 = no over-refusal)")

if __name__ == "__main__":
    main()
