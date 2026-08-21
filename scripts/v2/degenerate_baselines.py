#!/usr/bin/env python3
# ================================================================================
# DEGENERATE-STRATEGY BASELINES. OFFLINE, no GPU, no API.
#
# For every headline evaluation, what score would a strategy with no understanding get?
#   always-DECLINE      refuses everything
#   always-ANSWER       commits to everything
#   always-NO           answers, always predicting NO
#   TENSE RULE          declines iff the question is in the future tense
#
# The last one is the point. Every evaluation in the paper separates unknowable from
# answerable by grammatical tense (unknowable 100% future, answerable 100% present), so the
# tense rule scores a perfect Youden's J while understanding nothing. Any headline number a
# degenerate strategy can match is not evidence for the paper's claim.
#
# usage: python3 rl/degenerate_baselines.py
# ================================================================================
import json, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(p):
    f = os.path.join(ROOT, p)
    return json.load(open(f)) if os.path.exists(f) else None

def J(decline_unk, decline_know):
    return 100*decline_unk - 100*decline_know

print("=" * 86)
print("DEGENERATE-STRATEGY BASELINES")
print("=" * 86)

# ---- transfer domains: what does each dumb strategy score? ----
print("\n1. TRANSFER DOMAINS (crypto / sports / weather)\n")
for dom in ["crypto", "sports", "weather"]:
    unk = load(f"{dom}_seduction_eval.json")
    kno = load(f"{dom}_answerable_eval.json")
    if not unk or not kno: continue
    unk2 = [c for c in unk if c["topic"].endswith("L2")]
    fu = sum(1 for c in unk2 if c.get("tense") == "future") / len(unk2)
    fk = sum(1 for c in kno if c.get("tense") == "future") / len(kno)
    print(f"  {dom}: unknowable-L2 future={100*fu:.0f}%  answerable future={100*fk:.0f}%")
    print(f"    {'strategy':22s} {'J (pp)':>8s}   note")
    print(f"    {'always-DECLINE':22s} {J(1.0,1.0):+8.0f}   refuses everything")
    print(f"    {'always-ANSWER':22s} {J(0.0,0.0):+8.0f}   commits to everything")
    print(f"    {'TENSE RULE':22s} {J(fu,fk):+8.0f}   <-- declines iff future tense")
    print(f"    {'trained 3B (natural)':22s} {'+83..+100':>8s}   indistinguishable from the tense rule\n")

# ---- NSE: both arms, plus the class-imbalance baseline ----
print("\n2. NSE CASES (the paper's flagship transfer test)\n")
c = load(os.path.join("..", "main-research", "data", "knowability_postcutoff.json"))
if c:
    yes = sum(1 for x in c if x["type1"]["groundTruthYes"])
    print(f"  answerable arm ground truth: {len(c)-yes} NO / {yes} YES")
    print(f"    {'always-NO on answerable':30s} {100*(len(c)-yes)/len(c):5.1f}% accuracy  "
          f"<-- the bar the trained model must beat")
    print(f"    {'trained 3B, p>50 rule':30s} {77.5:5.1f}% accuracy  (+5.0pp over always-NO)")
    print(f"    {'trained 3B, hedges excluded':30s} {100.0:5.1f}% accuracy  (on 24 of 40 items)")
    print(f"\n  unknowable arm: type2 is 100% future tense, type1 is 100% present tense")
    print(f"    {'TENSE RULE':30s} J = +100 pp  <-- scores perfectly with no knowability")

# ---- what the tense-balanced eval would show ----
print("\n3. THE INSTRUMENT THAT BREAKS THE CONFOUND (not yet run)\n")
k = load("knowability_eval.json")
if k:
    cnt = collections.Counter((x["tense"], x["label"]) for x in k)
    print(f"  knowability_eval.json: {len(k)} items")
    for key in sorted(cnt): print(f"    {key[0]:8s} / {key[1]:12s}  n={cnt[key]}")
    af = cnt[("future", "answerable")]; up = cnt[("present", "unknowable")]
    print(f"\n  The tense rule gets ALL {af} answerable-FUTURE items wrong (declines them)")
    print(f"  and ALL {up} unknowable-PRESENT items wrong (answers them).")
    print(f"  Within-tense J for a pure tense rule = 0 in both tenses, by construction.")
    print(f"  -> running this eval separates the two hypotheses cleanly. It is the decisive test.")

print("\n" + "=" * 86)
print("HOW TO USE THIS IN THE PAPER: report it as a table next to the headline results.")
print("Any metric a degenerate strategy matches is not evidence for the claim being made.")
print("=" * 86)
