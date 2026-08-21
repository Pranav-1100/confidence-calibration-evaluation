#!/usr/bin/env python3
# ================================================================================
# STATISTICS ADDED AFTER INDEPENDENT REVIEW. OFFLINE, no GPU, no API.
#
# Each block here exists because a specific criticism landed:
#   1. BRIER DECOMPOSITION replaces the ECE-alone argument. ECE at 5 bins gives 0.043 and at
#      the field-standard 10-15 bins gives 0.16-0.21 on the same data, so the "standard metrics
#      say fine" claim inverted with an arbitrary choice. Brier decomposes into reliability +
#      resolution and is bin-free: reliability ~0 with resolution ~0 IS "sounds calibrated,
#      says nothing", which is the claim, stated in a way that cannot be tuned.
#   2. ECE AT SEVERAL BIN COUNTS, reported rather than hidden.
#   3. WILSON INTERVALS on the 0% cells - 0/40 is not "0%", it is [0, 8.8%].
#   4. BOOTSTRAP CI ON COHEN'S H, which was quoted as a bare point estimate.
#   5. TENSE COMPOSITION of every evaluation set, so the confound is documented numerically.
#
# usage: python3 rl/v2_stats_addendum.py
# ================================================================================
import json, os, re, math, random, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MR = os.path.join(ROOT, "..", "main-research", "data")
RNG = random.Random(0)

def load(*p):
    f = os.path.join(*p)
    return json.load(open(f)) if os.path.exists(f) else None


# ---------------------------------------------------------------- 1 + 2. calibration
def brier_decomposition(pairs, bins=10):
    """pairs = [(p in [0,1], outcome in {0,1})].
    Murphy decomposition: Brier = reliability - resolution + uncertainty."""
    n = len(pairs)
    obar = sum(o for _, o in pairs) / n
    unc = obar * (1 - obar)
    buckets = collections.defaultdict(list)
    for p, o in pairs:
        k = min(int(p * bins), bins - 1)
        buckets[k].append((p, o))
    rel = res = 0.0
    for k, grp in buckets.items():
        nk = len(grp)
        pbar = sum(p for p, _ in grp) / nk
        okbar = sum(o for _, o in grp) / nk
        rel += nk * (pbar - okbar) ** 2
        res += nk * (okbar - obar) ** 2
    rel /= n; res /= n
    brier = sum((p - o) ** 2 for p, o in pairs) / n
    return brier, rel, res, unc

def ece(pairs, bins):
    n = len(pairs)
    buckets = collections.defaultdict(list)
    for p, o in pairs:
        buckets[min(int(p * bins), bins - 1)].append((p, o))
    return sum(len(g) / n * abs(sum(p for p, _ in g)/len(g) - sum(o for _, o in g)/len(g))
               for g in buckets.values())

print("=" * 84)
print("1-2. CALIBRATION: Brier decomposition (bin-free) vs ECE (bin-fragile)")
print("=" * 84)
for label, fn, lvl in [("NSE L2 committed calls (12 models)", "agentic_postcutoff.json", "2")]:
    d = load(MR, fn)
    pairs = [(x["probability"]/100.0, 1 if x["sealedYes"] else 0) for x in d
             if str(x.get("level")) == lvl and (x.get("decision") or "").upper() == "ANSWER"
             and x.get("probability") is not None]
    b, rel, res, unc = brier_decomposition(pairs)
    print(f"\n  {label}  (n={len(pairs)})")
    print(f"    Brier            {b:.4f}   vs 0.2500 for uniformly answering 50%")
    print(f"    reliability      {rel:.4f}   (0 = stated probabilities are honest)")
    print(f"    RESOLUTION       {res:.4f}   (0 = the probabilities distinguish NOTHING)")
    print(f"    uncertainty      {unc:.4f}   (base-rate variance, the irreducible floor)")
    print(f"    check: rel - res + unc = {rel-res+unc:.4f}")
    print(f"    ECE by bin count:", "  ".join(f"M={m}: {ece(pairs,m):.3f}" for m in (5, 10, 15)))
    print(f"    -> resolution ~ {res:.3f} is the paper's claim, and it does not move with bins.")


# ---------------------------------------------------------------- 3. Wilson intervals
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, 100*(c-h)), min(100.0, 100*(c+h)))

print("\n" + "=" * 84)
print("3. WILSON 95% INTERVALS on the zero cells (0/40 is not '0%')")
print("=" * 84)
nse = load(ROOT, "rl", "nse_generations.json")
if nse:
    LABEL = r"(ANSWER|CALL[_ -]?TOOL|DECLINE|CANNOT[_ -]?RESOLVE)"
    PREFIX = r"(?:DECISION|RESPONSE|FINAL(?:\s+DECISION)?|VERDICT|ACTION|CHOICE|ANS)"
    def parse(t):
        t=(t or "").replace("*","")
        m=re.search(r"<answer>(.*?)</answer>",t,re.S|re.I); b=m.group(1) if m else t
        for sc in (b,t):
            dm=re.search(rf"{PREFIX}\s*:\s*{LABEL}",sc,re.I)
            if dm:
                s=dm.group(1).upper().replace(" ","_").replace("-","_")
                return "DECLINE" if "CANNOT" in s else s
        return None
    for ck in sorted({r["ckpt"] for r in nse}):
        for lvl in (0,1,2,3):
            rows=[r for r in nse if r["ckpt"]==ck and r["arm"]=="unk" and r["level"]==lvl]
            if not rows: continue
            k=sum(parse(r["text"])=="ANSWER" for r in rows)
            lo,hi=wilson(k,len(rows))
            name = "L2'" if lvl==3 else f"L{lvl}"
            print(f"  {ck:12s} {name:4s} commit {k}/{len(rows)} = {100*k/len(rows):.1f}%  "
                  f"95% CI [{lo:.1f}, {hi:.1f}]")
    print("\n  -> report as '0% (95% CI [0, 8.8])', not a bare 0%.")


# ---------------------------------------------------------------- 4. bootstrap CI on Cohen's h
def cohens_h(p1, p2):
    f = lambda p: 2*math.asin(math.sqrt(min(max(p,0.0),1.0)))
    return f(p1)-f(p2)

print("\n" + "=" * 84)
print("4. COHEN'S h WITH A BOOTSTRAP INTERVAL (was quoted as a bare point estimate)")
print("=" * 84)
d = load(MR, "agentic_postcutoff.json")
if d and nse:
    fr = [1 if (x.get("decision") or "").upper()=="ANSWER" else 0
          for x in d if str(x.get("level"))=="2"]
    tr = [1 if parse(r["text"])=="ANSWER" else 0
          for r in nse if r["ckpt"]=="SFT-2" and r["arm"]=="unk" and r["level"]==2]
    point = cohens_h(sum(fr)/len(fr), sum(tr)/len(tr))
    vals=[]
    for _ in range(2000):
        a=[fr[RNG.randrange(len(fr))] for _ in range(len(fr))]
        b=[tr[RNG.randrange(len(tr))] for _ in range(len(tr))]
        vals.append(cohens_h(sum(a)/len(a), sum(b)/len(b)))
    vals.sort()
    print(f"  frontier L2 {100*sum(fr)/len(fr):.1f}%  vs trained {100*sum(tr)/len(tr):.1f}%")
    print(f"  Cohen's h = {point:+.2f}  95% CI [{vals[50]:+.2f}, {vals[-51]:+.2f}]")


# ---------------------------------------------------------------- 5. tense composition
print("\n" + "=" * 84)
print("5. TENSE COMPOSITION OF EVERY EVALUATION SET (documents the confound)")
print("=" * 84)
for dom in ["crypto","sports","weather"]:
    u=load(ROOT,f"{dom}_seduction_eval.json"); k=load(ROOT,f"{dom}_answerable_eval.json")
    if not u or not k: continue
    cu=collections.Counter(x.get("tense") for x in u)
    ck2=collections.Counter(x.get("tense") for x in k)
    print(f"  {dom:8s} unknowable {dict(cu)}   answerable {dict(ck2)}")
ke=load(ROOT,"knowability_eval.json")
if ke:
    print(f"\n  knowability_eval.json (the tense-balanced instrument, NOT YET RUN):")
    for key,n in sorted(collections.Counter((x['tense'],x['label']) for x in ke).items()):
        print(f"    {key[0]:8s}/{key[1]:12s} n={n}")
print("\n" + "=" * 84)
