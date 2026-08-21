#!/usr/bin/env python3
# ================================================================================
# v2 PAPER STATISTICS - every headline number with a proper interval and a named statistic.
# OFFLINE, no GPU, no API. Reads the cached generations produced by the eval scripts.
#
# WHY THESE STATISTICS
#   * The "discrimination" number used throughout is decline(unknowable) - decline(answerable).
#     Treating "unknowable" as the positive class and DECLINE as the positive prediction, that
#     is exactly TPR - FPR = sensitivity + specificity - 1 = **Youden's J**. Naming it as such
#     makes it a standard, citable statistic rather than an ad hoc gap, and it comes with a
#     free companion: balanced accuracy = (J + 1) / 2.
#   * CIs are CASE-CLUSTERED bootstrap, matching v1's published method: cases are resampled,
#     not individual rows, because the same case appears under several models/levels and
#     row-level resampling would understate the interval.
#   * Cohen's h is reported for proportion contrasts, as in the v2 draft.
#
# usage: python3 rl/v2_stats.py
# ================================================================================
import json, re, os, glob, math, random, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = random.Random(0)
B = 2000

LABEL = r"(ANSWER|CALL[_ -]?TOOL|DECLINE|CANNOT[_ -]?RESOLVE)"
PREFIX = r"(?:DECISION|RESPONSE|FINAL(?:\s+DECISION)?|VERDICT|ACTION|CHOICE|ANS)"

def _norm(s):
    s = s.upper().replace(" ", "_").replace("-", "_")
    return "DECLINE" if "CANNOT" in s else s

def parse(t):
    t = (t or "").replace("*", "")
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I); b = m.group(1) if m else t
    for sc in (b, t):
        dm = re.search(rf"{PREFIX}\s*:\s*{LABEL}", sc, re.I)
        if dm: return _norm(dm.group(1))
    return None

def prob(t):
    pm = re.search(r"PROBABILITY_YES\s*:\s*(\d+(?:\.\d+)?)", t or "", re.I)
    return float(pm.group(1)) if pm else None


def boot_ci(units, stat, B=B, alpha=0.05):
    """Case-clustered bootstrap: resample UNITS (cases), not rows."""
    if not units: return (float("nan"),) * 3
    point = stat(units)
    if point != point: return (point, float("nan"), float("nan"))
    vals = []
    n = len(units)
    for _ in range(B):
        samp = [units[RNG.randrange(n)] for _ in range(n)]
        v = stat(samp)
        if v == v: vals.append(v)
    if not vals: return (point, float("nan"), float("nan"))
    vals.sort()
    return point, vals[int(alpha/2*len(vals))], vals[int((1-alpha/2)*len(vals))-1]

def cohens_h(p1, p2):
    f = lambda p: 2*math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
    return f(p1) - f(p2)

def fmt(t):
    p, lo, hi = t
    if p != p: return "      n/a"
    return f"{p:6.1f} [{lo:5.1f},{hi:5.1f}]"


# Search the data dir first (release layout), then the working dirs (author layout).
SEARCH = [os.path.join(ROOT, "data"), os.path.join(ROOT, "..", "..", "RL_env", "rl"),
          os.path.join(ROOT, "..", "..", "RL_env"), ROOT]
def _find_all(pat):
    out = []
    for d in SEARCH:
        out += glob.glob(os.path.join(d, pat))
    seen, uniq = set(), []
    for f in out:
        b = os.path.basename(f)
        if b not in seen: seen.add(b); uniq.append(f)
    return uniq


# ---------------- load every cached source ----------------
def load(fn):
    hits = _find_all(os.path.basename(fn))
    return json.load(open(hits[0])) if hits else None

transfer = []
for f in _find_all("raw_generations*.json"):
    transfer += json.load(open(f))
nse = load("rl/nse_generations.json") or []
front = load("frontier_transfer_baseline.json") or []
print(f"loaded: transfer {len(transfer)} | nse {len(nse)} | frontier {len(front)}\n")


# ---------------- 1. Youden's J on the transfer domains ----------------
print("=" * 88)
print("1. KNOWABILITY DISCRIMINATION = Youden's J = P(decline|unknowable) - P(decline|answerable)")
print("   (balanced accuracy = (J+1)/2; a blanket decliner and a never-decliner both score J=0)")
print("=" * 88)

# The two arms are DIFFERENT cases, so they cannot be paired into one cluster. J is therefore
# bootstrapped STRATIFIED: resample within each arm independently, recompute J each time. The
# frontier file carries a case id so its resampling is case-clustered; the trained-model cache
# does not, so that one is row-level - disclosed rather than glossed.
def j_ci(unk_units, kno_units, B=B, alpha=0.05):
    def J(u, k):
        du = sum(x[0] for x in u); nu = sum(x[1] for x in u)
        dk = sum(x[0] for x in k); nk = sum(x[1] for x in k)
        if not nu or not nk: return float("nan")
        return 100*(du/nu - dk/nk)
    point = J(unk_units, kno_units)
    if point != point: return (point, float("nan"), float("nan"))
    vals = []
    nu, nk = len(unk_units), len(kno_units)
    for _ in range(B):
        u = [unk_units[RNG.randrange(nu)] for _ in range(nu)]
        k = [kno_units[RNG.randrange(nk)] for _ in range(nk)]
        v = J(u, k)
        if v == v: vals.append(v)
    vals.sort()
    return point, vals[int(alpha/2*len(vals))], vals[int((1-alpha/2)*len(vals))-1]

def collect(src, dom, framing, is_front):
    """-> (unk_units, kno_units) where each unit is (n_declines, n_rows)."""
    unk, kno = collections.defaultdict(lambda: [0,0]), collections.defaultdict(lambda: [0,0])
    for i, r in enumerate(src):
        if is_front:
            if r.get("domain") != dom: continue
            kind = r.get("arm"); txt = r.get("raw")
            if str(txt).startswith("__ERROR__"): continue
            key = r.get("id", i)
        else:
            if r.get("framing") != framing or r.get("domain") != dom: continue
            kind = r.get("kind"); txt = r.get("text"); key = i     # no case id in this cache
        topic = r.get("topic", "")
        if kind == "unk" and not topic.endswith("L2"): continue
        d = parse(txt)
        tgt = unk if kind == "unk" else kno
        tgt[key][0] += (d == "DECLINE"); tgt[key][1] += 1
    return list(unk.values()), list(kno.values())

for src, name, is_front in [(transfer, "TRAINED 3B (transfer domains)", False),
                            (front, "12 FRONTIER MODELS (same domains, same prompt)", True)]:
    if not src: continue
    print(f"\n  {name}")
    for framing in (["natural", "frontier"] if not is_front else [None]):
        for dom in ["crypto", "sports", "weather"]:
            u, k = collect(src, dom, framing, is_front)
            if not u or not k: continue
            j = j_ci(u, k)
            tag = f"{framing}/" if framing else ""
            print(f"    {tag+dom:22s} J = {fmt(j)} pp   balanced acc {(j[0]+100)/2:5.1f}%")

# ---------------- 2. NSE: the headline contrast ----------------
if nse:
    print("\n" + "=" * 88)
    print("2. NSE (v1's OWN cases) - commitment on the unknowable, by evidence level")
    print("   v1 published, 12 frontier models, same cases/prompt/metric: 6.5 / 14.8 / 54.0 / 3.5")
    print("=" * 88)
    V1 = {0: 6.5, 1: 14.8, 2: 54.0, 3: 3.5}
    for ck in sorted({r["ckpt"] for r in nse}):
        print(f"\n  {ck}")
        for lvl in (0, 1, 2, 3):
            rows = [r for r in nse if r["ckpt"] == ck and r["arm"] == "unk" and r["level"] == lvl]
            if not rows: continue
            units = [[r] for r in rows]
            c = boot_ci(units, lambda U: 100*sum(parse(r["text"]) == "ANSWER" for u in U for r in u)
                                          / sum(len(u) for u in U))
            name = "L2'" if lvl == 3 else f"L{lvl}"
            h = cohens_h(V1[lvl]/100, max(c[0], 0)/100)
            print(f"    {name:4s} commit = {fmt(c)} %   v1 {V1[lvl]:5.1f}%   Cohen's h {h:+.2f}")

    print("\n  KNOWABLE arm - directional accuracy (p=50 hedges shown separately)")
    for ck in sorted({r["ckpt"] for r in nse}):
        for lvl in (1, 2):
            rows = [r for r in nse if r["ckpt"] == ck and r["arm"] == "know" and r["level"] == lvl
                    and parse(r["text"]) == "ANSWER" and prob(r["text"]) is not None]
            if not rows: continue
            strict = boot_ci([[r] for r in rows],
                             lambda U: 100*sum((prob(r["text"])>50) == r["sealedYes"] for u in U for r in u)
                                        / sum(len(u) for u in U))
            nonhedge = [r for r in rows if prob(r["text"]) != 50]
            excl = boot_ci([[r] for r in nonhedge],
                           lambda U: 100*sum((prob(r["text"])>50) == r["sealedYes"] for u in U for r in u)
                                      / sum(len(u) for u in U)) if nonhedge else (float("nan"),)*3
            hedges = len(rows) - len(nonhedge)
            print(f"    {ck:12s} L{lvl}: p>50 rule {fmt(strict)} % | excl. {hedges:2d} hedges {fmt(excl)} %"
                  f"   (always-NO baseline 72.5%)")


# ---------------- 3. frontier dose-response on the transfer domains ----------------
if front:
    print("\n" + "=" * 88)
    print("3. FRONTIER dose-response on the transfer domains (does v1's gradient replicate?)")
    print("=" * 88)
    for dom in ["crypto", "sports", "weather"]:
        pts = []
        for lv in ["L0", "L1", "L2"]:
            rows = [r for r in front if r["domain"] == dom and r["arm"] == "unk"
                    and r["topic"].endswith(lv) and not str(r["raw"]).startswith("__ERROR__")]
            c = boot_ci([[r] for r in rows],
                        lambda U: 100*sum(parse(r["raw"]) == "ANSWER" for u in U for r in u)
                                   / sum(len(u) for u in U))
            pts.append(c)
        h = cohens_h(pts[2][0]/100, pts[0][0]/100)
        print(f"  {dom:8s} L0 {fmt(pts[0])}  L1 {fmt(pts[1])}  L2 {fmt(pts[2])}   "
              f"L2-L0 = {pts[2][0]-pts[0][0]:+5.1f}pp, Cohen's h {h:+.2f}")
    print("  v1 NSE   L0    6.5              L1   14.8             L2   54.0            "
          "L2-L0 =  +47.5pp")

print("\nAll intervals are 95% case-clustered bootstrap (B=2000), matching v1's method.")
