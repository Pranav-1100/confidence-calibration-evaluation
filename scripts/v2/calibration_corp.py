# Reproduces the §5 calibration numbers from data/agentic_postcutoff.json:
# CORP decomposition (isotonic, bin-free), Brier Skill Score, AUROC by evidence level,
# the 55-65% / 35-45% bands, directional accuracy, and mean stated probability vs base rate.
# usage: python3 scripts/v2/calibration_corp.py
import json, os

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
rows = json.load(open(os.path.join(DATA, "agentic_postcutoff.json")))

def committed(level):
    return [(r["probability"] / 100, 1 if r["sealedYes"] else 0) for r in rows
            if str(r["level"]) == level and (r["decision"] or "").upper() == "ANSWER" and r["probability"] is not None]

def auroc(pairs):
    pos = [p for p, o in pairs if o]; neg = [p for p, o in pairs if not o]
    return (sum(a > b for a in pos for b in neg) + 0.5 * sum(a == b for a in pos for b in neg)) / (len(pos) * len(neg))

def isotonic(pairs):
    # pool-adjacent-violators on outcomes ordered by stated probability
    pairs = sorted(pairs); blocks = []
    for p, o in pairs:
        blocks.append([o, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            o2, n2 = blocks.pop(); blocks[-1][0] += o2; blocks[-1][1] += n2
    fitted = [o / n for o, n in blocks for _ in range(n)]
    return [(o, f) for (_, o), f in zip(pairs, fitted)]

for lv in "012":
    print(f"L{lv}: committed calls n={len(committed(lv))}, AUROC={auroc(committed(lv)):.3f}")

pr = committed("2"); n = len(pr)
obar = sum(o for _, o in pr) / n; unc = obar * (1 - obar)
bs = sum((p - o) ** 2 for p, o in pr) / n
bs_iso = sum((f - o) ** 2 for o, f in isotonic(pr)) / n
print(f"\nL2 CORP: MCB {bs - bs_iso:.4f} - DSC {unc - bs_iso:.4f} + UNC {unc:.4f} = {bs:.4f}")
print(f"Brier Skill Score vs climatology: {1 - bs / unc:.3f}")
print(f"mean stated probability {100 * sum(p for p, _ in pr) / n:.1f}% vs base rate {100 * obar:.1f}%")
for lo, hi in ((0.55, 0.65), (0.35, 0.45)):
    s = [o for p, o in pr if lo <= p <= hi]
    print(f"stated {int(lo * 100)}-{int(hi * 100)}%: event occurs {100 * sum(s) / len(s):.1f}% (n={len(s)})")
nh = [(p, o) for p, o in pr if abs(p - 0.5) > 1e-9]
print(f"directional accuracy, non-hedged calls: {100 * sum((p > 0.5) == bool(o) for p, o in nh) / len(nh):.1f}% (n={len(nh)})")
