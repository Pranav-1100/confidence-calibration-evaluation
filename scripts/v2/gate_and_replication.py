# Reproduces two numbers the paper reports from data that were previously not released:
#   §5  elicited-edge separability (AUROC 0.887, best-threshold 83.7%, own decisions 67.9% / 73.9%)
#       -> data/v2_gate_free.json + data/labeled_balanced.json
#   §3  48-event replication, scrambled minus real (+2.60pp, 90% CI [-0.66, +5.86])
#       -> data/powered_run.json (cases in data/powered_cases.json)
# usage: python3 scripts/v2/gate_and_replication.py
import json, os, re, random, collections
from statistics import mean

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
load = lambda f: json.load(open(os.path.join(DATA, f)))

# ------------------------------------------------------------ §5 separability
lab = {c["id"]: c for c in load("labeled_balanced.json")}
rows = [x for x in load("v2_gate_free.json") if not x.get("error") and x.get("decision")]
pairs = [(x, lab[x["caseId"]]) for x in rows if x["caseId"] in lab]
is_ans = lambda c: c["label"] == "answerable"
pos = [x["confidence"] for x, c in pairs if is_ans(c) and x.get("confidence") is not None]
neg = [x["confidence"] for x, c in pairs if not is_ans(c) and x.get("confidence") is not None]
auc = (sum(a > b for a in pos for b in neg) + 0.5 * sum(a == b for a in pos for b in neg)) / (len(pos) * len(neg))
best = max((sum(a >= t for a in pos) + sum(b < t for b in neg), t) for t in range(101))
own = mean(1 if (x["decision"] == "ANSWER" if is_ans(c) else x["decision"] in ("DECLINE", "CALL_TOOL")) else 0
           for x, c in pairs)
noct = [(x, c) for x, c in pairs if x["decision"] != "CALL_TOOL"]
own_noct = mean(1 if (x["decision"] == "ANSWER") == is_ans(c) else 0 for x, c in noct)
print("§5 separability (six open-weight models)")
print(f"  labeled rows analysed: {len(pos)} answerable + {len(neg)} unknowable = {len(pos) + len(neg)} (of {len(load('v2_gate_free.json'))} responses)")
print(f"  AUROC = {auc:.3f}   best-threshold accuracy = {100 * best[0] / (len(pos) + len(neg)):.1f}% (edge >= {best[1]})")
print(f"  models' own decisions = {100 * own:.1f}%  (tool calls scored as errors: {100 * own_noct:.1f}% excluding them)")

# ------------------------------------------------------------ §3 48-event replication
LABEL = r"(ANSWER|CALL[_ -]?TOOL|DECLINE|CANNOT[_ -]?RESOLVE)"
PREFIX = r"(?:DECISION|RESPONSE|FINAL(?:\s+DECISION)?|VERDICT|ACTION|CHOICE|ANS)"
def decision(t):
    t = (t or "").replace("*", "")
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I)
    for scope in ((m.group(1) if m else t), t):
        d = re.search(rf"{PREFIX}\s*:\s*{LABEL}", scope, re.I)
        if d: return "DECLINE" if "CANNOT" in d.group(1).upper() else d.group(1).upper().replace(" ", "_").replace("-", "_")
    return None

run = load("powered_run.json")
event = lambda r: r["id"].rsplit("_", 1)[0]
def replicate(rs, label):
    by = collections.defaultdict(list)
    for r in rs: by[event(r)].append(r)
    ev = sorted(by)
    def diff(evs):
        s = [r for e in evs for r in by[e]]
        rate = lambda a: sum(decision(r["raw"]) == "ANSWER" for r in s if r["arm"] == a) / sum(r["arm"] == a for r in s)
        return 100 * (rate("scram") - rate("rich"))
    rng = random.Random(0)
    bs = sorted(diff([rng.choice(ev) for _ in ev]) for _ in range(4000))
    print(f"  {label}: {len(rs)} rows, {len(ev)} events, scrambled minus real = {diff(ev):+.2f}pp, "
          f"90% CI [{bs[200]:+.2f}, {bs[3799]:+.2f}]")

print("§3 48-event replication (seven responsive models)")
replicate([r for r in run if not str(r.get("raw")).startswith("__ERROR__")], "API-error rows dropped (paper rule; none here)")
cost = lambda r: (r.get("usage") or {}).get("cost") if isinstance(r.get("usage"), dict) else None
replicate([r for r in run if (r.get("raw") or "").strip() or cost(r)], "sensitivity: one empty zero-cost response also dropped")
