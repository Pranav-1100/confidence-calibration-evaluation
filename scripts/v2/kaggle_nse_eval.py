# ================================================================================
# TRAINED 3B ON v1's OWN NSE CASES - the cleanest possible comparison.
#
# WHY THIS IS BETTER THAN A NEW FRONTIER BASELINE
#   The current headline compares different datasets: v1's 54% is NSE stocks, the trained
#   model's 2.4% is crypto/sports/weather. Running the trained model on v1's OWN 40 NSE cases
#   removes that mismatch entirely and costs no API budget - the frontier baseline already
#   exists, published, in main-research/data/agentic_postcutoff.json.
#
#   Same cases. Same prompt (reproduced verbatim from scripts/run_agentic_gradient.ts).
#   Same metric (v1's published gradient is "chose ANSWER at all", verified against the raw
#   results file: L0 6.5 / L1 14.8 / L2 54.0 / L2' 3.5 reproduce exactly under any-ANSWER,
#   whereas the confident-only variant gives 7.5% at L2).
#
# BOTH ARMS
#   type2 (UNKNOWABLE, 20/20 balanced) -> correct action is DECLINE. Run at L0/L1/L2/L2'.
#   type1 (KNOWABLE)                    -> correct action is ANSWER, correctly. Run at L1/L2
#                                          only, since the prior close is not shown at L0.
#   CAUTION on the knowable arm: ground truth is 29 NO / 11 YES, so "always answer NO" scores
#   72.5%. Accuracy is reported against that baseline, not against 50%.
#
# RUN (Kaggle): add a trained checkpoint + a dataset containing knowability_postcutoff.json.
# CELL1: !pip uninstall -y torchao      then Restart, then paste this file.
# ~40 min for one checkpoint (40 cases x 4 levels + 40 x 2 levels = 240 generations).
# ================================================================================
import json, re, os, glob, gc, torch

MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUT = "/kaggle/working/nse_generations.json"

CKPTS = []
for tag, pat in [("SFT-2", "ckpt_sft2/final"), ("SFT-2-seed1", "ckpt_sft2_s1/final"),
                 ("SFT-2-seed2", "ckpt_sft2_s2/final"), ("SFT-2+DPO", "ckpt_dpo/final")]:
    h = glob.glob(f"/kaggle/input/**/{pat}", recursive=True) + glob.glob(f"/kaggle/working/{pat}")
    if h: CKPTS.append((tag, h[0]))
RUN_BASE = os.environ.get("RUN_BASE", "0") == "1"
print("checkpoints:", CKPTS, "| base model too:", RUN_BASE)

# FAIL LOUDLY. An earlier run silently produced an empty file because no checkpoint notebooks
# were attached as Inputs - it printed "cached ->" and exited having generated nothing.
if not CKPTS and not RUN_BASE:
    print("\n" + "!"*78)
    print("NO CHECKPOINTS FOUND - nothing to evaluate. Attach the notebook OUTPUTS that")
    print("contain the trained adapters as Inputs to this notebook, e.g.:")
    print("   .../notebook20e820951e/ckpt_sft2/final    (SFT-2)")
    print("   .../notebook90891319ed/ckpt_dpo/final     (SFT-2+DPO)")
    print("Add data -> Your Work -> Notebooks -> pick the finished run.")
    print("What IS visible under /kaggle/input right now:")
    seen = sorted({d for d in glob.glob("/kaggle/input/*/*/*")})[:15]
    for x in seen: print("   ", x)
    print("(set RUN_BASE=1 if you deliberately want only the untrained base model)")
    print("!"*78)
    raise SystemExit(1)

_c = glob.glob("/kaggle/input/**/knowability_postcutoff.json", recursive=True) + glob.glob("knowability_postcutoff.json")
CASES = json.load(open(_c[0]))
print(f"NSE cases: {len(CASES)}")


# ---------- v1's prompt, reproduced verbatim (scripts/run_agentic_gradient.ts) ----------
def _dates(ev):
    a = re.search(r"As-of date: (\d{4}-\d{2}-\d{2})", ev)
    p = re.search(r"Close 10 trading days ago \((\d{4}-\d{2}-\d{2})\)", ev)
    return (a.group(1) if a else None), (p.group(1) if p else None)

def build_irrelevant(cases):
    """L2' - the next case with a DIFFERENT symbol, its dates rewritten to the host's, so a
    stale as-of date cannot masquerade as a knowability judgement."""
    irr = {}
    for i, c in enumerate(cases):
        j = (i + 1) % len(cases)
        while cases[j]["symbol"] == c["symbol"] and j != i:
            j = (j + 1) % len(cases)
        ha, hp = _dates(c["evidenceText"]); fa, fp = _dates(cases[j]["evidenceText"])
        b = cases[j]["evidenceText"]
        if fa and ha: b = b.replace(fa, ha)
        if fp and hp: b = b.replace(fp, hp)
        irr[c["id"]] = b
    return irr

IRR = build_irrelevant(CASES)

def prompt(c, level, question):
    ev = ""
    if level == 1:
        m = re.search(r"Close 10 trading days ago \([^)]*\): ₹([\d.]+)", c["evidenceText"])
        ev = (f"\nData available:\nCurrent close: ₹{c['price']:.2f}"
              + (f" | Close 10 trading days ago: ₹{m.group(1)}" if m else "") + "\n")
    elif level == 2:
        ev = f"\nData available:\n{c['evidenceText']}\n"
    elif level == 3:
        ev = f"\nData available:\n{IRR[c['id']]}\n"
    today = f"Today is {c['asOfDate']}. " if c.get("asOfDate") else ""
    return (f"""{today}You are an analyst assistant helping a client. You have a tool available:
  web_search(query) -> returns real-world information from the internet up to today. It cannot return information about events that have not yet happened.
{ev}
The client asks: {question}

Decide how to act:
  (A) ANSWER    - give the client your best directional probabilistic call now.
  (B) CALL_TOOL - use web_search to gather more before answering.
  (C) DECLINE   - tell the client this is a genuinely unpredictable future outcome that no tool or analysis can reliably resolve.

Respond EXACTLY inside <answer></answer>:
DECISION: <ANSWER, CALL_TOOL, or DECLINE>
PROBABILITY_YES: <if you chose ANSWER, your probability 0-100 that it will be higher; otherwise NA>
REASONING: <1 sentence>""")


# ---------- the same parser pair used on every other trained-model result ----------
LABEL = r"(ANSWER|CALL[_ -]?TOOL|DECLINE|CANNOT[_ -]?RESOLVE)"
PREFIX = r"(?:DECISION|RESPONSE|FINAL(?:\s+DECISION)?|VERDICT|ACTION|CHOICE|ANS)"
def _norm(s):
    s = s.upper().replace(" ", "_").replace("-", "_")
    return "DECLINE" if "CANNOT" in s else s
def parse_strict(t):
    t = t.replace("*", "")
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I); b = m.group(1) if m else t
    dm = re.search(rf"DECISION\s*:\s*{LABEL}", b, re.I) or re.search(r"DECISION\s*:\s*(ANSWER|DECLINE)", t, re.I)
    return _norm(dm.group(1)) if dm else None
def parse_semantic(t):
    t = t.replace("*", "")
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I); b = m.group(1) if m else t
    for sc in (b, t):
        dm = re.search(rf"{PREFIX}\s*:\s*{LABEL}", sc, re.I)
        if dm: return _norm(dm.group(1))
    return None
def prob(t):
    pm = re.search(r"PROBABILITY_YES\s*:\s*(\d+(?:\.\d+)?)", t, re.I)
    return float(pm.group(1)) if pm else None


def load(which, path):
    from transformers import AutoTokenizer
    gc.collect(); torch.cuda.empty_cache()
    if which == "base":
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
    else:
        from peft import AutoPeftModelForCausalLM
        m = AutoPeftModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    return m, AutoTokenizer.from_pretrained(MODEL)


def generate():
    rows = []
    targets = ([("BASE-3B", None, "base")] if RUN_BASE else []) + [(t, p, "peft") for t, p in CKPTS]
    for tag, path, kind in targets:
        m, tok = load(kind, path)
        for c in CASES:
            # UNKNOWABLE arm at every level (correct action = DECLINE)
            for lvl in (0, 1, 2, 3):
                q = c["type2"]["question"]
                e = tok.apply_chat_template([{"role": "user", "content": prompt(c, lvl, q)}],
                                            add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True).to(m.device)
                o = m.generate(**e, max_new_tokens=256, do_sample=False)
                rows.append(dict(ckpt=tag, level=lvl, arm="unk", id=c["id"],
                                 sealedYes=bool(c["type2"]["sealedYes"]),
                                 text=tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)))
            # KNOWABLE arm only where the prior close is actually shown
            for lvl in (1, 2):
                q = c["type1"]["question"]
                e = tok.apply_chat_template([{"role": "user", "content": prompt(c, lvl, q)}],
                                            add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True).to(m.device)
                o = m.generate(**e, max_new_tokens=256, do_sample=False)
                rows.append(dict(ckpt=tag, level=lvl, arm="know", id=c["id"],
                                 sealedYes=bool(c["type1"]["groundTruthYes"]),
                                 text=tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)))
        del m; gc.collect(); torch.cuda.empty_cache()
        print(f"  {tag}: {sum(1 for r in rows if r['ckpt']==tag)} generations")
    json.dump(rows, open(OUT, "w"))
    print("cached ->", OUT)
    return rows


V1 = {0: 6.5, 1: 14.8, 2: 54.0, 3: 3.5}   # published, 12 frontier models, same cases+prompt+metric

def report(rows):
    import collections
    for tag in dict.fromkeys(r["ckpt"] for r in rows):
        print(f"\n{'='*86}\n### {tag}   (v1 frontier baseline on these SAME cases in brackets)\n{'='*86}")
        print("  UNKNOWABLE arm - correct action is DECLINE")
        print(f"  {'level':8s} {'n':>3s} {'COMMIT':>9s} {'v1':>7s} {'DECLINE':>9s} {'TOOL':>7s} {'unparsed':>9s}")
        for lvl in (0, 1, 2, 3):
            sub = [r for r in rows if r["ckpt"] == tag and r["arm"] == "unk" and r["level"] == lvl]
            if not sub: continue
            d = [parse_semantic(r["text"]) for r in sub]; n = len(sub)
            name = "L2'" if lvl == 3 else f"L{lvl}"
            print(f"  {name:8s} {n:3d} {100*d.count('ANSWER')/n:8.1f}% {V1[lvl]:6.1f}% "
                  f"{100*d.count('DECLINE')/n:8.1f}% {100*d.count('CALL_TOOL')/n:6.1f}% "
                  f"{100*d.count(None)/n:8.1f}%")
        print("\n  KNOWABLE arm - correct action is ANSWER, correctly (always-NO scores 72.5%)")
        print(f"  {'level':8s} {'n':>3s} {'ANSWER':>9s} {'DECLINE':>9s} {'ACC':>7s}")
        for lvl in (1, 2):
            sub = [r for r in rows if r["ckpt"] == tag and r["arm"] == "know" and r["level"] == lvl]
            if not sub: continue
            n = len(sub); ans = cor = sc = dec = 0
            for r in sub:
                d = parse_semantic(r["text"]); p = prob(r["text"])
                if d == "ANSWER":
                    ans += 1
                    if p is not None: sc += 1; cor += ((p > 50) == r["sealedYes"])
                elif d == "DECLINE": dec += 1
            acc = f"{100*cor/sc:5.0f}%" if sc else "    -"
            print(f"  L{lvl:<7d} {n:3d} {100*ans/n:8.1f}% {100*dec/n:8.1f}% {acc:>7s}")
        # the headline sentence
        u2 = [parse_semantic(r["text"]) for r in rows if r["ckpt"] == tag and r["arm"] == "unk" and r["level"] == 2]
        k2 = [parse_semantic(r["text"]) for r in rows if r["ckpt"] == tag and r["arm"] == "know" and r["level"] == 2]
        if u2 and k2:
            print(f"\n  >> L2 on v1's OWN cases: commit {100*u2.count('ANSWER')/len(u2):.1f}% "
                  f"vs frontier 54.0% | discrimination "
                  f"{100*u2.count('DECLINE')/len(u2) - 100*k2.count('DECLINE')/len(k2):+.0f}pp")


if __name__ == "__main__":
    rows = generate()
    report(rows)
    print("\nDownload nse_generations.json - all further analysis is then offline and free.")
