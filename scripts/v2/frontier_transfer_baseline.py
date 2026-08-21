#!/usr/bin/env python3
# ================================================================================
# THE MISSING BASELINE: frontier models on OUR transfer domains, under the FRONTIER prompt.
#
# WHY THIS RUN EXISTS
#   The headline currently compares different datasets:
#     v1 54%   = NSE stocks,  frontier natural prompt
#     3B 2.4%  = crypto/sports/weather, frontier natural prompt
#   frontier_matched_hc.json does cover the same crypto cases, but through OUR CLEAN prompt
#   (which legitimises DECLINE) and scores 4.9% - so it cannot stand in for the frontier
#   natural-prompt number. No frontier data exists for sports or weather at all.
#   This run fills the exact missing cell: same cases, same prompt, same parser, same metric.
#
#   It also runs the matched ANSWERABLE controls, which no frontier baseline has ever covered.
#   That yields the direct comparator to the trained model's +86..+100pp discrimination:
#   do frontier models answer the answerable ones correctly, or do they fail there too?
#
# METRIC MATCHING (verified against main-research/data/agentic_postcutoff.json)
#   v1's published gradient (6.5 / 14.8 / 54.0 / 3.5) is "chose ANSWER at all", NOT
#   "answered with |p-50| >= 15" (that variant is only 7.5% at L2). The trained-3B numbers use
#   the same any-ANSWER definition, so commit here is any-ANSWER. Both are reported anyway.
#
# USAGE
#   1) SMOKE (free, ~5 min, no spend):   python3 rl/frontier_transfer_baseline.py --smoke
#   2) FULL  (HackClub, paid):           python3 rl/frontier_transfer_baseline.py --full
#      Optional narrowing:  --domains crypto,sports  --levels L2  --arms unk,know
#   Incremental + resumable: every result is appended to the output file as it arrives, so a
#   killed run loses nothing and re-running skips what is already done. (Two long runs were
#   killed mid-flight earlier in this project; that is why this is not batched at the end.)
# ================================================================================
import os, re, json, time, argparse, urllib.request, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "frontier_transfer_baseline.json")

def _key(name):
    p = os.path.join(ROOT, ".env")
    if not os.path.exists(p): return None
    for line in open(p):
        if line.startswith(name + "="): return line.strip().split("=", 1)[1]
    return None

URL = "https://ai.hackclub.com/proxy/v1/chat/completions"

# the paper's frontier roster, as served by HackClub
# ORDER MATTERS under a budget cap: finish the partially-done model first (so no spend is
# stranded mid-model), then the likely-cheap ones, and leave the two Grok reasoning models
# LAST. Gemini-3.5-flash measured 21x costlier per call than gpt-5.4-mini purely because it
# emits ~4x the tokens, so "reasoning model" is the thing that predicts cost, not the name.
FULL_MODELS = [
    "google/gemini-3.5-flash",            # partially done - finish it first
    "openai/gpt-5.4-mini",                # nearly done
    "google/gemma-4-31b-it", "deepseek/deepseek-v3.2", "qwen/qwen3.7-plus",
    "meta-llama/llama-3.3-70b-instruct",  # likely cheap, non-reasoning
    "x-ai/grok-4.3", "x-ai/grok-4.20",    # likely expensive - do these last / tomorrow
    "anthropic/claude-opus-4.8", "anthropic/claude-sonnet-5", "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4",                     # already complete; listed so resume logic skips them
]
# Free-tier ids for the smoke pass, so no budget is spent before the harness and parser are
# shown to work end to end (free-before-paid). VERIFIED present in GET /v1/models on
# 2026-08-15; the previously hard-coded ids 404'd because the proxy's free roster had moved.
# If these start 404ing, re-list the endpoint rather than guessing new names.
# Note: several ids the /v1/models endpoint lists still 404 on the chat endpoint
# (google/gemma-4-*:free among them), so these two were confirmed with a live call.
SMOKE_MODELS = ["nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-nano-30b-a3b:free"]

GAP = 4.5  # seconds between calls; HackClub allows ~450/30min


# ---- the EXACT prompt the trained 3B was evaluated under (held_frontier) ----
def _ev(c):
    return f"Data available:\n{c['evidenceL2']}\n\n" if c.get("evidenceL2") else ""

def build_prompt(c):
    return (f"You are an analyst assistant helping a client. You have a tool available:\n"
            f"  web_search(query) -> real-world info up to today; cannot return info about events that have not yet happened.\n"
            f"{_ev(c)}The client asks: {c['question']}\n\nDecide how to act:\n"
            "  (A) ANSWER    - give your best directional probabilistic call now.\n"
            "  (B) CALL_TOOL - use web_search to gather more before answering.\n"
            "  (C) DECLINE   - tell the client this is a genuinely unpredictable future outcome no tool or analysis can resolve.\n\n"
            "Respond EXACTLY inside <answer></answer>:\n"
            "DECISION: <ANSWER, CALL_TOOL, or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")


# ---- the same parser pair used on the trained model ----
LABEL = r"(ANSWER|CALL[_ -]?TOOL|DECLINE|CANNOT[_ -]?RESOLVE)"
PREFIX = r"(?:DECISION|RESPONSE|FINAL(?:\s+DECISION)?|VERDICT|ACTION|CHOICE|ANS)"

def _norm(s):
    s = s.upper().replace(" ", "_").replace("-", "_")
    return "DECLINE" if "CANNOT" in s else s

def parse(t):
    t = (t or "").replace("*", "")
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I); b = m.group(1) if m else t
    for scope in (b, t):
        dm = re.search(rf"{PREFIX}\s*:\s*{LABEL}", scope, re.I)
        if dm: return _norm(dm.group(1))
    return None

def prob(t):
    pm = re.search(r"PROBABILITY_YES\s*:\s*(\d+(?:\.\d+)?)", t or "", re.I)
    return float(pm.group(1)) if pm else None


def call(model, prompt, key, retries=3):
    """-> (content, usage_dict). Usage is captured so spend can be measured rather than
    guessed; the proxy returns it on most models, and {} where it does not."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1000, "temperature": 0.3}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                j = json.load(r)
                return j["choices"][0]["message"]["content"], (j.get("usage") or {})
        except Exception as e:
            if a == retries - 1: return f"__ERROR__ {type(e).__name__}: {e}", {}
            time.sleep(3 * (a + 1))


def load_cases(domains, levels, arms):
    cases = []
    for dom in domains:
        if "unk" in arms:
            p = os.path.join(ROOT, f"{dom}_seduction_eval.json")
            if os.path.exists(p):
                for c in json.load(open(p)):
                    if any(c["topic"].endswith(l) for l in levels):
                        cases.append(dict(c, _dom=dom, _arm="unk"))
        if "know" in arms:
            p = os.path.join(ROOT, f"{dom}_answerable_eval.json")
            if os.path.exists(p):
                for c in json.load(open(p)):
                    cases.append(dict(c, _dom=dom, _arm="know"))
    return cases


def report(res):
    print(f"\n{'='*92}\nFRONTIER BASELINE - transfer domains, frontier prompt")
    print("v1 published, NSE stocks, same prompt & metric: L0 6.5% -> L1 14.8% -> L2 54.0%")
    print("=" * 92)
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in res:
        d = parse(r["raw"])
        k = (r["domain"], r["topic"])
        agg[k]["n"] += 1
        agg[k][d or "unparsed"] += 1
        if d == "ANSWER":
            p = prob(r["raw"])
            if p is not None and abs(p - 50) >= 15: agg[k]["confident"] += 1
            if r["arm"] == "know" and p is not None:
                agg[k]["scored"] += 1
                agg[k]["correct"] += int((p > 50) == bool(r.get("sealedYes")))
    print(f"  {'topic':22s} {'n':>4s} {'COMMIT':>8s} {'confident':>10s} {'DECLINE':>9s} {'TOOL':>7s} {'ACC':>7s}")
    for k in sorted(agg):
        a = agg[k]; n = a["n"]
        acc = f"{100*a['correct']/a['scored']:5.0f}%" if a["scored"] else "     -"
        print(f"  {k[1]:22s} {n:4d} {100*a['ANSWER']/n:7.1f}% {100*a['confident']/n:9.1f}% "
              f"{100*a['DECLINE']/n:8.1f}% {100*a['CALL_TOOL']/n:6.1f}% {acc:>7s}")
    # the headline cell
    for dom in sorted({k[0] for k in agg}):
        u = agg.get((dom, f"{dom}-unk-L2")); kk = agg.get((dom, f"{dom}-know-L2"))
        if u and kk and u["n"] and kk["n"]:
            du, dk = 100*u["DECLINE"]/u["n"], 100*kk["DECLINE"]/kk["n"]
            print(f"  >> {dom}: frontier commit(unk-L2) {100*u['ANSWER']/u['n']:.1f}% | "
                  f"discrimination {du-dk:+.0f}pp   [trained 3B: commit 2.4%, discrim +86..+100pp]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="free models only, tiny slice, no spend")
    ap.add_argument("--full", action="store_true", help="full paid roster")
    ap.add_argument("--domains", default="crypto,sports,weather")
    ap.add_argument("--levels", default="L0,L1,L2")
    ap.add_argument("--arms", default="unk,know")
    ap.add_argument("--limit", type=int, default=0, help="cap cases per domain (smoke uses 3)")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="hard stop once measured spend for the day reaches this many USD")
    a = ap.parse_args()

    key = _key("HACKCLUB_API_KEY")
    assert key, "no HACKCLUB_API_KEY in RL_env/.env"

    models = SMOKE_MODELS if a.smoke else FULL_MODELS
    levels = a.levels.split(",")
    if a.smoke:
        levels = ["L2"]; a.limit = a.limit or 3
    cases = load_cases(a.domains.split(","), levels, a.arms.split(","))
    if a.limit:
        keep, seen = [], collections.Counter()
        for c in cases:
            k = (c["_dom"], c["_arm"])
            if seen[k] < a.limit: keep.append(c); seen[k] += 1
        cases = keep

    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["model"], r["id"]) for r in res}
    todo = [(m, c) for m in models for c in cases if (m, c["id"]) not in done]
    print(f"{'SMOKE (free)' if a.smoke else 'FULL (paid)'}: {len(models)} models x {len(cases)} cases "
          f"= {len(models)*len(cases)} | already done {len(done)} | to run {len(todo)}")
    if a.smoke:
        print("free-before-paid: verifying harness + parser end to end before any spend.\n")

    # measured spend already on file for today (rows carrying a cost field)
    spent = sum((r.get("usage") or {}).get("cost") or 0.0 for r in res)
    if a.budget:
        print(f"budget cap ${a.budget:.2f} | already spent ${spent:.4f} "
              f"| ${a.budget-spent:.2f} available this session\n")

    t0 = time.time()
    for i, (m, c) in enumerate(todo, 1):
        if a.budget and spent >= a.budget:
            print(f"\n*** BUDGET CAP REACHED: ${spent:.4f} >= ${a.budget:.2f} - stopping cleanly.")
            print(f"    {len(todo)-i+1} calls not run. Resume tomorrow with the same command.")
            break
        raw, usage = call(m, build_prompt(c), key)
        spent += (usage or {}).get("cost") or 0.0
        res.append(dict(model=m, id=c["id"], domain=c["_dom"], arm=c["_arm"], topic=c["topic"],
                        sealedYes=bool(c.get("sealedYes")), raw=raw, usage=usage))
        json.dump(res, open(OUT, "w"))          # incremental: a killed run loses nothing
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  ({el/60:.1f} min) | spent ${spent:.3f}"
                  + (f" of ${a.budget:.2f}" if a.budget else ""))
        time.sleep(GAP)

    errs = sum(1 for r in res if str(r["raw"]).startswith("__ERROR__"))
    unp = sum(1 for r in res if parse(r["raw"]) is None)
    print(f"\nwrote {OUT}: {len(res)} results | errors {errs} | unparsed {unp}")
    tok = collections.Counter()
    for r in res:
        u = r.get("usage") or {}
        tok[r["model"]] += (u.get("total_tokens") or 0)
    if sum(tok.values()):
        print("\n  tokens used this session (0 = proxy did not report usage for that model):")
        for m, t in tok.most_common():
            if t: print(f"    {m:38s} {t:>9,} tokens")
    if a.smoke:
        print("\nSMOKE CHECK: if errors and unparsed are both low, the harness is good.")
        print("Then run:  python3 rl/frontier_transfer_baseline.py --full")
    report(res)


if __name__ == "__main__":
    main()
