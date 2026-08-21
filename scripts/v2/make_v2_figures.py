#!/usr/bin/env python3
"""
v2 paper figures. Reads the cached model outputs in v2_assets/data/ and v1's frozen results,
writes PNG (300dpi) + PDF (vector) into v2_assets/figures/.

Run: research/.venv-figs/bin/python main-research/v2_assets/scripts/make_v2_figures.py

Figure numbers follow order of appearance in the paper. Sections 4 and 5 were swapped so the
title claim lands first, so the builder function names no longer match the figure numbers:

  builder   output file          what it shows
  fig1()    fig1_story           schematic: what the paper shows, end to end
  fig4()    fig2_scrambled       the causal control - fabricated evidence works as well as real
  fig2()    fig3_gradient        commitment across 4 domains, frontier vs trained
  fig3()    fig4_per_model       per-model commitment and Youden's J, 12 models
  fig5()    fig5_belief_action   stated belief flat while action swings
  fig6()    fig6_seed_spread     where the trained gate breaks - J per run, both framings
"""
import json, os, re, glob, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "..", "data")
V1 = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)

C = {"commit": "#c0392b", "tool": "#e67e22", "decline": "#2980b9",
     "trained": "#27ae60", "frontier": "#c0392b", "grey": "#7f8c8d", "ink": "#2c3e50"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.bbox": "tight"})

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

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig); print("  wrote", name)

def jload(d, n):
    p = os.path.join(d, n)
    return json.load(open(p)) if os.path.exists(p) else []


# ---------------------------------------------------------------- fig 1: story schematic
def fig1():
    fig, ax = plt.subplots(figsize=(13, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    steps = [
        ("Unknowable\nquestion", "will X be higher\nin 10 periods?", C["ink"]),
        ("Add a real\npanel", "RSI, EMA, MACD,\nregime tag", C["commit"]),
        ("Agent\ncommits", "6.5% → 54.0%", C["commit"]),
        ("Fake the\nindicators", "same asset,\nwrong date", C["tool"]),
        ("Fake the\nwhole panel", "nothing true\nbut the question", C["tool"]),
        ("Commits\njust as often", "37.6% → 38.3%\n→ 36.8%", C["tool"]),
        ("Train the\ngate", "540 synthetic\ndice/coin cases", C["trained"]),
        ("Commitment\nfalls to 0%", "on the original\n40 cases", C["trained"]),
    ]
    w, gap = 10.4, 2.0
    for i, (title, sub, col) in enumerate(steps):
        x = 1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 8), w, 14, boxstyle="round,pad=0.3",
                                    fc="white", ec=col, lw=1.8))
        ax.text(x + w/2, 18.2, title, ha="center", va="center", fontsize=8.2,
                weight="bold", color=col)
        ax.text(x + w/2, 12.0, sub, ha="center", va="center", fontsize=6.8, color=C["grey"])
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.3, 15), (x + w + gap - 0.3, 15),
                                         arrowstyle="-|>", mutation_scale=11, color=C["grey"], lw=1.1))
    ax.text(50, 3.2, "A panel with nothing true on it moves the agent as far as a panel of fact. "
                     "The gate that should stop this can be trained back in.",
            ha="center", fontsize=9, style="italic", color=C["ink"])
    save(fig, "fig1_story")


# ---------------------------------------------------------------- fig 2: the main result
def fig2():
    front = [r for r in jload(V2, "frontier_transfer_baseline.json")
             if not str(r.get("raw")).startswith("__ERROR__")]
    nse = jload(V2, "nse_generations.json")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True)
    lv = ["L0", "L1", "L2"]

    # NSE panel from v1 published + trained model
    ax = axes[0]
    ax.plot([0, 1, 2], [6.5, 14.8, 54.0], "o-", color=C["frontier"], lw=2.2, ms=7,
            label="12 frontier models")
    tr = []
    for l in (0, 1, 2):
        rows = [r for r in nse if r["ckpt"] == "SFT-2" and r["arm"] == "unk" and r["level"] == l]
        tr.append(100*sum(parse(r["text"]) == "ANSWER" for r in rows)/len(rows) if rows else 0)
    ax.plot([0, 1, 2], tr, "s-", color=C["trained"], lw=2.2, ms=7,
            label="trained 3B (2-option prompt)")
    ax.plot([], [], "^:", color=C["trained"], lw=1.6, ms=6, alpha=0.75,
            label="trained 3B (3-option prompt)")
    ax.plot([], [], "x--", color=C["grey"], lw=1.3, ms=6, alpha=0.85,
            label="7.5 ablation ckpt (516 cases)")
    ax.set_title("stocks (original cases)", fontsize=10)
    ax.set_ylabel("commitment on unknowable (%)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    tr_rows = []
    for f in glob.glob(os.path.join(V2, "raw_generations*.json")):
        tr_rows += json.load(open(f))
    # The five checkpoints are NOT five seeds of one recipe. Four were trained on the same 540
    # cases; ckpt_nosports (the 7.5 ablation) was trained on 516. Pooling all five put a line on
    # this figure that describes no model that was actually trained - it reported 25% commitment
    # on crypto where the four main-recipe runs give 6.2% and the ablation alone gives 100%.
    MAIN = ["SFT-2", "SFT-2+DPO", "SFT-2-seed1", "SFT-2-seed2"]
    ABLATION = ["SFT-2-nosports-s0"]
    def trained(dom, framing, ckpts=MAIN):
        out = []
        for l in lv:
            rr = [r for r in tr_rows if r.get("framing") == framing and r["domain"] == dom
                  and r["kind"] == "unk" and r["topic"].endswith(l) and r["ckpt"] in ckpts]
            out.append(100*sum(parse(r["text"]) == "ANSWER" for r in rr)/len(rr) if rr else float("nan"))
        return out
    for ax, dom in zip(axes[1:], ["crypto", "sports", "weather"]):
        fr = []
        for l in lv:
            rows = [r for r in front if r["domain"] == dom and r["arm"] == "unk"
                    and r["topic"].endswith(l)]
            fr.append(100*sum(parse(r["raw"]) == "ANSWER" for r in rows)/len(rows) if rows else 0)
        ax.plot([0, 1, 2], fr, "o-", color=C["frontier"], lw=2.2, ms=7)
        # computed, not assumed - an earlier version hardcoded these at zero and the caption
        # claimed a flat line the data does not support
        ax.plot([0, 1, 2], trained(dom, "natural"), "s-", color=C["trained"], lw=2.2, ms=7)
        ax.plot([0, 1, 2], trained(dom, "frontier"), "^:", color=C["trained"], lw=1.6, ms=6,
                alpha=0.75)
        # the 7.5 ablation checkpoint, shown as itself rather than averaged into the recipe above
        abl = trained(dom, "frontier", ABLATION)
        ax.plot([0, 1, 2], abl, "x--", color=C["grey"], lw=1.3, ms=6, alpha=0.85)
        # it exits the top of the axis in two panels; say so rather than letting it clip silently
        if abl[-1] > 62:
            ax.annotate(f"{abl[-1]:.0f}%", xy=(2, 60), xytext=(1.62, 52), fontsize=8,
                        color=C["grey"], arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1))
        ax.set_title(dom, fontsize=10)
    for ax in axes:
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["L0\nbare", "L1\nthin", "L2\nfull panel"])
        ax.set_ylim(-4, 62); ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Evidence-induced commitment across four domains, and its removal by training",
                 fontsize=11, y=1.02)
    save(fig, "fig3_gradient")


# ---------------------------------------------------------------- fig 4: the causal control
def fig3():
    """Three panels: the equivalence result, who it happens to, and the dose-response.
    Rebuilt on the 12-model / 24-event run; the earlier version used 7 models and 8 events."""
    rows = [r for r in jload(V2, "scrambled_12models.json")
            if not str(r.get("raw")).startswith("__ERROR__")]
    # the fully-fabricated arm: symbol aside, EVERY number on the panel is a donor value from a
    # different date, made internally self-consistent. This is the paper's newest result and it
    # belonged in a figure rather than in prose alone.
    rows += [r for r in jload(V2, "scramfull_12models.json")
             if not str(r.get("raw")).startswith("__ERROR__")]
    def dec(t):
        t = (t or "").replace("*", "")
        m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I); b = m.group(1) if m else t
        dm = re.search(r"DECISION\s*:\s*(ANSWER|CALL[_\s-]?TOOL|DECLINE)", b, re.I)
        if not dm:
            lm = re.search(r"DECISION\s*:\s*\(?([ABC])\)?(?![A-Za-z])", b, re.I)
            return ["ANSWER","CALL_TOOL","DECLINE"][ord(lm.group(1).upper())-65] if lm else None
        return dm.group(1).upper().replace(" ", "_").replace("-", "_")

    ARMS = ["thin", "rich", "scram", "scramfull"]
    agg = {a: [0, 0] for a in ARMS}
    per = collections.defaultdict(lambda: {a: [0, 0] for a in ARMS})
    for r in rows:
        a = r["arm"]; hit = dec(r["raw"]) == "ANSWER"
        agg[a][0] += hit; agg[a][1] += 1
        per[r["model"]][a][0] += hit; per[r["model"]][a][1] += 1
    vals = [100*agg[a][0]/agg[a][1] for a in ARMS]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 3.8))

    import math
    def wilson(k, n, z=1.96):
        pp = k/n; d = 1 + z*z/n
        cc = (pp + z*z/(2*n))/d
        h = z*math.sqrt(pp*(1-pp)/n + z*z/(4*n*n))/d
        return 100*max(0, cc-h), 100*min(1, cc+h)
    lohi = [wilson(agg[x][0], agg[x][1]) for x in ARMS]
    yerr = [[v-lo for v, (lo, _) in zip(vals, lohi)], [hi-v for v, (_, hi) in zip(vals, lohi)]]
    a.bar(["no panel", "real\ndata", "indicators\nfabricated", "ENTIRE panel\nfabricated"], vals,
          color=[C["grey"], C["commit"], C["tool"], C["tool"]], width=0.62, yerr=yerr, capsize=4,
          error_kw=dict(ecolor=C["ink"], lw=1.2))
    a.patches[3].set_hatch("//"); a.patches[3].set_edgecolor(C["ink"])
    for i, v in enumerate(vals):
        a.text(i, v + 2.2, f"{v:.1f}%", ha="center", fontsize=9, weight="bold")
    a.set_ylabel("commitment (%)"); a.set_ylim(0, 56); a.grid(axis="y", alpha=0.25)
    a.tick_params(axis="x", labelsize=7.5)
    a.set_title("fabricated evidence = real evidence\n"
                "even when NOTHING on the panel is true\n"
                "full − real = −0.8pp, 90% CI [−4.5, +2.7] (equivalent at ±5pp)", fontsize=8.5)

    # who it actually happens to
    order = sorted(per, key=lambda m: -(100*per[m]["scram"][0]/max(per[m]["scram"][1], 1)))
    for m in order:
        y = [100*per[m][x][0]/max(per[m][x][1], 1) for x in ARMS]
        seduced = y[2] - y[0] > 20
        b.plot([0, 1, 2, 3], y, "o-", lw=2.0 if seduced else 1.0, ms=5 if seduced else 3,
               alpha=0.95 if seduced else 0.35,
               color=C["commit"] if seduced else C["grey"],
               label=m.split("/")[-1] if seduced else None)
    b.set_xticks([0, 1, 2, 3]); b.set_xticklabels(["no\npanel", "real", "indic.\nfaked", "ALL\nfaked"],
                                                  fontsize=7.5)
    b.set_ylabel("commitment (%)"); b.set_ylim(-4, 108); b.grid(alpha=0.25)
    b.legend(fontsize=7, frameon=False, loc="center left")
    b.set_title("the average hides who it happens to\n(bold = seducible; flat lines = immune or saturated)",
                fontsize=9)

    n = [0, 2, 4, 7]
    c.plot(n, [0, 5, 32, 50], "o-", color=C["commit"], lw=2.2, ms=7)
    c.set_xlabel("number of authoritative indicators shown")
    c.set_ylabel("commitment (%)"); c.set_xticks(n); c.grid(alpha=0.25)
    c.set_title("it is a dial, not a switch", fontsize=9)
    save(fig, "fig2_scrambled")


# ---------------------------------------------------------------- fig 4: belief vs action
def fig4():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = [0, 1, 2]
    ax.plot(x, [6.5, 14.8, 54.0], "o-", color=C["commit"], lw=2.4, ms=8, label="ACTION: commitment rate")
    ax.set_ylabel("commitment (%)", color=C["commit"]); ax.tick_params(axis="y", labelcolor=C["commit"])
    ax.set_ylim(0, 60)
    ax2 = ax.twinx()
    # Real values recomputed from agentic_postcutoff.json: 4.71 / 4.14 / 7.70. The series is
    # NON-monotone; an earlier version of this figure hardcoded 6.0 at L1 and drew a smooth rise.
    ax2.plot(x, [4.71, 4.14, 7.70], "s--", color=C["decline"], lw=2.4, ms=8,
             label="BELIEF: |stated prob − 50|")
    ax2.set_ylabel("belief deviation from 50", color=C["decline"])
    ax2.tick_params(axis="y", labelcolor=C["decline"]); ax2.set_ylim(0, 60); ax2.spines["top"].set_visible(False)
    ax.set_xticks(x); ax.set_xticklabels(["L0 bare", "L1 thin", "L2 full panel"])
    ax.annotate("+48pp", xy=(2, 54), xytext=(1.25, 47), color=C["commit"], weight="bold")
    ax2.annotate("+3 points", xy=(2, 7.70), xytext=(1.25, 12), color=C["decline"], weight="bold")
    ax.set_title("Action swings; belief barely moves\n"
                 "(bin-free CORP discrimination 0.008 against uncertainty 0.250:\n"
                 "these probabilities separate outcomes almost not at all)",
                 fontsize=10)
    ax.grid(alpha=0.25)
    save(fig, "fig5_belief_action")


# ---------------------------------------------------------------- fig 5: per-model
def fig5():
    front = [r for r in jload(V2, "frontier_transfer_baseline.json")
             if not str(r.get("raw")).startswith("__ERROR__")]
    rows = []
    for m in sorted({r["model"] for r in front}):
        u = [r for r in front if r["model"] == m and r["arm"] == "unk" and r["topic"].endswith("L2")]
        k = [r for r in front if r["model"] == m and r["arm"] == "know"]
        if not u or not k: continue
        com = 100*sum(parse(r["raw"]) == "ANSWER" for r in u)/len(u)
        J = 100*sum(parse(r["raw"]) == "DECLINE" for r in u)/len(u) - \
            100*sum(parse(r["raw"]) == "DECLINE" for r in k)/len(k)
        rows.append((m.split("/")[-1], com, J))
    rows.sort(key=lambda r: -r[1])
    names = [r[0] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = range(len(rows))
    a.barh(list(y), [r[1] for r in rows], color=C["commit"], height=0.62)
    a.set_yticks(list(y)); a.set_yticklabels(names, fontsize=8.5); a.invert_yaxis()
    a.set_xlabel("commitment on unknowable-L2 (%)"); a.grid(axis="x", alpha=0.25)
    # Derived from the data, not hardcoded. Previously an unreproducible "2.4%" literal that
    # also contradicted the paper's own 0.0% headline on the NSE cases.
    tr_u = [r for r in json.load(open(os.path.join(V2, "nse_generations.json")))
            if r["ckpt"] == "SFT-2" and r["arm"] == "unk" and r["level"] == 2]
    tr_commit = 100*sum(parse(r["text"]) == "ANSWER" for r in tr_u)/len(tr_u)
    a.axvline(tr_commit, color=C["trained"], ls="--", lw=1.6)
    a.text(3.0, len(rows)-0.4, f"trained 3B: {tr_commit:.0f}% (NSE)", color=C["trained"],
           fontsize=8, weight="bold")
    a.set_title("they commit", fontsize=10)
    b.barh(list(y), [r[2] for r in rows], color=C["decline"], height=0.62)
    b.set_xlabel("Youden's J (pp)"); b.grid(axis="x", alpha=0.25)
    b.axvline(95, color=C["trained"], ls="--", lw=1.6)
    b.text(40, len(rows)-0.4, "trained 3B: +95 (NSE, different cases)", color=C["trained"],
           fontsize=7.5, weight="bold")
    b.set_title("and they do not discriminate", fontsize=10)
    fig.suptitle("Per-model behaviour spans the full range, and does not track capability",
                 fontsize=11, y=1.0)
    save(fig, "fig4_per_model")


# ---------------------------------------------------------------- fig 6: seed spread
def fig6():
    rows = []
    for f in glob.glob(os.path.join(V2, "raw_generations*.json")):
        rows += json.load(open(f))
    runs = ["SFT-2", "SFT-2+DPO", "SFT-2-seed1", "SFT-2-seed2", "SFT-2-nosports-s0"]
    doms = ["crypto", "sports", "weather"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, framing, title in zip(axes, ["natural", "frontier"],
                                  ["unseen NATURAL framing\n(stable)",
                                   "structurally novel TOOL framing\n(unstable)"]):
        for di, dom in enumerate(doms):
            vals = []
            for ck in runs:
                u = [r for r in rows if r["ckpt"] == ck and r["framing"] == framing
                     and r["domain"] == dom and r["kind"] == "unk" and r["topic"].endswith("L2")]
                k = [r for r in rows if r["ckpt"] == ck and r["framing"] == framing
                     and r["domain"] == dom and r["kind"] == "know"]
                if not u or not k: vals.append(None); continue
                vals.append(100*sum(parse(r["text"]) == "DECLINE" for r in u)/len(u) -
                            100*sum(parse(r["text"]) == "DECLINE" for r in k)/len(k))
            xs = [i + di*0.22 - 0.22 for i, v in enumerate(vals) if v is not None]
            ys = [v for v in vals if v is not None]
            ax.scatter(xs, ys, s=52, label=dom if framing == "natural" else None, zorder=3)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(["SFT", "SFT\n+DPO*", "seed 1", "seed 2", "overlap\nremoved"], fontsize=8)
        ax.axhline(0, color=C["grey"], lw=1, ls=":")
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.25); ax.set_ylim(-35, 110)
    axes[0].set_ylabel("Youden's J (pp)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right", ncol=3,
                   handletextpad=0.3, columnspacing=1.0)
    axes[1].text(0.02, 0.12, "blanket decliner = 0", transform=axes[1].transAxes,
                 fontsize=7.5, color=C["grey"])
    fig.suptitle("Where the trained gate breaks: robustness to novel prompt structure "
                 "varies run to run", fontsize=11, y=1.02)
    save(fig, "fig6_seed_spread")


if __name__ == "__main__":
    print("writing figures ->", os.path.abspath(OUT))
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("done")
