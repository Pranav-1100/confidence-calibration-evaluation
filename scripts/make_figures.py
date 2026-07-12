#!/usr/bin/env python3
"""
Paper figures for the agentic-gradient project. Reads the frozen result JSONs and
writes PNG (300dpi) + PDF (vector, for LaTeX) into research/figures/.

Run with the venv python: .venv-figs/bin/python scripts/make_figures.py

Figures:
  fig1_causal_sandwich  — pooled commitment by condition L0/L1/L2/L2' (the money figure)
  fig2_model_heatmap    — per-model commitment by condition (Seduced/Immune split)
  fig3_earned_check     — accuracy vs baselines + Brier gap with CI per level
  fig4_mitigation       — baseline vs triage (vs placebo if present) + overcorrection panel
  fig5_localization     — triage classification -> action flow (bar version)
"""
import json, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

def load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p): return []
    return [x for x in json.load(open(p)) if not x.get("error") and x.get("decision")]

base = load("agentic_postcutoff.json")
mit = load("agentic_mitigated.json")
plc = load("agentic_placebo.json")
disc_b = load("discrimination_all.json")
disc_m = load("discrimination_mitigated.json")

LVL = {0: "L0\nbare", 1: "L1\nprice", 2: "L2\nfull technicals", 3: "L2′\nirrelevant"}
C = {"commit": "#c0392b", "tool": "#e67e22", "decline": "#2980b9", "mit": "#27ae60", "plc": "#8e44ad"}

def rate(rows, pred):
    return 100 * sum(map(pred, rows)) / len(rows) if rows else float("nan")
is_ans = lambda x: x["decision"] == "ANSWER"
is_tool = lambda x: x["decision"] == "CALL_TOOL"

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

# ---------- fig 1: causal sandwich ----------
fig, ax = plt.subplots(figsize=(6.5, 4))
levels = [0, 1, 2, 3]
ans = [rate([x for x in base if x["level"] == l], is_ans) for l in levels]
tool = [rate([x for x in base if x["level"] == l], is_tool) for l in levels]
xs = range(len(levels))
ax.bar(xs, ans, 0.55, color=C["commit"], label="ANSWER (commit)")
ax.bar(xs, tool, 0.55, bottom=ans, color=C["tool"], alpha=0.65, label="CALL_TOOL")
for i, a in enumerate(ans):
    ax.text(i, a + tool[i] + 2, f"{a:.0f}%", ha="center", fontsize=11, fontweight="bold", color=C["commit"])
ax.set_xticks(xs); ax.set_xticklabels([LVL[l] for l in levels])
ax.set_ylabel("% of decisions (12 models × 40 cases)")
ax.set_title("Non-predictive but relevant-looking evidence induces commitment\non a provably 50/50 question (correct action: DECLINE)")
ax.legend(frameon=False); ax.set_ylim(0, 80); ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig1_causal_sandwich")

# ---------- fig 2: per-model heatmap ----------
models = sorted(set(x["model"] for x in base))
order = sorted(models, key=lambda m: -rate([x for x in base if x["model"] == m and x["level"] == 2], is_ans))
grid = [[rate([x for x in base if x["model"] == m and x["level"] == l], is_ans) for l in levels] for m in order]
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([LVL[l].replace("\n", " ") for l in levels], fontsize=8)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 55 else "black")
ax.set_title("Commitment rate (%) per model and condition", fontsize=11)
fig.colorbar(im, shrink=0.75, label="% ANSWER")
save(fig, "fig2_model_heatmap")

# ---------- fig 3: earned check ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
lv123 = [1, 2]
labels, accs, majs, briers = [], [], [], []
for l in [0, 1, 2]:
    d = [x for x in base if x["level"] == l and is_ans(x) and x["probability"] not in (None, 50)]
    if len(d) < 5: continue
    labels.append(LVL[l].replace("\n", " "))
    accs.append(100 * sum(1 for x in d if (x["probability"] > 50) == x["sealedYes"]) / len(d))
    ups = sum(1 for x in d if x["sealedYes"]); majs.append(100 * max(ups, len(d) - ups) / len(d))
    briers.append(sum((x["probability"] / 100 - (1 if x["sealedYes"] else 0)) ** 2 for x in d) / len(d))
xs = range(len(labels))
a1.bar([x - 0.18 for x in xs], accs, 0.36, color=C["commit"], label="model committed calls")
a1.bar([x + 0.18 for x in xs], majs, 0.36, color="#7f8c8d", label="always-majority (same rows)")
a1.axhline(50, ls="--", c="k", lw=0.8); a1.text(len(labels)-0.55, 51, "chance-50", fontsize=8)
a1.set_xticks(xs); a1.set_xticklabels(labels); a1.set_ylabel("directional accuracy (%)")
a1.set_title("Committed calls beat neither baseline"); a1.legend(frameon=False, fontsize=8)
a1.set_ylim(0, 70); a1.spines[["top", "right"]].set_visible(False)
a2.bar(xs, briers, 0.5, color=C["commit"])
a2.axhline(0.25, ls="--", c="k", lw=0.8); a2.text(-0.4, 0.253, "always-50 (uninformative) = 0.250", fontsize=8)
a2.set_xticks(xs); a2.set_xticklabels(labels); a2.set_ylabel("Brier score (lower = better)")
a2.set_title("Stated probabilities: worse than saying 50%")
a2.set_ylim(0.2, 0.3); a2.spines[["top", "right"]].set_visible(False)
save(fig, "fig3_earned_check")

# ---------- fig 4: mitigation (+placebo if present) ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
conds = [("baseline", base, C["commit"]), ("+ placebo prompt", plc, C["plc"]), ("+ triage prompt", mit, C["mit"])]
conds = [(n, d, c) for n, d, c in conds if d]
vals = [rate([x for x in d if x["level"] == 2], is_ans) for _, d, _ in conds]
a1.bar(range(len(conds)), vals, 0.5, color=[c for _, _, c in conds])
for i, v in enumerate(vals): a1.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
a1.set_xticks(range(len(conds))); a1.set_xticklabels([n for n, _, _ in conds], fontsize=9)
a1.set_ylabel("L2 commitment (%)"); a1.set_ylim(0, 65)
a1.set_title("Epistemic-triage prompt cuts commitment ~79%")
a1.spines[["top", "right"]].set_visible(False)
if disc_b and disc_m:
    types = ["KNOWN", "EPISTEMIC", "ALEATORIC"]; correct = {"KNOWN": "ANSWER", "EPISTEMIC": "CALL_TOOL", "ALEATORIC": "CANNOT_RESOLVE"}
    bvals = [rate([x for x in disc_b if x["type"] == t], lambda x, t=t: x["decision"] == correct[t]) for t in types]
    mvals = [rate([x for x in disc_m if x["type"] == t], lambda x, t=t: x["decision"] == correct[t]) for t in types]
    xs = range(3)
    a2.bar([x - 0.18 for x in xs], bvals, 0.36, color="#7f8c8d", label="baseline")
    a2.bar([x + 0.18 for x in xs], mvals, 0.36, color=C["mit"], label="+ triage")
    a2.set_xticks(xs); a2.set_xticklabels(types, fontsize=8); a2.set_ylim(90, 101)
    a2.set_ylabel("correct-action rate (%)"); a2.set_title("…at zero cost where acting is correct")
    a2.legend(frameon=False, fontsize=8); a2.spines[["top", "right"]].set_visible(False)
save(fig, "fig4_mitigation")

# ---------- fig 5: localization ----------
cat = [x for x in mit if x.get("category") is not None and x["level"] == 2]
if cat:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    n3 = [x for x in cat if x["category"] == 3]; n12 = [x for x in cat if x["category"] in (1, 2)]
    groups = [("classified (3)\nIRREDUCIBLE", n3), ("classified (1)/(2)\n(judgment corrupted)", n12)]
    commit = [100 * sum(1 for x in g if is_ans(x)) / len(g) if g else 0 for _, g in groups]
    share = [100 * len(g) / len(cat) for _, g in groups]
    ax.bar(range(2), commit, 0.45, color=[C["mit"], C["commit"]])
    for i, (c, s) in enumerate(zip(commit, share)):
        ax.text(i, c + 2, f"commit {c:.0f}%\n({s:.0f}% of rows)", ha="center", fontsize=9)
    ax.set_xticks(range(2)); ax.set_xticklabels([g for g, _ in groups], fontsize=9)
    ax.set_ylabel("% committing anyway"); ax.set_ylim(0, 110)
    ax.set_title("Under triage: when the classification is right, the action is right —\nall residual failure is upstream, in the judgment")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig5_localization")

print("done ->", OUT)

# ---------- fig 6: belief vs action divergence ----------
fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
lv = [0, 1, 2]
commit = [rate([x for x in base if x["level"] == l], is_ans) for l in lv]
belief = []
for l in lv:
    pv = [abs(x["probability"] - 50) for x in base if x["level"] == l and is_ans(x) and x["probability"] is not None]
    belief.append(sum(pv) / len(pv) if pv else float("nan"))
ax1.plot(lv, commit, "o-", color=C["commit"], lw=2.5, ms=7, label="commitment rate (action)")
ax1.set_ylabel("commitment: % choosing ANSWER", color=C["commit"])
ax1.set_ylim(0, 60); ax1.tick_params(axis="y", labelcolor=C["commit"])
ax2 = ax1.twinx()
ax2.plot(lv, belief, "s--", color="#2c3e50", lw=2, ms=6, label="stated-belief overconfidence")
ax2.set_ylabel("belief: mean |p − 50| among ANSWERs", color="#2c3e50")
ax2.set_ylim(0, 60); ax2.tick_params(axis="y", labelcolor="#2c3e50")
ax1.set_xticks(lv); ax1.set_xticklabels(["L0 bare", "L1 price", "L2 full technicals"])
ax1.set_title("Beliefs barely move; the action gate swings\n(same scale on both axes)")
l1, lb1 = ax1.get_legend_handles_labels(); l2_, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2_, lb1 + lb2, frameon=False, fontsize=9, loc="upper left")
ax1.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
save(fig, "fig6_belief_vs_action")
