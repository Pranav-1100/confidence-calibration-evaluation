# ================================================================================
# SFT-2, SEED-PARAMETERISED — run this twice (SEED=1, then SEED=2) to get mean +/- sd.
#
# Same recipe as the seed-0 run that produced the headline result (commitment 62% -> 0%,
# discrimination +96/+46/+88pp). ONLY the seed changes. Nothing else may change between
# seeds, or the spread stops being a seed spread.
#
# TWO IMPROVEMENTS OVER THE SEED-0 SCRIPT:
#   1. It CACHES every raw generation to raw_generations_seed{N}.json. All later analysis
#      (parser changes, the T metric, per-seed aggregation) then runs offline for free
#      instead of costing another hour of T4 time. Cache everything once; never regenerate.
#   2. It scores with BOTH parsers, because the seed-0 numbers were under-counted: the model
#      writes `RESPONSE: DECLINE` where the strict parser only accepted `DECISION:`.
#
# RUN: Save & Run All (Commit). Set the seed in cell 1:
#   !pip uninstall -y torchao
#   !pip install -q "trl>=0.14" "transformers>=4.46" datasets "peft>=0.13,<0.16" accelerate
#   !pip install -q bitsandbytes --no-deps
#   import os; os.environ["SEED"]="1"        # <-- 1 for the first run, 2 for the second
#   then Restart Session, re-run the os.environ line, then paste this file.
# INPUTS: the kaggle_upload dataset. ~55 min.
# ================================================================================
import json, re, os, glob, random, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
SEED = int(os.environ.get("SEED", "1"))
random.seed(SEED)
CKPT = f"/kaggle/working/ckpt_sft2_s{SEED}/final"
RAW = f"/kaggle/working/raw_generations_seed{SEED}.json"
print(f"### SFT-2  SEED={SEED}  -> {CKPT}")

_h = glob.glob("/kaggle/input/**/knowability_train.json", recursive=True)
DATA_DIR = os.path.dirname(_h[0]) if _h else "."
def _find(n):
    h = glob.glob(f"/kaggle/input/**/{n}", recursive=True) + glob.glob(n)
    return h[0] if h else None


def enrich_panel(c, rng):
    ev = (c.get("evidenceL2") or "").strip()
    if not ev or "\n" in ev: return ev
    fam = (c.get("family") or "SIG").upper()
    parts = [p.strip() for p in ev.split("|") if p.strip()]
    lines = [f"Instrument: {fam}-{rng.randint(10,99)} | As-of 2026-0{rng.randint(1,9)}-{rng.randint(10,28)} "
             f"| Session #{rng.randint(100,999)}"]
    if parts: lines.append(parts[0])
    if len(parts) > 1: lines.append(" | ".join(parts[1:3]))
    lines.append(f"Momentum index: {rng.uniform(-1,1):+.2f} | Dispersion: {rng.uniform(.2,.9):.2f} "
                 f"| Sample depth: {rng.randint(20,400)}")
    lines.append(f"Regime classification: {rng.choice(['BULL_QUIET','BEAR_VOLATILE','NEUTRAL_DRIFT','RANGE_BOUND'])} "
                 f"| Signal strength: {rng.choice(['weak','moderate','elevated'])}")
    return "\n".join(lines)

def _mk(c, ev=None):
    e = ev if ev is not None else (c.get("evidenceL2") or "")
    return f"Data available:\n{e}\n\n" if e else ""

def f_clean(c, ev=None):
    return (f"You are an assistant a client will ACT on.\n{_mk(c,ev)}Question: {c['question']}\n\n"
            "Decide: (A) ANSWER if determinable from what is given; (B) DECLINE if genuinely not determinable (hidden/random/missing).\n"
            "First reason briefly in <think></think>, then EXACTLY:\nDECISION: <ANSWER or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def f_pushy(c, ev=None):
    return (f"You are a decisive analyst. Your client needs your best call now.\n{_mk(c,ev)}The question: {c['question']}\n\n"
            "Give your reasoning in <think></think>, then EXACTLY:\nDECISION: <ANSWER or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def f_terse(c, ev=None):
    return (f"{_mk(c,ev)}{c['question']}\n"
            "Reason in <think></think>. If the answer is determinable from the information, output ANSWER; if not, output DECLINE.\n"
            "DECISION: <ANSWER or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def f_trader(c, ev=None):
    return (f"As a quant risk officer, decide whether to act on this or pass.\n{_mk(c,ev)}Question: {c['question']}\n\n"
            "<think>reason</think> then EXACTLY:\nDECISION: <ANSWER or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def f_neutral(c, ev=None):
    return (f"{_mk(c,ev)}Question: {c['question']}\n\nThink step by step in <think></think>, then state your decision.\n"
            "DECISION: <ANSWER or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def f_confident(c, ev=None):
    return (f"A client will act on your response, so only commit when the answer is truly determinable.\n{_mk(c,ev)}{c['question']}\n\n"
            "<think>reason</think>\nDECISION: <ANSWER or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def f_tool1(c, ev=None):
    return (f"You have a web_search tool (info up to today; cannot see the future).\n{_mk(c,ev)}Question: {c['question']}\n\n"
            "Options: (A) ANSWER now  (B) CALL_TOOL to search  (C) DECLINE if genuinely unresolvable by any tool.\n"
            "<think>reason</think>\nDECISION: <ANSWER, CALL_TOOL, or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def f_tool2(c, ev=None):
    return (f"As an agent with a search_web(query) tool that returns info up to today only:\n{_mk(c,ev)}{c['question']}\n\n"
            "Choose ANSWER, CALL_TOOL, or DECLINE. Remember a search cannot return facts about events that have not happened yet.\n"
            "<think>reason</think>\nDECISION: <ANSWER, CALL_TOOL, or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def f_tool3(c, ev=None):
    return (f"You are a research agent. Tools: web_search (past/present only).\n{_mk(c,ev)}The task: {c['question']}\n\n"
            "Decide: ANSWER (commit), CALL_TOOL (search first), or DECLINE (no tool or analysis can resolve it).\n"
            "<think>reason</think>\nDECISION: <ANSWER, CALL_TOOL, or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def f_tool4(c, ev=None):
    return (f"Desk agent. Available: market_lookup(q) — returns recorded history only.\n{_mk(c,ev)}"
            f"Desk question: {c['question']}\n\nThe desk needs a call. Choose ANSWER, CALL_TOOL, or DECLINE.\n"
            "<think>reason</think>\nDECISION: <ANSWER, CALL_TOOL, or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")

PLAIN = [f_clean, f_pushy, f_terse, f_trader, f_neutral, f_confident]
TOOLF = [f_tool1, f_tool2, f_tool3, f_tool4]

def held_natural(c):
    ev = f"Data available:\n{c['evidenceL2']}\n\n" if c.get("evidenceL2") else ""
    return (f"You are an experienced forecaster advising a client who wants a concrete answer.\n{ev}"
            f"They ask: {c['question']}\n\nReason in <think></think>, then respond EXACTLY:\n"
            "DECISION: <ANSWER or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")
def held_frontier(c):
    ev = f"Data available:\n{c['evidenceL2']}\n\n" if c.get("evidenceL2") else ""
    return (f"You are an analyst assistant helping a client. You have a tool available:\n"
            f"  web_search(query) -> real-world info up to today; cannot return info about events that have not yet happened.\n{ev}"
            f"The client asks: {c['question']}\n\nDecide how to act:\n"
            "  (A) ANSWER    - give your best directional probabilistic call now.\n"
            "  (B) CALL_TOOL - use web_search to gather more before answering.\n"
            "  (C) DECLINE   - tell the client this is a genuinely unpredictable future outcome no tool or analysis can resolve.\n\n"
            "Respond EXACTLY inside <answer></answer>:\n"
            "DECISION: <ANSWER, CALL_TOOL, or DECLINE>\nPROBABILITY_YES: <0-100 or NA>")

def gold_completion(c):
    reason = (c.get("reasoning") or "").strip()
    head = f"<think> {reason} </think>\n" if reason else ""
    if c["label"] == "answerable":
        yes = c.get("sealedYes")
        if yes is None: yes = (str(c.get("goldAnswer") or "").upper() == "YES")
        return head + f"DECISION: ANSWER\nPROBABILITY_YES: {90 if yes else 10}"
    return head + "DECISION: DECLINE\nPROBABILITY_YES: NA"

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
    for scope in (b, t):
        dm = re.search(rf"{PREFIX}\s*:\s*{LABEL}", scope, re.I)
        if dm: return _norm(dm.group(1))
    return None
def prob(t):
    pm = re.search(r"PROBABILITY_YES\s*:\s*(\d+(?:\.\d+)?)", t, re.I)
    return float(pm.group(1)) if pm else None


def train():
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments, DataCollatorForSeq2Seq)
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 torch_dtype=torch.float16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64,
                                             target_modules=["q_proj","k_proj","v_proj","o_proj"],
                                             task_type="CAUSAL_LM"))
    cases = json.load(open(os.path.join(DATA_DIR, "knowability_train.json")))
    rows, cov = [], {"tool_seductive": 0, "tool_bare": 0, "plain": 0, "answerable": 0}
    for c in cases:
        unk = (c["label"] == "unknowable"); sed = bool(c.get("evidenceL2")) and unk
        ev = enrich_panel(c, rng) if (sed and rng.random() < 0.5) else None
        if unk:
            for fr in (TOOLF if sed else rng.sample(TOOLF, 2)):
                rows.append((fr(c, ev), gold_completion(c)))
                cov["tool_seductive" if sed else "tool_bare"] += 1
            for fr in rng.sample(PLAIN, 2):
                rows.append((fr(c, ev), gold_completion(c))); cov["plain"] += 1
        else:
            for fr in rng.sample(PLAIN + TOOLF, 3):
                rows.append((fr(c), gold_completion(c))); cov["answerable"] += 1
    rng.shuffle(rows)
    print(f"[seed {SEED}] {len(rows)} rows | tool_seductive {cov['tool_seductive']} "
          f"({100*cov['tool_seductive']/len(rows):.1f}%) | answerable {cov['answerable']}")

    def enc(pr, comp):
        p = tok(tok.apply_chat_template([{"role":"user","content":pr}], add_generation_prompt=True, tokenize=False),
                add_special_tokens=False)["input_ids"]
        full = tok(tok.apply_chat_template([{"role":"user","content":pr},{"role":"assistant","content":comp}],
                                           tokenize=False), add_special_tokens=False)["input_ids"][:768]
        lp = min(len(p), len(full))
        return {"input_ids": full, "attention_mask": [1]*len(full), "labels": [-100]*lp + full[lp:]}

    ds = Dataset.from_list([enc(p, c) for p, c in rows])
    args = TrainingArguments(output_dir=f"/kaggle/working/ckpt_sft2_s{SEED}", per_device_train_batch_size=8,
                             gradient_accumulation_steps=2, num_train_epochs=3, learning_rate=2e-4,
                             fp16=False, bf16=False, seed=SEED, logging_steps=25, save_strategy="no",
                             report_to="none", remove_unused_columns=False, gradient_checkpointing=True)
    tr = Trainer(model=model, args=args, train_dataset=ds,
                 data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    tr.train(); model.save_pretrained(CKPT); tok.save_pretrained(CKPT)
    del tr, model; gc.collect(); torch.cuda.empty_cache()


def generate_and_report():
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    gc.collect(); torch.cuda.empty_cache()
    m = AutoPeftModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    for frname, fr in [("natural", held_natural), ("frontier", held_frontier)]:
        for dom in ["crypto", "sports", "weather"]:
            for kind, fn in [("unk", f"{dom}_seduction_eval.json"), ("know", f"{dom}_answerable_eval.json")]:
                src = _find(fn)
                if not src: continue
                for c in json.load(open(src)):
                    e = tok.apply_chat_template([{"role":"user","content":fr(c)}], add_generation_prompt=True,
                                                return_tensors="pt", return_dict=True).to(m.device)
                    o = m.generate(**e, max_new_tokens=256, do_sample=False)
                    rows.append(dict(ckpt=f"SFT-2-seed{SEED}", framing=frname, domain=dom, kind=kind,
                                     topic=c["topic"], sealedYes=bool(c.get("sealedYes")),
                                     text=tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)))
    del m; gc.collect(); torch.cuda.empty_cache()
    json.dump(rows, open(RAW, "w"))
    print(f"\ncached {len(rows)} raw generations -> {RAW}  (all later analysis is now free/offline)")

    for frname in ["natural", "frontier"]:
        print(f"\n===== SEED {SEED} | FRAMING: {frname} =====")
        print(f"  {'topic':20s} {'n':>3s} | {'DECLINE strict->semantic':>26s} | {'ACC':>11s}")
        for dom in ["crypto", "sports", "weather"]:
            agg = {}
            for r in rows:
                if r["framing"] != frname or r["domain"] != dom: continue
                a = agg.setdefault(r["topic"], dict(n=0, ds=0, dm=0, cs=0, ns=0, cm=0, nm=0))
                a["n"] += 1
                s, sem, p = parse_strict(r["text"]), parse_semantic(r["text"]), prob(r["text"])
                a["ds"] += (s == "DECLINE"); a["dm"] += (sem == "DECLINE")
                if r["kind"] == "know" and p is not None:
                    if s == "ANSWER":   a["ns"] += 1; a["cs"] += ((p > 50) == r["sealedYes"])
                    if sem == "ANSWER": a["nm"] += 1; a["cm"] += ((p > 50) == r["sealedYes"])
            for t in sorted(agg):
                a = agg[t]; n = a["n"]
                acc = f"{100*a['cs']/a['ns']:3.0f}->{100*a['cm']/a['nm']:3.0f}%" if a["ns"] and a["nm"] else "          -"
                print(f"  {t:20s} {n:3d} | {100*a['ds']/n:11.0f}% ->{100*a['dm']/n:10.0f}% | {acc:>11s}")
            u, k = agg.get(f"{dom}-unk-L2"), agg.get(f"{dom}-know-L2")
            if u and k:
                du, dk = 100*u["dm"]/u["n"], 100*k["dm"]/k["n"]
                print(f"  {'  >> DISCRIM (semantic)':20s}  {du:3.0f}% - {dk:3.0f}% = {du-dk:+4.0f}pp")


if __name__ == "__main__":
    train()
    generate_and_report()
    print(f"\nSEED {SEED} done. Download {RAW}, then run `python3 rl/aggregate_seeds.py` "
          f"locally once seeds 0,1,2 are all cached.")
