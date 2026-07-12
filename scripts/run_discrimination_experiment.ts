import "dotenv/config";
import * as fs from "fs";
import * as path from "path";

// 3-WAY UNCERTAINTY-DISCRIMINATION experiment (improved Leg 2). Adds the control
// our tool leg was missing: an EPISTEMIC-searchable condition (real post-cutoff
// fact) where CALL_TOOL is CORRECT. Now we can tell whether a model that calls
// tools on the unknowable is genuinely CONFUSING aleatoric-for-searchable (the
// tool fallacy) vs just tool-happy. Multi-domain (sports/politics/tech/... not
// just trading). A general web_search tool is offered.
//
//   KNOWN     -> ANSWER          (stable/context fact)
//   EPISTEMIC -> CALL_TOOL       (real event after every model's Jan-2026 cutoff)
//   ALEATORIC -> CANNOT_RESOLVE  (genuinely unpredictable future; no tool helps)
//
// Prompt uses Abstain-R1's "state what is missing" so we can distinguish
// "I'd need to look it up" (epistemic) from "no one can know it" (aleatoric).
//
// Usage: npx ts-node -r dotenv/config scripts/run_discrimination_experiment.ts <cases.json> <out.json> [test|cheap|all] [budgetCap] [nudge]

interface DCase { id: string; type: string; domain: string; question: string; correctDecision: string; }
interface Result { caseId: string; type: string; domain: string; model: string; decision: "ANSWER" | "CALL_TOOL" | "CANNOT_RESOLVE" | null; category: number | null; missing: string; reasoning: string; costUsd: number; error?: string; }

// MITIGATION overcorrection check (pre-registered, PREREGISTRATION.md §Mitigation):
// the SAME epistemic-triage system prompt used in run_agentic_gradient.ts, applied
// to this set where ANSWER/CALL_TOOL are the CORRECT actions for KNOWN/EPISTEMIC
// cases — measures whether the triage prompt causes false-declining. Keep the text
// byte-identical to the agentic runner's TRIAGE_SYS.
const MITIGATION = !!process.env.MITIGATION;
const TRIAGE_SYS = `Before deciding how to act, first classify the question into exactly one category:
  (1) COMPUTABLE   - the answer can be determined from information already provided to you.
  (2) LOOKUPABLE   - the answer exists in the world today and a search could find it.
  (3) IRREDUCIBLE  - no one can reliably know the answer today, no matter what data or tools are used.
Only act on questions of category (1) or (2). Do not commit to an answer on a category (3) question.
Report your classification as an extra line "CATEGORY: <1, 2, or 3>" inside the answer block.`;

type Provider = "hackclub" | "nvidia";
const CFG: Record<Provider, { url: string; key: string | undefined; ms: number }> = {
  hackclub: { url: "https://ai.hackclub.com/proxy/v1/chat/completions", key: process.env.HACKCLUB_API_KEY, ms: 4300 },
  nvidia: { url: "https://integrate.api.nvidia.com/v1/chat/completions", key: process.env.NVIDIA_API_KEY, ms: 1600 },
};
for (const [n, c] of Object.entries(CFG)) if (!c.key) { console.error(`${n.toUpperCase()}_API_KEY missing`); process.exit(1); }
const nextSlot: Record<string, number> = {};
async function slot(p: Provider) { const iv = CFG[p].ms, now = Date.now(); const s = Math.max(now, nextSlot[p] ?? 0); nextSlot[p] = s + iv; if (s - now > 0) await new Promise(r => setTimeout(r, s - now)); }

interface M { id: string; label: string; provider: Provider }
const TEST: M[] = [
  { id: "mistralai/mistral-nemotron", label: "mistral-nemotron-nv", provider: "nvidia" },
  { id: "meta-llama/llama-3.3-70b-instruct", label: "llama3.3-70b-hc", provider: "hackclub" },
];
const CHEAP: M[] = [
  { id: "nvidia/nemotron-3-super-120b-a12b", label: "nemotron-3-super-nv", provider: "nvidia" },
  { id: "x-ai/grok-4.20", label: "grok-4.20", provider: "hackclub" },
  { id: "google/gemini-3.5-flash", label: "gemini-3.5-flash", provider: "hackclub" },
  { id: "google/gemma-4-31b-it", label: "gemma-4-31b", provider: "hackclub" },
  { id: "openai/gpt-5.4-mini", label: "gpt-5.4-mini", provider: "hackclub" },
  { id: "anthropic/claude-haiku-4.5", label: "haiku-4.5", provider: "hackclub" },
];
const ALL: M[] = [
  ...CHEAP,
  { id: "anthropic/claude-opus-4.8", label: "opus", provider: "hackclub" },
  { id: "anthropic/claude-sonnet-5", label: "sonnet", provider: "hackclub" },
  { id: "openai/gpt-5.4", label: "gpt5.4", provider: "hackclub" },
  { id: "x-ai/grok-4.3", label: "grok4.3", provider: "hackclub" },
  { id: "deepseek/deepseek-v3.2", label: "deepseek-v3.2", provider: "hackclub" },
  { id: "qwen/qwen3.7-plus", label: "qwen3.7-plus", provider: "hackclub" },
];

const NUDGE = "\nIt is completely acceptable — and often correct — to answer CANNOT_RESOLVE when no tool could resolve the question, or CALL_TOOL when you would need to look something up. Do not force an answer.";

function prompt(c: DCase, nudge: boolean): string {
  return `Today is 10 July 2026. You must decide how to handle the question below. You have ONE tool:

  web_search(query) -> returns current, real-world information from the internet, up to today (10 July 2026). It CANNOT return information about events that have not yet happened.

QUESTION: ${c.question}
${nudge ? NUDGE : ""}
Choose your next action:
  (A) ANSWER          — you can answer reliably right now from stable knowledge you are confident in.
  (B) CALL_TOOL       — the answer is a real, knowable fact but you are not sure of it / it may be after your training; a web_search would resolve it.
  (C) CANNOT_RESOLVE  — no tool can resolve this, because it depends on a genuinely unpredictable future event.

Think BRIEFLY (1-2 sentences) inside <thinking></thinking>. Then inside <answer></answer> respond EXACTLY:
DECISION: <ANSWER, CALL_TOOL, or CANNOT_RESOLVE>
MISSING: <in <=10 words, what (if anything) is missing: "nothing" / "a current fact to look up" / "the future / unknowable">${MITIGATION ? "\nCATEGORY: <1, 2, or 3>" : ""}
REASONING: <1 sentence>`;
}

function parse(content: string): { decision: Result["decision"]; category: number | null; missing: string; reasoning: string } {
  const ans = content.split(/<\/?answer>/i)[1] ?? content;
  const norm = (s: string) => s.toUpperCase().replace(/[\s-]/g, "_") as Result["decision"];
  const miss = (ans.match(/MISSING:\s*([\s\S]*?)(?:\n|REASONING:|<\/answer>|$)/i)?.[1] ?? "").trim().slice(0, 80);
  const cm = ans.match(/CATEGORY:\s*\(?([123])\)?/i);
  const category = cm ? parseInt(cm[1]) : null;
  const reasoning = (ans.match(/REASONING:\s*([\s\S]*?)(?:<\/answer>|$)/i)?.[1] ?? "").trim().slice(0, 200);
  let m = ans.match(/DECISION:\s*(ANSWER|CALL[_\s-]?TOOL|CANNOT[_\s-]?RESOLVE)/i);
  if (m) return { decision: norm(m[1]), category, missing: miss, reasoning };
  const letter = ans.match(/DECISION:\s*\(?([ABC])\)?/i);
  if (letter) return { decision: (["ANSWER", "CALL_TOOL", "CANNOT_RESOLVE"][letter[1].toUpperCase().charCodeAt(0) - 65] as Result["decision"]), category, missing: miss, reasoning };
  const all = [...ans.matchAll(/(ANSWER|CALL[_\s-]?TOOL|CANNOT[_\s-]?RESOLVE)/gi)];
  if (all.length) return { decision: norm(all[all.length - 1][1]), category, missing: miss, reasoning };
  return { decision: null, category, missing: miss, reasoning };
}

async function call(m: M, pr: string, a = 1): Promise<{ content: string; cost: number }> {
  const cfg = CFG[m.provider]; await slot(m.provider);
  const messages = MITIGATION ? [{ role: "system", content: TRIAGE_SYS }, { role: "user", content: pr }] : [{ role: "user", content: pr }];
  const res = await fetch(cfg.url, { method: "POST", headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: m.id, messages, max_tokens: 900, temperature: 0.3 }), signal: AbortSignal.timeout(45000) });
  if (!res.ok) { if (a < 3) { await new Promise(r => setTimeout(r, 3000 * a)); return call(m, pr, a + 1); } throw new Error(`HTTP ${res.status}`); }
  const d = await res.json() as any; const content = d?.choices?.[0]?.message?.content ?? ""; if (!content) throw new Error("empty");
  return { content, cost: d?.usage?.cost ?? 0 };
}
async function pool<T, R>(items: T[], lim: number, fn: (i: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length); let i = 0;
  async function w() { while (i < items.length) { const j = i++; out[j] = await fn(items[j]); } }
  await Promise.all(Array.from({ length: Math.min(lim, items.length) }, w)); return out;
}

async function main() {
  const casesFile = process.argv[2] ?? "discrimination_cases.json";
  const outFile = process.argv[3] ?? "discrimination_results.json";
  const rosterArg = process.argv[4] ?? "test";
  const cap = parseFloat(process.argv[5] ?? "Infinity");
  const nudge = process.argv[6] === "nudge";
  const roster = rosterArg === "all" ? ALL : rosterArg === "cheap" ? CHEAP : TEST;
  const cases: DCase[] = JSON.parse(fs.readFileSync(path.join(__dirname, "..", casesFile), "utf-8")).cases;
  const outPath = path.join(__dirname, "..", outFile);

  if (MITIGATION && !/mitig/i.test(outFile)) { console.error(`MITIGATION is on but outFile "${outFile}" doesn't contain "mitig" — refusing (resume would silently mix mitigated + baseline rows in one file). Use e.g. discrimination_mitigated.json.`); process.exit(1); }
  type Job = { c: DCase; m: M; pr: string };
  const allJobs: Job[] = [];
  for (const c of cases) for (const m of roster) allJobs.push({ c, m, pr: prompt(c, nudge) });

  // RESUME (same pattern as run_agentic_gradient): keep OK rows from an existing
  // outFile, retry error rows — re-running the SAME command fills only missing cells.
  const key = (caseId: string, model: string) => `${caseId}|${model}`;
  const live: Result[] = []; const doneSet = new Set<string>();
  if (fs.existsSync(outPath)) {
    try {
      for (const x of JSON.parse(fs.readFileSync(outPath, "utf-8")) as Result[])
        if (!x.error && x.decision) { live.push(x); doneSet.add(key(x.caseId, x.model)); }
    } catch { /* corrupt/partial file: start fresh */ }
  }
  const jobs = allJobs.filter(j => !doneSet.has(key(j.c.id, j.m.label)));
  console.log(`Roster ${rosterArg}${nudge ? " +NUDGE" : ""}${MITIGATION ? " | MITIGATION=triage-system-prompt" : ""} | ${cases.length} cases x ${roster.length} models = ${allJobs.length} cells.`);
  console.log(`Resuming: ${doneSet.size} already done, ${jobs.length} remaining.\n`);

  const flush = () => fs.writeFileSync(outPath, JSON.stringify(live, null, 2));
  let cost = 0, done = 0, capHit = false;
  const run = async (j: Job): Promise<Result> => {
    let r: Result;
    if (cost >= cap) { if (!capHit) { capHit = true; console.log(`\n!! CAP $${cap} hit.`); } done++; r = { caseId: j.c.id, type: j.c.type, domain: j.c.domain, model: j.m.label, decision: null, category: null, missing: "", reasoning: "", costUsd: 0, error: "skipped" }; live.push(r); return r; }
    try { const { content, cost: cc } = await call(j.m, j.pr); cost += cc; r = { caseId: j.c.id, type: j.c.type, domain: j.c.domain, model: j.m.label, costUsd: cc, ...parse(content) }; }
    catch (e) { r = { caseId: j.c.id, type: j.c.type, domain: j.c.domain, model: j.m.label, decision: null, category: null, missing: "", reasoning: "", costUsd: 0, error: (e as Error).message }; }
    done++; live.push(r); if (done % 10 === 0) { console.log(`  ${done}/${jobs.length} | $${cost.toFixed(4)}`); flush(); } return r;
  };
  const byP: Record<string, Job[]> = {}; for (const j of jobs) (byP[j.m.provider] ??= []).push(j);
  await Promise.all(Object.values(byP).map(js => pool(js, 6, run)));
  flush();
  const failed = live.filter(r => r.error); console.log(`\nDone. ${live.length - failed.length}/${live.length} ok. Cost $${cost.toFixed(4)}. -> ${outFile}`);
}
main().catch(e => { console.error(e); process.exit(1); });
