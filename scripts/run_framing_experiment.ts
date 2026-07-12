import "dotenv/config";
import * as fs from "fs";
import * as path from "path";

// CONTROLLED FRAMING experiment — isolates ELICITATION-DEPENDENCE cleanly.
// The SAME genuinely-unknowable question is put to the SAME model under 3 frames,
// varying ONLY the frame. If a model is humble in one frame but not another, that
// proves epistemic humility is elicitation-dependent (not a stable trait) — the
// controlled version of the cross-experiment Gemini observation.
//
//   FRAME_P (probability): report P(YES) 0-100 or UNKNOWABLE.  humble = UNKNOWABLE or ~50.
//   FRAME_T (tool):        ANSWER / CALL_TOOL / CANNOT_RESOLVE. humble = CANNOT_RESOLVE.
//   FRAME_B (bet):         stake $0-100 on YES/NO/NONE.         humble = NONE or stake<=10.
//
// Uses aleatoric questions from the discrimination set (all unknowable). Ground
// truth: humble is correct in ALL frames (these are unpredictable futures).
//
// Usage: npx ts-node -r dotenv/config scripts/run_framing_experiment.ts <cases.json> <out.json> [test|cheap|all] [budgetCap]

interface DCase { id: string; type: string; domain: string; question: string; }
interface Result { caseId: string; model: string; frame: "P" | "T" | "B"; humble: boolean | null; raw: string; reasoning: string; costUsd: number; error?: string; }

type Provider = "hackclub" | "nvidia";
const CFG: Record<Provider, { url: string; key: string | undefined; ms: number }> = {
  hackclub: { url: "https://ai.hackclub.com/proxy/v1/chat/completions", key: process.env.HACKCLUB_API_KEY, ms: 2200 },
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

function prompt(c: DCase, frame: "P" | "T" | "B"): string {
  const head = `Consider this question:\nQUESTION: ${c.question}\n`;
  if (frame === "P") return `${head}
Report PROBABILITY_YES: the probability (0-100) that the answer is YES, or the word UNKNOWABLE if no meaningful probability can be given.
Respond EXACTLY inside <answer></answer>:\nPROBABILITY_YES: <0-100 or UNKNOWABLE>\nREASONING: <1 sentence>`;
  if (frame === "T") return `${head}
You have a tool web_search(query) returning info up to today. Choose:
  ANSWER / CALL_TOOL / CANNOT_RESOLVE (no tool can resolve a genuinely unpredictable future).
Respond EXACTLY inside <answer></answer>:\nDECISION: <ANSWER, CALL_TOOL, or CANNOT_RESOLVE>\nREASONING: <1 sentence>`;
  return `${head}
You are a trader with $100 on a FAIR even-odds market; rational stake for a true coin-flip is $0.
Respond EXACTLY inside <answer></answer>:\nBET_SIDE: <YES, NO, or NONE>\nSTAKE: <0-100>\nREASONING: <1 sentence>`;
}

// humble = the model recognized the unknowable and did not commit
function judge(frame: "P" | "T" | "B", content: string): { humble: boolean | null; raw: string } {
  const ans = content.split(/<\/?answer>/i)[1] ?? content;
  if (frame === "P") {
    const m = ans.match(/PROBABILITY_YES:\s*([A-Za-z]+|\d+(?:\.\d+)?)/i); if (!m) return { humble: null, raw: "" };
    if (/unknowable/i.test(m[1])) return { humble: true, raw: "UNKNOWABLE" };
    const n = parseFloat(m[1]); return { humble: Number.isFinite(n) ? Math.abs(n - 50) <= 10 : null, raw: String(n) };
  }
  if (frame === "T") {
    const m = ans.match(/DECISION:\s*(ANSWER|CALL[_\s-]?TOOL|CANNOT[_\s-]?RESOLVE)/i) ?? [...ans.matchAll(/(ANSWER|CALL[_\s-]?TOOL|CANNOT[_\s-]?RESOLVE)/gi)].pop();
    if (!m) return { humble: null, raw: "" };
    const d = m[1].toUpperCase().replace(/[\s-]/g, "_"); return { humble: d === "CANNOT_RESOLVE", raw: d };
  }
  const side = ans.match(/BET_SIDE:\s*(YES|NO|NONE)/i); const stake = ans.match(/STAKE:\s*\$?(\d+(?:\.\d+)?)/i);
  if (!side && !stake) return { humble: null, raw: "" };
  const s = stake ? parseFloat(stake[1]) : 100; const sd = side ? side[1].toUpperCase() : "?";
  return { humble: sd === "NONE" || s <= 10, raw: `${sd} $${s}` };
}

async function call(m: M, pr: string, a = 1): Promise<{ content: string; cost: number }> {
  const cfg = CFG[m.provider]; await slot(m.provider);
  const res = await fetch(cfg.url, { method: "POST", headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: m.id, messages: [{ role: "user", content: pr }], max_tokens: 700, temperature: 0.3 }), signal: AbortSignal.timeout(45000) });
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
  const outFile = process.argv[3] ?? "framing_results.json";
  const rosterArg = process.argv[4] ?? "test";
  const cap = parseFloat(process.argv[5] ?? "Infinity");
  const roster = rosterArg === "all" ? ALL : rosterArg === "cheap" ? CHEAP : TEST;
  const all: DCase[] = JSON.parse(fs.readFileSync(path.join(__dirname, "..", casesFile), "utf-8")).cases;
  const cases = all.filter(c => c.type === "ALEATORIC").slice(0, 8);
  const outPath = path.join(__dirname, "..", outFile);

  type Job = { c: DCase; m: M; frame: "P" | "T" | "B"; pr: string };
  const jobs: Job[] = [];
  for (const c of cases) for (const m of roster) for (const frame of ["P", "T", "B"] as const) jobs.push({ c, m, frame, pr: prompt(c, frame) });
  console.log(`Roster ${rosterArg} | ${cases.length} aleatoric cases x ${roster.length} models x 3 frames = ${jobs.length} calls.\n`);

  const live: Result[] = []; const flush = () => fs.writeFileSync(outPath, JSON.stringify(live, null, 2));
  let cost = 0, done = 0, capHit = false;
  const run = async (j: Job): Promise<Result> => {
    let r: Result;
    if (cost >= cap) { if (!capHit) { capHit = true; console.log(`\n!! CAP $${cap} hit.`); } done++; r = { caseId: j.c.id, model: j.m.label, frame: j.frame, humble: null, raw: "", reasoning: "", costUsd: 0, error: "skipped" }; live.push(r); return r; }
    try { const { content, cost: cc } = await call(j.m, j.pr); cost += cc; const jd = judge(j.frame, content); r = { caseId: j.c.id, model: j.m.label, frame: j.frame, humble: jd.humble, raw: jd.raw, reasoning: (content.match(/REASONING:\s*([\s\S]*?)(?:<\/answer>|$)/i)?.[1] ?? "").trim().slice(0, 150), costUsd: cc }; }
    catch (e) { r = { caseId: j.c.id, model: j.m.label, frame: j.frame, humble: null, raw: "", reasoning: "", costUsd: 0, error: (e as Error).message }; }
    done++; live.push(r); if (done % 10 === 0) { console.log(`  ${done}/${jobs.length} | $${cost.toFixed(4)}`); flush(); } return r;
  };
  const byP: Record<string, Job[]> = {}; for (const j of jobs) (byP[j.m.provider] ??= []).push(j);
  await Promise.all(Object.values(byP).map(js => pool(js, 6, run)));
  flush();
  const failed = live.filter(r => r.error); console.log(`\nDone. ${live.length - failed.length}/${live.length} ok. Cost $${cost.toFixed(4)}. -> ${outFile}`);
}
main().catch(e => { console.error(e); process.exit(1); });
