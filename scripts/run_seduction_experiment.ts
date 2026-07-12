import "dotenv/config";
import * as fs from "fs";
import * as path from "path";

// SEDUCTION experiment — the causal smoking gun for the central thesis:
// "domain context induces an illusion of predictability about irreducibly-random events."
//
// A fair coin (true P = EXACTLY 50%) decides a stock's UP/DOWN. Two conditions:
//   CONTROL   : coin only, no stock evidence -> correct answer is 50.
//   TREATMENT : identical coin, but rich (causally IRRELEVANT) technical evidence shown.
// If a model's confidence moves AWAY from 50 in TREATMENT, the domain evidence
// seduced it into false predictability about a provably-random outcome. Prediction:
// tool-fallacy failers (gemini/gemma) get seduced; humble models (opus/nemotrons) stay 50.
//
// Usage: npx ts-node -r dotenv/config scripts/run_seduction_experiment.ts <cases.json> <out.json> [test|cheap|all] [budgetCap]

interface KCase { id: string; symbol: string; evidenceText: string; }
interface Result { caseId: string; model: string; condition: "control" | "treatment"; probability: number | null; abstained: boolean; reasoning: string; costUsd: number; error?: string; }

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
// Focused roster within a tight budget: the tool-fallacy FAILERS (gemini,gemma,gpt) +
// a PERFECT one (opus) + free nemotrons, to test the predicted split directly.
const CHEAP: M[] = [
  { id: "nvidia/nemotron-3-super-120b-a12b", label: "nemotron-3-super-nv", provider: "nvidia" },
  { id: "google/gemini-3.5-flash", label: "gemini-3.5-flash", provider: "hackclub" },
  { id: "google/gemma-4-31b-it", label: "gemma-4-31b", provider: "hackclub" },
  { id: "openai/gpt-5.4-mini", label: "gpt-5.4-mini", provider: "hackclub" },
  { id: "x-ai/grok-4.20", label: "grok-4.20", provider: "hackclub" },
  { id: "anthropic/claude-haiku-4.5", label: "haiku-4.5", provider: "hackclub" },
];
const ALL: M[] = [
  ...CHEAP,
  { id: "anthropic/claude-opus-4.8", label: "opus", provider: "hackclub" },
  { id: "openai/gpt-5.4", label: "gpt5.4", provider: "hackclub" },
  { id: "deepseek/deepseek-v3.2", label: "deepseek-v3.2", provider: "hackclub" },
];

function prompt(c: KCase, cond: "control" | "treatment"): string {
  const evidence = cond === "treatment"
    ? `\nFor context, here is real recent technical data for ${c.symbol} (note: this data does NOT influence the coin):\n${c.evidenceText}\n`
    : "";
  return `A fair coin (exactly 50% Heads, 50% Tails) will be flipped once. If it lands HEADS we record the result as "UP"; if TAILS, "DOWN".
${evidence}
QUESTION: What is the probability (0-100) that the recorded result is "UP"?

Think in ONE sentence inside <thinking></thinking>. Then inside <answer></answer> respond EXACTLY:
PROBABILITY_YES: <number 0-100, or UNKNOWABLE>
REASONING: <1 sentence>`;
}

function parse(content: string) {
  const ans = content.split(/<\/?answer>/i)[1] ?? content;
  const p = ans.match(/PROBABILITY_YES:\s*([A-Za-z]+|\d+(?:\.\d+)?)/i);
  const r = ans.match(/REASONING:\s*([\s\S]*?)(?:<\/answer>|$)/i);
  const reasoning = (r ? r[1] : content).trim().slice(0, 250);
  if (!p) return { probability: null, abstained: false, reasoning };
  if (/unknowable/i.test(p[1])) return { probability: null, abstained: true, reasoning };
  const n = parseFloat(p[1]); return { probability: Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : null, abstained: false, reasoning };
}

async function call(m: M, pr: string, a = 1): Promise<{ content: string; cost: number }> {
  const cfg = CFG[m.provider]; await slot(m.provider);
  const res = await fetch(cfg.url, { method: "POST", headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: m.id, messages: [{ role: "user", content: pr }], max_tokens: 600, temperature: 0.3 }), signal: AbortSignal.timeout(45000) });
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
  const casesFile = process.argv[2] ?? "knowability_cases_n25.json";
  const outFile = process.argv[3] ?? "seduction_results.json";
  const rosterArg = process.argv[4] ?? "test";
  const cap = parseFloat(process.argv[5] ?? "Infinity");
  const roster = rosterArg === "all" ? ALL : rosterArg === "cheap" ? CHEAP : TEST;
  const allCases: KCase[] = JSON.parse(fs.readFileSync(path.join(__dirname, "..", casesFile), "utf-8"));
  const cases = allCases.slice(0, 12); // keep tight for budget
  const outPath = path.join(__dirname, "..", outFile);

  type Job = { c: KCase; m: M; cond: "control" | "treatment"; pr: string };
  const jobs: Job[] = [];
  for (const c of cases) for (const m of roster) for (const cond of ["control", "treatment"] as const) jobs.push({ c, m, cond, pr: prompt(c, cond) });
  console.log(`Roster ${rosterArg} | ${cases.length} cases x ${roster.length} models x 2 conditions = ${jobs.length} calls.\n`);

  const live: Result[] = []; const flush = () => fs.writeFileSync(outPath, JSON.stringify(live, null, 2));
  let cost = 0, done = 0, capHit = false;
  const run = async (j: Job): Promise<Result> => {
    let r: Result;
    if (cost >= cap) { if (!capHit) { capHit = true; console.log(`\n!! CAP $${cap} hit — skipping rest.`); } done++; r = { caseId: j.c.id, model: j.m.label, condition: j.cond, probability: null, abstained: false, reasoning: "", costUsd: 0, error: "skipped" }; live.push(r); return r; }
    try { const { content, cost: cc } = await call(j.m, j.pr); cost += cc; r = { caseId: j.c.id, model: j.m.label, condition: j.cond, costUsd: cc, ...parse(content) }; }
    catch (e) { r = { caseId: j.c.id, model: j.m.label, condition: j.cond, probability: null, abstained: false, reasoning: "", costUsd: 0, error: (e as Error).message }; }
    done++; live.push(r); if (done % 10 === 0) { console.log(`  ${done}/${jobs.length} | $${cost.toFixed(4)}`); flush(); } return r;
  };
  const byP: Record<string, Job[]> = {}; for (const j of jobs) (byP[j.m.provider] ??= []).push(j);
  await Promise.all(Object.values(byP).map(js => pool(js, 6, run)));
  flush();
  const failed = live.filter(r => r.error); console.log(`\nDone. ${live.length - failed.length}/${live.length} ok. Cost $${cost.toFixed(4)}. -> ${outFile}`);
}
main().catch(e => { console.error(e); process.exit(1); });
