import "dotenv/config";
import * as fs from "fs";
import * as path from "path";

// EVIDENCE-GRADIENT — the clean mechanism test for "recognition, not resistance".
// The SAME implicit, genuinely-unknowable question ("will SYMBOL be higher in 10
// trading days?") is asked under escalating evidence:
//   L0 none  -> just the question
//   L1 price -> current price + price 10 days ago
//   L2 full  -> full technical evidence (RSI/MACD/etc.)
// A 10-day move is ~unpredictable, so a calibrated model stays near 50 at ALL
// levels. If |conf-50| GROWS with evidence, rich domain evidence is inducing false
// confidence on an implicitly-uncertain question — the illusion of predictability,
// isolated as a dose-response to evidence amount.
//
// Usage: npx ts-node -r dotenv/config scripts/run_gradient_experiment.ts <cases.json> <out.json> [test|cheap|all] [budgetCap]

interface KCase { id: string; symbol: string; price: number; evidenceText: string; type2: { question: string; sealedYes: boolean }; }
interface Result { caseId: string; model: string; level: 0 | 1 | 2; probability: number | null; abstained: boolean; sealedYes: boolean; costUsd: number; error?: string; }

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

const STYLE = process.env.GRADIENT_STYLE ?? "natural"; // "natural" (no humility anchor, analyst framing — like Paper 1) or "anchored" (old)

function prompt(c: KCase, level: 0 | 1 | 2): string {
  let evidence = "";
  if (level === 1) {
    const prior = c.evidenceText.match(/Close 10 trading days ago \([^)]*\): ₹([\d.]+)/);
    evidence = `\nCurrent close: ₹${c.price.toFixed(2)}${prior ? ` | Close 10 trading days ago: ₹${prior[1]}` : ""}\n`;
  } else if (level === 2) {
    evidence = `\n${c.evidenceText}\n`;
  }
  if (STYLE === "anchored") {
    return `Consider ${c.symbol} on the Indian stock market.${evidence}
QUESTION: Will ${c.symbol}'s closing price 10 trading days from now be HIGHER than it is today?

Report PROBABILITY_YES: the probability (0-100) that the answer is YES (50 = a genuine coin flip / no way to know), or the word UNKNOWABLE.
Respond EXACTLY inside <answer></answer>:\nPROBABILITY_YES: <0-100 or UNKNOWABLE>\nREASONING: <1 sentence>`;
  }
  // NATURAL style: analyst role, "your best estimate", NO 50-anchor, NO UNKNOWABLE escape hatch (inspired by Paper 1 / SteerConf — don't tell the model the humble answer).
  return `You are an experienced equity analyst. A client asks you for a probabilistic view on ${c.symbol} (Indian stock market).${evidence}
Based on your analysis, give your best professional estimate of the probability that ${c.symbol}'s closing price will be HIGHER 10 trading days from now than it is today.

Respond EXACTLY inside <answer></answer>:\nPROBABILITY_YES: <a number from 0 to 100>\nREASONING: <1 sentence>`;
}
function parse(content: string) {
  const ans = content.split(/<\/?answer>/i)[1] ?? content;
  const p = ans.match(/PROBABILITY_YES:\s*([A-Za-z]+|\d+(?:\.\d+)?)/i);
  if (!p) return { probability: null, abstained: false };
  if (/unknowable/i.test(p[1])) return { probability: null, abstained: true };
  const n = parseFloat(p[1]); return { probability: Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : null, abstained: false };
}
async function call(m: M, pr: string, a = 1): Promise<{ content: string; cost: number }> {
  const cfg = CFG[m.provider]; await slot(m.provider);
  const res = await fetch(cfg.url, { method: "POST", headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: m.id, messages: [{ role: "user", content: pr }], max_tokens: 500, temperature: 0.3 }), signal: AbortSignal.timeout(45000) });
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
  const outFile = process.argv[3] ?? "gradient_results.json";
  const rosterArg = process.argv[4] ?? "test";
  const cap = parseFloat(process.argv[5] ?? "Infinity");
  const roster = rosterArg === "all" ? ALL : rosterArg === "cheap" ? CHEAP : TEST;
  const all: KCase[] = JSON.parse(fs.readFileSync(path.join(__dirname, "..", casesFile), "utf-8"));
  const cases = all.slice(0, 12);
  const outPath = path.join(__dirname, "..", outFile);

  type Job = { c: KCase; m: M; level: 0 | 1 | 2 };
  const jobs: Job[] = [];
  for (const c of cases) for (const m of roster) for (const level of [0, 1, 2] as const) jobs.push({ c, m, level });
  console.log(`Roster ${rosterArg} | STYLE=${STYLE} | ${cases.length} cases x ${roster.length} models x 3 evidence-levels = ${jobs.length} calls.\n`);

  const live: Result[] = []; const flush = () => fs.writeFileSync(outPath, JSON.stringify(live, null, 2));
  let cost = 0, done = 0, capHit = false;
  const run = async (j: Job): Promise<Result> => {
    let r: Result;
    if (cost >= cap) { if (!capHit) { capHit = true; console.log(`\n!! CAP hit.`); } done++; r = { caseId: j.c.id, model: j.m.label, level: j.level, probability: null, abstained: false, sealedYes: j.c.type2.sealedYes, costUsd: 0, error: "skipped" }; live.push(r); return r; }
    try { const { content, cost: cc } = await call(j.m, prompt(j.c, j.level)); cost += cc; r = { caseId: j.c.id, model: j.m.label, level: j.level, sealedYes: j.c.type2.sealedYes, costUsd: cc, ...parse(content) }; }
    catch (e) { r = { caseId: j.c.id, model: j.m.label, level: j.level, probability: null, abstained: false, sealedYes: j.c.type2.sealedYes, costUsd: 0, error: (e as Error).message }; }
    done++; live.push(r); if (done % 10 === 0) { console.log(`  ${done}/${jobs.length} | $${cost.toFixed(4)}`); flush(); } return r;
  };
  const byP: Record<string, Job[]> = {}; for (const j of jobs) (byP[j.m.provider] ??= []).push(j);
  await Promise.all(Object.values(byP).map(js => pool(js, 6, run)));
  flush();
  console.log(`\nDone. ${live.filter(r => !r.error).length}/${live.length} ok. Cost $${cost.toFixed(4)}. -> ${outFile}`);
}
main().catch(e => { console.error(e); process.exit(1); });
