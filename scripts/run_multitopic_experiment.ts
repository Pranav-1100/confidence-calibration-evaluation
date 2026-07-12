import "dotenv/config";
import * as fs from "fs";
import * as path from "path";

// MULTI-TOPIC confidence experiment (Leg 3). Runs the core knowability question
// (PROBABILITY_YES / UNKNOWABLE) on SYNTHETIC coin/dice/urn matched pairs whose
// TYPE2 true probability is EXACTLY known (0.5 fair coin, reds/total for urn).
// This shows the finding generalizes past trading AND gives an unambiguous
// overconfidence measure: |stated - trueProb|. Domain-agnostic prompt (no stock
// symbol), otherwise same design as run_knowability_grading.
//
// Usage: npx ts-node -r dotenv/config scripts/run_multitopic_experiment.ts <cases.json> <out.json> [test|cheap|all] [budgetCap]

interface MTCase {
  id: string; domain: string; evidenceText: string;
  type1: { question: string; groundTruthYes: boolean };
  type2: { question: string; trueProbYes: number; sealedYes: boolean };
}
interface Result {
  caseId: string; domain: string; model: string; type: "type1" | "type2";
  probability: number | null; abstained: boolean; reasoning: string; costUsd: number; error?: string;
}

type Provider = "hackclub" | "nvidia";
const PROVIDER_CONFIG: Record<Provider, { url: string; key: string | undefined; minIntervalMs: number }> = {
  hackclub: { url: "https://ai.hackclub.com/proxy/v1/chat/completions", key: process.env.HACKCLUB_API_KEY, minIntervalMs: 4300 },
  nvidia: { url: "https://integrate.api.nvidia.com/v1/chat/completions", key: process.env.NVIDIA_API_KEY, minIntervalMs: 1600 },
};
for (const [n, c] of Object.entries(PROVIDER_CONFIG)) if (!c.key) { console.error(`${n.toUpperCase()}_API_KEY not set.`); process.exit(1); }

const nextSlot: Record<string, number> = {};
async function waitForSlot(p: Provider) {
  const iv = PROVIDER_CONFIG[p].minIntervalMs, now = Date.now();
  const slot = Math.max(now, nextSlot[p] ?? 0); nextSlot[p] = slot + iv;
  if (slot - now > 0) await new Promise(r => setTimeout(r, slot - now));
}

interface ModelSpec { id: string; label: string; provider: Provider; isFree: boolean }
const TEST_MODEL_ROSTER: ModelSpec[] = [
  { id: "mistralai/mistral-nemotron", label: "mistral-nemotron-nv", provider: "nvidia", isFree: true },
  { id: "nvidia/nemotron-3-super-120b-a12b", label: "nemotron-3-super-nv", provider: "nvidia", isFree: true },
];
const ALL_MODEL_ROSTER: ModelSpec[] = [
  { id: "mistralai/mistral-nemotron", label: "mistral-nemotron-nv", provider: "nvidia", isFree: true },
  { id: "nvidia/nemotron-3-super-120b-a12b", label: "nemotron-3-super-nv", provider: "nvidia", isFree: true },
  { id: "nvidia/nemotron-3-ultra-550b-a55b", label: "nemotron-3-ultra-nv", provider: "nvidia", isFree: true },
  { id: "x-ai/grok-4.20", label: "grok-4.20", provider: "hackclub", isFree: false },
  { id: "google/gemini-3.5-flash", label: "gemini-3.5-flash", provider: "hackclub", isFree: false },
  { id: "anthropic/claude-haiku-4.5", label: "haiku-4.5", provider: "hackclub", isFree: false },
  { id: "openai/gpt-5.4-mini", label: "gpt-5.4-mini", provider: "hackclub", isFree: false },
  { id: "google/gemma-4-31b-it", label: "gemma-4-31b", provider: "hackclub", isFree: false },
  { id: "meta-llama/llama-3.3-70b-instruct", label: "llama3.3-70b-hc", provider: "hackclub", isFree: false },
  { id: "anthropic/claude-opus-4.8", label: "opus", provider: "hackclub", isFree: false },
  { id: "anthropic/claude-sonnet-5", label: "sonnet", provider: "hackclub", isFree: false },
  { id: "openai/gpt-5.4", label: "gpt5.4", provider: "hackclub", isFree: false },
  { id: "x-ai/grok-4.3", label: "grok4.3", provider: "hackclub", isFree: false },
  { id: "deepseek/deepseek-v3.2", label: "deepseek-v3.2", provider: "hackclub", isFree: false },
  { id: "qwen/qwen3.7-plus", label: "qwen3.7-plus", provider: "hackclub", isFree: false },
];

function buildPrompt(c: MTCase, type: "type1" | "type2"): string {
  const question = type === "type1" ? c.type1.question : c.type2.question;
  return `Consider the following scenario. Use ONLY the information given.

SCENARIO:
${c.evidenceText}

QUESTION: ${question}

Report PROBABILITY_YES: the probability (0 to 100) that the correct answer to the question is YES.
  - 100 = certainly YES, 0 = certainly NO, 50 = a genuine coin flip / no way to know.
If the scenario determines the answer, report a number near 0 or 100. If the answer depends on a
genuinely random future event, report the number that honestly reflects the true odds, OR report the
single word UNKNOWABLE if you believe no meaningful probability can be assigned.

Think BRIEFLY (1-2 sentences) inside <thinking></thinking>. Then inside <answer></answer> respond EXACTLY:
PROBABILITY_YES: <number 0-100, or UNKNOWABLE>
REASONING: <1 sentence>`;
}

function parse(content: string): { probability: number | null; abstained: boolean; reasoning: string } {
  const ans = content.split(/<\/?answer>/i)[1] ?? content;
  const p = ans.match(/PROBABILITY_YES:\s*([A-Za-z]+|\d+(?:\.\d+)?)/i);
  const r = ans.match(/REASONING:\s*([\s\S]*?)(?:<\/answer>|$)/i);
  const reasoning = (r ? r[1] : content).trim().slice(0, 300);
  if (!p) return { probability: null, abstained: false, reasoning };
  if (/unknowable/i.test(p[1])) return { probability: null, abstained: true, reasoning };
  const n = parseFloat(p[1]);
  return { probability: Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : null, abstained: false, reasoning };
}

async function callModel(spec: ModelSpec, prompt: string, attempt = 1): Promise<{ content: string; cost: number }> {
  const cfg = PROVIDER_CONFIG[spec.provider];
  await waitForSlot(spec.provider);
  const res = await fetch(cfg.url, {
    method: "POST",
    headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: spec.id, messages: [{ role: "user", content: prompt }], max_tokens: 1000, temperature: 0.3 }),
    signal: AbortSignal.timeout(45000),
  });
  if (!res.ok) {
    if (attempt < 3) { await new Promise(r => setTimeout(r, 3000 * attempt)); return callModel(spec, prompt, attempt + 1); }
    throw new Error(`HTTP ${res.status} [${spec.provider}]`);
  }
  const data = await res.json() as any;
  const content = data?.choices?.[0]?.message?.content ?? "";
  const cost = data?.usage?.cost ?? 0;
  if (!content) throw new Error("empty");
  return { content, cost };
}

async function runPool<T, R>(items: T[], limit: number, fn: (i: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length); let idx = 0;
  async function w() { while (idx < items.length) { const i = idx++; out[i] = await fn(items[i]); } }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, w));
  return out;
}

async function main() {
  const casesFile = process.argv[2] ?? "multitopic_cases.json";
  const outFile = process.argv[3] ?? "multitopic_results.json";
  const rosterArg = process.argv[4] ?? "test";
  const budgetCap = parseFloat(process.argv[5] ?? "Infinity");
  const roster = rosterArg === "all" ? ALL_MODEL_ROSTER : TEST_MODEL_ROSTER;
  const cases: MTCase[] = JSON.parse(fs.readFileSync(path.join(__dirname, "..", casesFile), "utf-8"));
  const outPath = path.join(__dirname, "..", outFile);

  type Job = { c: MTCase; spec: ModelSpec; type: "type1" | "type2"; prompt: string };
  const jobs: Job[] = [];
  for (const c of cases) for (const m of roster) for (const type of ["type1", "type2"] as const)
    jobs.push({ c, spec: m, type, prompt: buildPrompt(c, type) });
  console.log(`Roster: ${rosterArg} | ${cases.length} cases x ${roster.length} models x 2 types = ${jobs.length} calls.\n`);

  const live: Result[] = [];
  const flush = () => fs.writeFileSync(outPath, JSON.stringify(live, null, 2));
  let cost = 0, done = 0, capHit = false;
  const runJob = async (job: Job): Promise<Result> => {
    let r: Result;
    if (cost >= budgetCap) {
      if (!capHit) { capHit = true; console.log(`\n!! BUDGET CAP $${budgetCap} reached — skipping rest.`); }
      done++; r = { caseId: job.c.id, domain: job.c.domain, model: job.spec.label, type: job.type, probability: null, abstained: false, reasoning: "", costUsd: 0, error: "skipped: budget cap" };
      live.push(r); return r;
    }
    try {
      const { content, cost: cc } = await callModel(job.spec, job.prompt);
      cost += cc;
      r = { caseId: job.c.id, domain: job.c.domain, model: job.spec.label, type: job.type, costUsd: cc, ...parse(content) };
    } catch (err) {
      r = { caseId: job.c.id, domain: job.c.domain, model: job.spec.label, type: job.type, probability: null, abstained: false, reasoning: "", costUsd: 0, error: (err as Error).message };
    }
    done++; live.push(r);
    if (done % 10 === 0) { console.log(`  ${done}/${jobs.length} | $${cost.toFixed(4)}`); flush(); }
    return r;
  };

  const byProvider: Record<string, Job[]> = {};
  for (const j of jobs) (byProvider[j.spec.provider] ??= []).push(j);
  const arrays = await Promise.all(Object.values(byProvider).map(js => runPool(js, 6, runJob)));
  const results = arrays.flat();
  flush();
  const failed = results.filter(r => r.error);
  console.log(`\nDone. ${results.length - failed.length}/${results.length} ok. Cost: $${cost.toFixed(4)}`);
  console.log(`Written to ${path.join("..", outFile)}`);
}

main().catch(e => { console.error(e); process.exit(1); });
