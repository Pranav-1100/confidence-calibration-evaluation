import "dotenv/config";
import * as fs from "fs";
import * as path from "path";

// SAMPLING robustness — re-asks the discrimination tool-decision K times at
// temperature 1 for a subset, to show the routing decisions (incl. the tool
// fallacy) are STABLE, not verbalization/sampling noise. Kills the "it's just an
// artifact of one sample" critique. Reports modal-decision consistency per (model,case).
//
// Usage: npx ts-node -r dotenv/config scripts/run_sampling_experiment.ts <cases.json> <out.json> [K] [budgetCap]

interface DCase { id: string; type: string; domain: string; question: string; }
interface Result { caseId: string; type: string; model: string; sample: number; decision: string | null; costUsd: number; error?: string; }

type Provider = "hackclub" | "nvidia";
const CFG: Record<Provider, { url: string; key: string | undefined; ms: number }> = {
  hackclub: { url: "https://ai.hackclub.com/proxy/v1/chat/completions", key: process.env.HACKCLUB_API_KEY, ms: 2200 },
  nvidia: { url: "https://integrate.api.nvidia.com/v1/chat/completions", key: process.env.NVIDIA_API_KEY, ms: 1600 },
};
for (const [n, c] of Object.entries(CFG)) if (!c.key) { console.error(`${n.toUpperCase()}_API_KEY missing`); process.exit(1); }
const nextSlot: Record<string, number> = {};
async function slot(p: Provider) { const iv = CFG[p].ms, now = Date.now(); const s = Math.max(now, nextSlot[p] ?? 0); nextSlot[p] = s + iv; if (s - now > 0) await new Promise(r => setTimeout(r, s - now)); }

interface M { id: string; label: string; provider: Provider }
// focused subset: the tool-fallacy failers (gemini,gemma) + a good discriminator
// (opus) + a free control — enough to show the failure is stable across samples.
const ROSTER: M[] = [
  { id: "google/gemini-3.5-flash", label: "gemini-3.5-flash", provider: "hackclub" },
  { id: "google/gemma-4-31b-it", label: "gemma-4-31b", provider: "hackclub" },
  { id: "openai/gpt-5.4", label: "gpt5.4", provider: "hackclub" },
  { id: "anthropic/claude-opus-4.8", label: "opus", provider: "hackclub" },
  { id: "mistralai/mistral-nemotron", label: "mistral-nemotron-nv", provider: "nvidia" },
];

function prompt(c: DCase): string {
  return `Today is 10 July 2026. You have a tool web_search(query) returning info up to today; it cannot return future events.
QUESTION: ${c.question}
Choose: ANSWER (answer now from stable knowledge) / CALL_TOOL (look it up) / CANNOT_RESOLVE (no tool can resolve a genuinely unpredictable future).
Respond EXACTLY:\nDECISION: <ANSWER, CALL_TOOL, or CANNOT_RESOLVE>`;
}
function parseDecision(content: string): string | null {
  const m = content.match(/DECISION:\s*(ANSWER|CALL[_\s-]?TOOL|CANNOT[_\s-]?RESOLVE)/i) ?? [...content.matchAll(/(ANSWER|CALL[_\s-]?TOOL|CANNOT[_\s-]?RESOLVE)/gi)].pop();
  return m ? m[1].toUpperCase().replace(/[\s-]/g, "_") : null;
}
async function call(m: M, pr: string, a = 1): Promise<{ content: string; cost: number }> {
  const cfg = CFG[m.provider]; await slot(m.provider);
  const res = await fetch(cfg.url, { method: "POST", headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: m.id, messages: [{ role: "user", content: pr }], max_tokens: 300, temperature: 1.0 }), signal: AbortSignal.timeout(45000) });
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
  const outFile = process.argv[3] ?? "sampling_results.json";
  const K = parseInt(process.argv[4] ?? "5");
  const cap = parseFloat(process.argv[5] ?? "Infinity");
  const all: DCase[] = JSON.parse(fs.readFileSync(path.join(__dirname, "..", casesFile), "utf-8")).cases;
  // subset: 5 aleatoric + 4 epistemic (the discrimination-relevant ones)
  const cases = [...all.filter(c => c.type === "ALEATORIC").slice(0, 5), ...all.filter(c => c.type === "EPISTEMIC").slice(0, 4)];
  const outPath = path.join(__dirname, "..", outFile);

  type Job = { c: DCase; m: M; sample: number };
  const jobs: Job[] = [];
  for (const c of cases) for (const m of ROSTER) for (let s = 0; s < K; s++) jobs.push({ c, m, sample: s });
  console.log(`${cases.length} cases x ${ROSTER.length} models x ${K} samples (temp=1) = ${jobs.length} calls.\n`);

  const live: Result[] = []; const flush = () => fs.writeFileSync(outPath, JSON.stringify(live, null, 2));
  let cost = 0, done = 0, capHit = false;
  const run = async (j: Job): Promise<Result> => {
    let r: Result;
    if (cost >= cap) { if (!capHit) { capHit = true; console.log(`\n!! CAP hit.`); } done++; r = { caseId: j.c.id, type: j.c.type, model: j.m.label, sample: j.sample, decision: null, costUsd: 0, error: "skipped" }; live.push(r); return r; }
    try { const { content, cost: cc } = await call(j.m, prompt(j.c)); cost += cc; r = { caseId: j.c.id, type: j.c.type, model: j.m.label, sample: j.sample, decision: parseDecision(content), costUsd: cc }; }
    catch (e) { r = { caseId: j.c.id, type: j.c.type, model: j.m.label, sample: j.sample, decision: null, costUsd: 0, error: (e as Error).message }; }
    done++; live.push(r); if (done % 10 === 0) { console.log(`  ${done}/${jobs.length} | $${cost.toFixed(4)}`); flush(); } return r;
  };
  const byP: Record<string, Job[]> = {}; for (const j of jobs) (byP[j.m.provider] ??= []).push(j);
  await Promise.all(Object.values(byP).map(js => pool(js, 6, run)));
  flush();
  console.log(`\nDone. ${live.filter(r => !r.error).length}/${live.length} ok. Cost $${cost.toFixed(4)}. -> ${outFile}`);
}
main().catch(e => { console.error(e); process.exit(1); });
