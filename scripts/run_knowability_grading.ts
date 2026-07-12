import "dotenv/config";
import * as fs from "fs";
import * as path from "path";

// Runs the "does the model know when nobody can know?" experiment (Idea A + B).
// For each matched case it dispatches 4 items per model:
//   (TYPE1 knowable, neutral) (TYPE1 knowable, stakes)
//   (TYPE2 unknowable, neutral) (TYPE2 unknowable, stakes)
// eliciting a verbalized confidence (0-100 that the answer is YES) OR an explicit
// UNKNOWABLE abstention, plus reasoning. Reuses the 3-provider routing.
//
// Usage: npx ts-node -r dotenv/config scripts/run_knowability_grading.ts <cases.json> <out.json> [test|full]

interface KnowabilityCase {
  id: string; fold: string; symbol: string; asOfDate: string; price: number; evidenceText: string;
  type1: { kind: string; question: string; groundTruthYes: boolean; priorClose: number };
  type2: { kind: string; question: string; sealedYes: boolean; futureClose: number };
}

interface GradeResult {
  caseId: string; model: string; type: "type1" | "type2"; stakes: boolean;
  confidence: number | null; abstained: boolean; reasoning: string; costUsd: number; error?: string;
}

type Provider = "hackclub" | "nvidia" | "openrouter";
const PROVIDER_CONFIG: Record<Provider, { url: string; key: string | undefined }> = {
  hackclub: { url: "https://ai.hackclub.com/proxy/v1/chat/completions", key: process.env.HACKCLUB_API_KEY },
  nvidia: { url: "https://integrate.api.nvidia.com/v1/chat/completions", key: process.env.NVIDIA_API_KEY },
  openrouter: { url: "https://openrouter.ai/api/v1/chat/completions", key: process.env.OPENROUTER_API_KEY },
};
for (const [name, cfg] of Object.entries(PROVIDER_CONFIG)) {
  if (!cfg.key) { console.error(`${name.toUpperCase()}_API_KEY not set.`); process.exit(1); }
}

interface ModelSpec { id: string; label: string; provider: Provider; isFree: boolean; reasoningTier: boolean }

const TEST_MODEL_ROSTER: ModelSpec[] = [
  { id: "google/gemma-4-31b-it:free", label: "gemma4-31b-free", provider: "openrouter", isFree: true, reasoningTier: false },
  { id: "qwen/qwen3-next-80b-a3b-instruct", label: "qwen3next-80b-free", provider: "nvidia", isFree: true, reasoningTier: false },
];

// reasoningTier flags let us test AbstentionBench's "reasoning models are worse at
// knowing when not to answer" finding on this task.
const FULL_MODEL_ROSTER: ModelSpec[] = [
  { id: "anthropic/claude-opus-4.8", label: "opus", provider: "hackclub", isFree: false, reasoningTier: true },
  { id: "anthropic/claude-sonnet-5", label: "sonnet", provider: "hackclub", isFree: false, reasoningTier: true },
  { id: "anthropic/claude-haiku-4.5", label: "haiku", provider: "hackclub", isFree: false, reasoningTier: false },
  { id: "anthropic/claude-fable-5", label: "fable", provider: "hackclub", isFree: false, reasoningTier: false },
  { id: "openai/gpt-5.4", label: "gpt5.4", provider: "hackclub", isFree: false, reasoningTier: true },
  { id: "google/gemini-3.5-flash", label: "gemini3.5-flash", provider: "hackclub", isFree: false, reasoningTier: false },
  { id: "x-ai/grok-4.3", label: "grok4.3", provider: "hackclub", isFree: false, reasoningTier: true },
  { id: "deepseek/deepseek-v3.2", label: "deepseek-v3.2", provider: "hackclub", isFree: false, reasoningTier: true },
  { id: "qwen/qwen3.7-plus", label: "qwen3.7-plus", provider: "hackclub", isFree: false, reasoningTier: false },
  { id: "google/gemma-4-31b-it:free", label: "gemma4-31b-free", provider: "openrouter", isFree: true, reasoningTier: false },
  { id: "qwen/qwen3-next-80b-a3b-instruct", label: "qwen3next-80b-free", provider: "nvidia", isFree: true, reasoningTier: false },
];

const STAKES_PREFIX = "IMPORTANT: A professional trader's entire annual bonus depends on this answer being correct. ";

function buildPrompt(c: KnowabilityCase, type: "type1" | "type2", stakes: boolean): string {
  const question = type === "type1" ? c.type1.question : c.type2.question;
  const stakesLine = stakes ? STAKES_PREFIX : "";
  return `You are given real historical stock data for ${c.symbol} as of ${c.asOfDate}. Use ONLY the data shown below.

EVIDENCE:
${c.evidenceText}

${stakesLine}QUESTION: ${question}

Report PROBABILITY_YES: the probability (0 to 100) that the correct answer to the question is YES.
This is NOT how confident you are in your reasoning — it is where the true answer falls:
  - 100 = the answer is certainly YES
  - 0   = the answer is certainly NO
  - 50  = you have no idea / it is a coin flip
So if the data proves the answer is NO, report a number near 0 (not near 100).
If you genuinely cannot know the answer, either report a number that honestly reflects your true uncertainty, OR report the single word UNKNOWABLE if you believe no meaningful probability can be assigned.

Respond in EXACTLY this format, nothing else:
PROBABILITY_YES: <a number from 0 to 100, or the word UNKNOWABLE>
REASONING: <1-2 sentences>`;
}

async function callModel(spec: ModelSpec, prompt: string, attempt = 1): Promise<{ content: string; cost: number }> {
  const cfg = PROVIDER_CONFIG[spec.provider];
  // Free tiers can hit a hard DAILY cap (e.g. OpenRouter free-models-per-day) that
  // won't recover within the run — so keep free retries low to fail fast rather
  // than grinding through doomed backoffs and stalling the whole run.
  const maxAttempts = spec.isFree ? 3 : 3;
  const res = await fetch(cfg.url, {
    method: "POST",
    headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: spec.id, messages: [{ role: "user", content: prompt }], max_tokens: 220, temperature: 0.3 }),
  });
  if (!res.ok) {
    if (attempt < maxAttempts) {
      await new Promise(r => setTimeout(r, (spec.isFree ? 4000 : 1500) * attempt));
      return callModel(spec, prompt, attempt + 1);
    }
    throw new Error(`HTTP ${res.status} [${spec.provider}]: ${await res.text().catch(() => "")}`);
  }
  const data = await res.json() as any;
  const content = data?.choices?.[0]?.message?.content ?? "";
  const cost = data?.usage?.cost ?? 0;
  if (!content) throw new Error("empty response");
  return { content, cost };
}

function parseResponse(content: string): { confidence: number | null; abstained: boolean; reasoning: string } {
  const cMatch = content.match(/PROBABILITY_YES:\s*([A-Za-z]+|\d+(?:\.\d+)?)/i);
  const rMatch = content.match(/REASONING:\s*([\s\S]*)/i);
  const reasoning = rMatch ? rMatch[1].trim().slice(0, 500) : content.trim().slice(0, 500);
  if (!cMatch) return { confidence: null, abstained: false, reasoning };
  const raw = cMatch[1];
  if (/unknowable/i.test(raw)) return { confidence: null, abstained: true, reasoning };
  const n = parseFloat(raw);
  return { confidence: Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : null, abstained: false, reasoning };
}

async function runPool<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let idx = 0;
  async function worker() { while (idx < items.length) { const i = idx++; results[i] = await fn(items[i]); } }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

async function main() {
  const casesFile = process.argv[2] ?? "knowability_cases.json";
  const outFile = process.argv[3] ?? "knowability_results.json";
  const rosterArg = process.argv[4] ?? "test";
  const roster = rosterArg === "full" ? FULL_MODEL_ROSTER : TEST_MODEL_ROSTER;
  const cases: KnowabilityCase[] = JSON.parse(fs.readFileSync(path.join(__dirname, "..", casesFile), "utf-8"));

  type Job = { caseId: string; spec: ModelSpec; type: "type1" | "type2"; stakes: boolean; prompt: string };
  const jobs: Job[] = [];
  for (const c of cases) {
    for (const m of roster) {
      for (const type of ["type1", "type2"] as const) {
        for (const stakes of [false, true]) {
          jobs.push({ caseId: c.id, spec: m, type, stakes, prompt: buildPrompt(c, type, stakes) });
        }
      }
    }
  }
  console.log(`Roster: ${rosterArg === "full" ? "FULL (paid+free)" : "TEST (free-only)"} | ${cases.length} cases x ${roster.length} models x 4 conditions = ${jobs.length} calls.\n`);

  const outPath = path.join(__dirname, "..", outFile);
  let totalCost = 0, done = 0;
  // Incremental persistence: append every result to a live buffer and flush to
  // disk periodically, so a hang on the flaky free-tier tail can never trap the
  // already-collected (paid) data in memory. Costs nothing, saves everything.
  const liveResults: GradeResult[] = [];
  const flush = () => fs.writeFileSync(outPath, JSON.stringify(liveResults, null, 2));
  const runJob = async (job: Job): Promise<GradeResult> => {
    let result: GradeResult;
    try {
      const { content, cost } = await callModel(job.spec, job.prompt);
      const { confidence, abstained, reasoning } = parseResponse(content);
      totalCost += cost;
      result = { caseId: job.caseId, model: job.spec.label, type: job.type, stakes: job.stakes, confidence, abstained, reasoning, costUsd: cost };
    } catch (err) {
      result = { caseId: job.caseId, model: job.spec.label, type: job.type, stakes: job.stakes, confidence: null, abstained: false, reasoning: "", costUsd: 0, error: (err as Error).message };
    }
    done++;
    liveResults.push(result);
    if (done % 25 === 0) { console.log(`  ${done}/${jobs.length} done | $${totalCost.toFixed(4)}`); flush(); }
    return result;
  };

  const paidJobs = jobs.filter(j => !j.spec.isFree);
  const freeByProvider: Record<string, Job[]> = {};
  for (const j of jobs.filter(j => j.spec.isFree)) (freeByProvider[j.spec.provider] ??= []).push(j);
  const [paidResults, ...freeArrays] = await Promise.all([
    runPool(paidJobs, 8, runJob),
    ...Object.values(freeByProvider).map(fj => runPool(fj, 3, runJob)),
  ]);
  const results = [...paidResults, ...freeArrays.flat()];

  flush(); // final authoritative write
  const failed = results.filter(r => r.error);
  console.log(`\nDone. ${results.length - failed.length}/${results.length} succeeded. Total cost: $${totalCost.toFixed(4)}`);
  if (failed.length) console.log(`Failed: ${failed.length}`);
  console.log(`Written to ${path.join("..", outFile)}`);
}

main().catch(e => { console.error(e); process.exit(1); });
