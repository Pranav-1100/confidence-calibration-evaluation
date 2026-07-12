import "dotenv/config";

// Probes candidate FREE models on NVIDIA and OpenRouter for reliability: fires a
// few simple requests at each and reports success rate + latency, so we can pick
// a dependable free roster. Respects rate limits via per-provider pacing
// (NVIDIA <=39/min, OpenRouter <=20/min + a hard daily cap). Does NOT touch HackClub.
//
// Usage: npx ts-node -r dotenv/config scripts/probe_free_models.ts [nvidia|openrouter|both] [reqsPerModel]

const CFG = {
  nvidia: { url: "https://integrate.api.nvidia.com/v1/chat/completions", key: process.env.NVIDIA_API_KEY, intervalMs: 1700 },
  openrouter: { url: "https://openrouter.ai/api/v1/chat/completions", key: process.env.OPENROUTER_API_KEY, intervalMs: 3200 },
};

const NVIDIA_MODELS = [
  // Re-checking the 3 confirmed-reliable ones:
  "mistralai/mistral-nemotron",
  "nvidia/nemotron-3-super-120b-a12b",
  "meta/llama-3.1-70b-instruct",
  // New candidates to test:
  "nvidia/nemotron-3-ultra-550b-a55b",
  "nvidia/nemotron-3.5-content-safety",
  "mistralai/mistral-medium-3.5-128b",
  "deepseek-ai/deepseek-v4-pro",
];

const OPENROUTER_MODELS = [
  "deepseek/deepseek-r1:free",
  "deepseek/deepseek-v3.2:free",
  "meta-llama/llama-3.3-70b-instruct:free",
  "google/gemma-4-31b-it:free",
  "qwen/qwen3-next-80b-a3b-instruct:free",
  "nvidia/nemotron-3-ultra-550b-a55b:free",
];

async function probeOne(provider: "nvidia" | "openrouter", model: string): Promise<{ ok: boolean; ms: number; status: number; note: string }> {
  const cfg = CFG[provider];
  const t0 = Date.now();
  try {
    const res = await fetch(cfg.url, {
      method: "POST",
      headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "Reply with exactly: OK" }], max_tokens: 10, temperature: 0 }),
      signal: AbortSignal.timeout(30000),
    });
    const ms = Date.now() - t0;
    if (!res.ok) {
      const body = (await res.text().catch(() => "")).slice(0, 80).replace(/\s+/g, " ");
      return { ok: false, ms, status: res.status, note: body };
    }
    const data = await res.json() as any;
    const content = data?.choices?.[0]?.message?.content ?? "";
    return { ok: !!content, ms, status: 200, note: content ? "" : "empty" };
  } catch (e) {
    return { ok: false, ms: Date.now() - t0, status: 0, note: (e as Error).name === "TimeoutError" ? "timeout" : (e as Error).message.slice(0, 60) };
  }
}

async function probeProvider(provider: "nvidia" | "openrouter", models: string[], reqs: number) {
  const cfg = CFG[provider];
  console.log(`\n${"=".repeat(80)}\n${provider.toUpperCase()} — ${models.length} models x ${reqs} reqs (pacing ${cfg.intervalMs}ms)\n${"=".repeat(80)}`);
  console.log(`${"Model".padEnd(52)}${"ok".padStart(6)}${"avg ms".padStart(9)}  notes`);
  for (const model of models) {
    let ok = 0; const lats: number[] = []; const notes = new Set<string>();
    for (let i = 0; i < reqs; i++) {
      const r = await probeOne(provider, model);
      if (r.ok) { ok++; lats.push(r.ms); } else notes.add(`${r.status}:${r.note}`);
      await new Promise(res => setTimeout(res, cfg.intervalMs)); // pace to respect rate limit
    }
    const avg = lats.length ? Math.round(lats.reduce((a, b) => a + b, 0) / lats.length) : 0;
    const verdict = ok === reqs ? "✓ reliable" : ok === 0 ? "✗ dead" : "~ flaky";
    console.log(`${model.padEnd(52)}${(ok + "/" + reqs).padStart(6)}${(avg || "-").toString().padStart(9)}  ${verdict} ${[...notes].slice(0, 2).join(" | ")}`);
  }
}

async function main() {
  const which = process.argv[2] ?? "both";
  const reqs = parseInt(process.argv[3] ?? "4");
  if (which === "nvidia" || which === "both") await probeProvider("nvidia", NVIDIA_MODELS, reqs);
  if (which === "openrouter" || which === "both") await probeProvider("openrouter", OPENROUTER_MODELS, reqs);
  console.log("\nDone.");
}

main().catch(e => { console.error(e); process.exit(1); });
