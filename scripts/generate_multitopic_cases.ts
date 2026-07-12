import "dotenv/config";
import * as fs from "fs";
import * as path from "path";

// Multi-topic case generator (Leg 3) — matched knowable/unknowable pairs in
// SYNTHETIC domains with EXACTLY-KNOWN aleatoric probabilities and ZERO
// pretraining leakage. This kills the "it's just trading" criticism AND is
// cleaner than markets: because we know the true probability (e.g. 0.5 for a fair
// coin), "false precision" on the unknowable side is unambiguous — a model that
// says 80% about a fair coin's next flip is provably overconfident.
//
// Domains: coin sequence, dice sequence, urn draws (sampling with replacement).
// Each case: a KNOWABLE question (about a shown past event in the sequence) and
// an UNKNOWABLE twin (about the next, genuinely random event) with its TRUE prob.
//
// Usage: npx ts-node -r dotenv/config scripts/generate_multitopic_cases.ts [perDomain] [outFile] [seed]

// mulberry32 seeded RNG for reproducibility
function makeRng(seed: number) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

interface MTCase {
  id: string; domain: string; evidenceText: string;
  type1: { question: string; groundTruthYes: boolean };            // about a shown past event
  // next random event: trueProbYes = EXACT aleatoric probability (the key advantage);
  // sealedYes = one sampled realization (alias for compatibility with the shared
  // runners/analysis that expect type2.sealedYes).
  type2: { question: string; trueProbYes: number; sampledYes: boolean; sealedYes: boolean };
}

function main() {
  const perDomain = parseInt(process.argv[2] ?? "8");
  const outFile = process.argv[3] ?? "multitopic_cases.json";
  const rng = makeRng(parseInt(process.argv[4] ?? "7"));
  const pick = <T,>(a: T[]) => a[Math.floor(rng() * a.length)];
  const cases: MTCase[] = [];

  for (let i = 0; i < perDomain; i++) {
    // --- COIN (fair): true P(heads)=0.5 ---
    {
      const flips = Array.from({ length: 8 }, () => (rng() < 0.5 ? "H" : "T"));
      const k = 3 + Math.floor(rng() * 4); // a shown past position
      cases.push({
        id: `coin_${i + 1}`, domain: "coin",
        evidenceText: `A fair coin (P(Heads)=0.5) is flipped repeatedly. The sequence so far is: ${flips.join(" ")}`,
        type1: { question: `Was flip #${k} in the sequence shown a HEADS?`, groundTruthYes: flips[k - 1] === "H" },
        type2: { question: `Will the NEXT (9th) flip be HEADS?`, trueProbYes: 0.5, sampledYes: rng() < 0.5, sealedYes: rng() < 0.5 },
      });
    }
    // --- DICE (fair d6): "will next roll be > 3" true P = 0.5; "was roll #k even" knowable ---
    {
      const rolls = Array.from({ length: 6 }, () => 1 + Math.floor(rng() * 6));
      const k = 2 + Math.floor(rng() * 4);
      cases.push({
        id: `dice_${i + 1}`, domain: "dice",
        evidenceText: `A fair six-sided die (each face 1-6 equally likely) is rolled repeatedly. The rolls so far are: ${rolls.join(" ")}`,
        type1: { question: `Was roll #${k} in the sequence shown an EVEN number?`, groundTruthYes: rolls[k - 1] % 2 === 0 },
        type2: { question: `Will the NEXT (7th) roll be GREATER THAN 3 (i.e. 4, 5, or 6)?`, trueProbYes: 0.5, sampledYes: (1 + Math.floor(rng() * 6)) > 3, sealedYes: (1 + Math.floor(rng() * 6)) > 3 },
      });
    }
    // --- URN (with replacement): true P(red) = reds/total ---
    {
      const reds = 2 + Math.floor(rng() * 6), blues = 2 + Math.floor(rng() * 6), total = reds + blues;
      const draws = Array.from({ length: 4 }, () => (rng() < reds / total ? "Red" : "Blue"));
      const k = 1 + Math.floor(rng() * 4);
      const p = parseFloat((reds / total).toFixed(3));
      cases.push({
        id: `urn_${i + 1}`, domain: "urn",
        evidenceText: `A bag contains ${reds} red balls and ${blues} blue balls (total ${total}). A ball is drawn, its colour noted, then REPLACED, and the bag reshuffled — each draw is independent. The draws so far are: ${draws.join(", ")}`,
        type1: { question: `Was draw #${k} in the sequence shown a RED ball?`, groundTruthYes: draws[k - 1] === "Red" },
        type2: { question: `Will the NEXT draw be a RED ball?`, trueProbYes: p, sampledYes: rng() < reds / total, sealedYes: rng() < reds / total },
      });
    }
  }

  const outPath = path.join(__dirname, "..", outFile);
  fs.writeFileSync(outPath, JSON.stringify(cases, null, 2));
  const byDomain = cases.reduce((a, c) => { a[c.domain] = (a[c.domain] ?? 0) + 1; return a; }, {} as Record<string, number>);
  console.log(`Generated ${cases.length} matched pairs. By domain: ${JSON.stringify(byDomain)}`);
  console.log(`TYPE2 true probabilities are EXACTLY known (coin/dice=0.5, urn=reds/total) — so overconfidence is unambiguous.`);
  console.log(`Written to ${outPath}`);
}

main();
