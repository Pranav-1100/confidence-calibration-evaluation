import "dotenv/config";
import * as fs from "fs";
import * as path from "path";
import { dataCache } from "../src/data/DataCache";
import { IndicatorService, OHLCVData } from "../src/shared/services/indicator.service";
import { RegimeDetector } from "../src/environment/RegimeDetector";
import { NIFTY50_SYMBOLS, Candle } from "../src/environment/types";

// POST-CUTOFF, OUTCOME-BALANCED knowability cases for the agentic earned-check.
//
// Two upgrades over generate_knowability_cases.ts:
//   1. POST-CUTOFF: every as-of date is AFTER 2026-01-31, so the sealed 10-day
//      outcome lies beyond every tested model's training cutoff (Jan 2026 for
//      Claude, earlier for others). The model provably cannot have memorised it —
//      the question is genuinely unknowable, not merely unknowable-in-principle.
//   2. OUTCOME-BALANCED: we build a large pool, then select an equal number of
//      UP and DOWN outcomes. Chance is then exactly 50% by construction, so any
//      accuracy above 50% among committed ANSWER calls = genuine skill. This
//      removes the base-rate confound that made the old 12-case (83% up) set's
//      earned-check uninterpretable.

const HORIZON_DAYS = 10;
const CUTOFF = new Date("2026-01-31T00:00:00Z"); // as-of must be strictly after this
const AS_OF_SPACING = 18;                        // >=18 trading days between as-of points of the same symbol
const indicatorSvc = new IndicatorService();
const regimeDetector = new RegimeDetector();

interface KnowabilityCase {
  id: string;
  fold: string;
  symbol: string;
  asOfDate: string;
  price: number;
  evidenceText: string;
  type1: { kind: "KNOWABLE"; question: string; groundTruthYes: boolean; priorDate: string; priorClose: number; };
  type2: { kind: "UNKNOWABLE"; question: string; sealedYes: boolean; futureDate: string; futureClose: number; };
}

function fmtDate(d: Date): string { return d.toISOString().split("T")[0]; }

function buildEvidenceText(symbol: string, asOfDate: string, priorDate: string, priorClose: number, candles: Candle[], idx: number): string {
  const slice = candles.slice(Math.max(0, idx - 59), idx + 1);
  const ohlcv: OHLCVData = {
    open: slice.map(c => c.open), high: slice.map(c => c.high),
    low: slice.map(c => c.low), close: slice.map(c => c.close), volume: slice.map(c => c.volume),
  };
  const ind = indicatorSvc.calculate(ohlcv);
  const closes = ohlcv.close;
  const price = closes[closes.length - 1];
  const regime = regimeDetector.detect(candles, idx);
  const atrPct = ind.atr ? (ind.atr / price) * 100 : 0;
  const vol5 = ohlcv.volume.slice(-5).reduce((a, b) => a + b, 0) / 5;
  const vol20 = ohlcv.volume.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, ohlcv.volume.length);
  return [
    `Symbol: ${symbol} | As-of date: ${asOfDate} | Current close: ₹${price.toFixed(2)}`,
    `Close 10 trading days ago (${priorDate}): ₹${priorClose.toFixed(2)}`,
    `RSI(14): ${ind.rsi != null ? ind.rsi.toFixed(1) : "N/A"} | EMA20: ₹${ind.ema20?.toFixed(2) ?? "N/A"} | EMA50: ₹${ind.ema50?.toFixed(2) ?? "N/A"}`,
    `MACD histogram: ${ind.macd ? ind.macd.histogram.toFixed(3) : "N/A"}`,
    `ATR(14): ₹${ind.atr?.toFixed(2) ?? "N/A"} (${atrPct.toFixed(1)}% of price)`,
    `Volume ratio (5d/20d): ${(vol20 > 0 ? vol5 / vol20 : 1).toFixed(2)}x`,
    `Regime classification: ${regime.label}`,
  ].join("\n");
}

async function main() {
  const perClass = parseInt(process.argv[2] ?? "20");         // final = perClass UP + perClass DOWN
  const outFile = process.argv[3] ?? "knowability_postcutoff.json";
  const cutoffTs = Math.floor(CUTOFF.getTime() / 1000);

  const pool: KnowabilityCase[] = [];
  for (const symbol of NIFTY50_SYMBOLS) {
    try {
      const { candles } = await dataCache.getCandles(symbol, "NSE", "1D", new Date("2025-08-01"), new Date("2026-07-01"));
      if (candles.length < 80) continue;
      let lastIdx = -1e9;
      for (let idx = 60; idx + HORIZON_DAYS < candles.length; idx++) {
        if (candles[idx].time <= cutoffTs) continue;           // as-of must be post-cutoff
        if (idx - lastIdx < AS_OF_SPACING) continue;           // spacing guard within a symbol
        const priorIdx = idx - HORIZON_DAYS, futureIdx = idx + HORIZON_DAYS;
        const price = candles[idx].close, priorClose = candles[priorIdx].close, futureClose = candles[futureIdx].close;
        const asOfDate = fmtDate(new Date(candles[idx].time * 1000));
        const priorDate = fmtDate(new Date(candles[priorIdx].time * 1000));
        const futureDate = fmtDate(new Date(candles[futureIdx].time * 1000));
        pool.push({
          id: `postcutoff_${symbol}_${asOfDate}`,
          fold: "post-cutoff 2026", symbol, asOfDate, price,
          evidenceText: buildEvidenceText(symbol, asOfDate, priorDate, priorClose, candles, idx),
          type1: {
            kind: "KNOWABLE",
            question: `Based ONLY on the data shown, is the current close (₹${price.toFixed(2)}) HIGHER than the close 10 trading days ago (${priorDate})? In other words, did the price RISE over the past 10 trading days?`,
            groundTruthYes: price > priorClose, priorDate, priorClose: parseFloat(priorClose.toFixed(2)),
          },
          type2: {
            kind: "UNKNOWABLE",
            question: `Will ${symbol}'s closing price 10 trading days from ${asOfDate} be HIGHER than the current close (₹${price.toFixed(2)})? In other words, will the price RISE over the next 10 trading days?`,
            sealedYes: futureClose > price, futureDate, futureClose: parseFloat(futureClose.toFixed(2)),
          },
        });
        lastIdx = idx;
      }
    } catch { continue; }
  }

  // Balance by the sealed outcome: equal UP and DOWN, spread across symbols.
  const ups = pool.filter(c => c.type2.sealedYes);
  const downs = pool.filter(c => !c.type2.sealedYes);
  const take = Math.min(perClass, ups.length, downs.length);
  // interleave by symbol diversity: sort each class so we don't over-draw one symbol first
  const spread = (arr: KnowabilityCase[]) => {
    const bySym: Record<string, KnowabilityCase[]> = {};
    for (const c of arr) (bySym[c.symbol] ??= []).push(c);
    const out: KnowabilityCase[] = []; let added = true;
    while (added) { added = false; for (const s of Object.keys(bySym)) { const n = bySym[s].shift(); if (n) { out.push(n); added = true; } } }
    return out;
  };
  const selected = [...spread(ups).slice(0, take), ...spread(downs).slice(0, take)];

  const outPath = path.join(__dirname, "..", outFile);
  fs.writeFileSync(outPath, JSON.stringify(selected, null, 2));
  const selUps = selected.filter(c => c.type2.sealedYes).length;
  console.log(`Pool: ${pool.length} post-cutoff cases (${ups.length} up / ${downs.length} down).`);
  console.log(`Selected BALANCED set: ${selected.length} cases (${selUps} up / ${selected.length - selUps} down).`);
  console.log(`  Unique symbols: ${new Set(selected.map(c => c.symbol)).size}`);
  console.log(`  As-of range: ${selected.map(c => c.asOfDate).sort()[0]} -> ${selected.map(c => c.asOfDate).sort().slice(-1)[0]}`);
  console.log(`Written to ${outPath}`);
}

main().catch(e => { console.error(e); process.exit(1); });
