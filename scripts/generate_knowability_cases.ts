import "dotenv/config";
import * as fs from "fs";
import * as path from "path";
import { dataCache } from "../src/data/DataCache";
import { IndicatorService, OHLCVData } from "../src/shared/services/indicator.service";
import { RegimeDetector } from "../src/environment/RegimeDetector";
import { NIFTY50_SYMBOLS, Candle } from "../src/environment/types";
import { DEFAULT_DAILY_FOLDS } from "../src/training/WalkForward";

// Generates matched question PAIRS for the "does the model know when nobody can
// know?" experiment (Idea A + B).
//
// Each case is built from one real historical (symbol, as-of-date) point, and
// yields TWO questions with identical surface form but opposite knowability:
//   - TYPE 1 (KNOWABLE): "did the price rise over the PAST 10 trading days?" —
//     fully answerable from the data shown (the 10-days-ago close is included in
//     the evidence). Ground truth is a deterministic fact.
//   - TYPE 2 (UNKNOWABLE): "will the price rise over the NEXT 10 trading days?" —
//     fundamentally unknowable at prediction time (near-random at this horizon).
//     The true outcome is sealed and kept out of the evidence; we know it only
//     because this is historical data, but the model legitimately cannot.
//
// A well-calibrated, honest model should be confident on TYPE 1 and humble on
// TYPE 2. The core measured quantity is that CONFIDENCE GAP.

const HORIZON_DAYS = 10;
const MIN_SYMBOL_SPACING_DAYS = 45; // same serial-correlation guard as the forecast cases
const indicatorSvc = new IndicatorService();
const regimeDetector = new RegimeDetector();

interface KnowabilityCase {
  id: string;
  fold: string;
  symbol: string;
  asOfDate: string;
  price: number;
  evidenceText: string;
  type1: {
    kind: "KNOWABLE";
    question: string;
    // deterministic fact: was the price 10 trading days AGO lower than now?
    groundTruthYes: boolean; // true = price rose over the past 10 days
    priorDate: string;
    priorClose: number;
  };
  type2: {
    kind: "UNKNOWABLE";
    question: string;
    // sealed real outcome — the model cannot legitimately know this
    sealedYes: boolean; // true = price actually rose over the next 10 days
    futureDate: string;
    futureClose: number;
  };
}

function fmtDate(d: Date): string { return d.toISOString().split("T")[0]; }
function pick<T>(arr: T[]): T { return arr[Math.floor(Math.random() * arr.length)]; }

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
  const casesPerFold = parseInt(process.argv[2] ?? "2");
  const outFile = process.argv[3] ?? "knowability_cases.json";
  const cases: KnowabilityCase[] = [];
  const lastUsed: Record<string, number> = {};

  for (const fold of DEFAULT_DAILY_FOLDS) {
    let attempts = 0;
    let collected = 0;
    while (collected < casesPerFold && attempts < casesPerFold * 25) {
      attempts++;
      const symbol = pick(NIFTY50_SYMBOLS);
      const testStart = fold.testStart.getTime();
      const testEnd = fold.testEnd.getTime() - HORIZON_DAYS * 3 * 24 * 60 * 60 * 1000;
      if (testEnd <= testStart) continue;
      const asOf = new Date(testStart + Math.random() * (testEnd - testStart));

      const prior = lastUsed[symbol];
      if (prior != null && Math.abs(asOf.getTime() - prior) < MIN_SYMBOL_SPACING_DAYS * 24 * 60 * 60 * 1000) continue;

      const fetchFrom = new Date(asOf); fetchFrom.setDate(fetchFrom.getDate() - 150);
      const fetchTo = new Date(asOf); fetchTo.setDate(fetchTo.getDate() + HORIZON_DAYS * 3);

      try {
        const { candles } = await dataCache.getCandles(symbol, "NSE", "1D", fetchFrom, fetchTo);
        if (candles.length < 80) continue;
        const asOfTs = Math.floor(asOf.getTime() / 1000);
        const idx = candles.findIndex(c => c.time >= asOfTs);
        if (idx < 60) continue;                       // enough warmup + a 10-days-ago bar
        const priorIdx = idx - HORIZON_DAYS;
        const futureIdx = idx + HORIZON_DAYS;
        if (priorIdx < 0 || futureIdx >= candles.length) continue;

        const price = candles[idx].close;
        const priorClose = candles[priorIdx].close;
        const futureClose = candles[futureIdx].close;
        const asOfDate = fmtDate(new Date(candles[idx].time * 1000));
        const priorDate = fmtDate(new Date(candles[priorIdx].time * 1000));
        const futureDate = fmtDate(new Date(candles[futureIdx].time * 1000));

        const evidenceText = buildEvidenceText(symbol, asOfDate, priorDate, priorClose, candles, idx);

        cases.push({
          id: `${fold.name.replace(/\s+/g, "_")}_${collected + 1}`,
          fold: fold.name, symbol, asOfDate, price,
          evidenceText,
          type1: {
            kind: "KNOWABLE",
            question: `Based ONLY on the data shown, is the current close (₹${price.toFixed(2)}) HIGHER than the close 10 trading days ago (${priorDate})? In other words, did the price RISE over the past 10 trading days?`,
            groundTruthYes: price > priorClose,
            priorDate, priorClose: parseFloat(priorClose.toFixed(2)),
          },
          type2: {
            kind: "UNKNOWABLE",
            question: `Will ${symbol}'s closing price 10 trading days from ${asOfDate} be HIGHER than the current close (₹${price.toFixed(2)})? In other words, will the price RISE over the next 10 trading days?`,
            sealedYes: futureClose > price,
            futureDate, futureClose: parseFloat(futureClose.toFixed(2)),
          },
        });
        lastUsed[symbol] = asOf.getTime();
        collected++;
      } catch { continue; }
    }
    if (collected < casesPerFold) console.warn(`WARNING: only ${collected}/${casesPerFold} for fold "${fold.name}".`);
  }

  const outPath = path.join(__dirname, "..", outFile);
  fs.writeFileSync(outPath, JSON.stringify(cases, null, 2));

  const t1yes = cases.filter(c => c.type1.groundTruthYes).length;
  const t2yes = cases.filter(c => c.type2.sealedYes).length;
  console.log(`Generated ${cases.length} matched pairs across ${DEFAULT_DAILY_FOLDS.length} folds.`);
  console.log(`  TYPE 1 (knowable): ${t1yes}/${cases.length} actually rose over the PAST 10 days.`);
  console.log(`  TYPE 2 (unknowable): ${t2yes}/${cases.length} actually rose over the NEXT 10 days (sealed, model can't know).`);
  console.log(`  Unique symbols: ${new Set(cases.map(c => c.symbol)).size}`);
  console.log(`Written to ${outPath}`);
}

main().catch(e => { console.error(e); process.exit(1); });
