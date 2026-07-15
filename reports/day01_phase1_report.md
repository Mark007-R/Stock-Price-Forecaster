# StockAI Production Upgrade — Day 1 / 10
## Phase 1: Audit + Leakage Fix + Honest Baseline

**Date:** 2026-07-15 · **Field:** Time-Series ML / Quant Backtesting

---

### Resume gap progress
**Gap:** the project ships an LSTM "stock price predictor" with a silent data leak, no baseline, and no
backtest — an impressive-looking RMSE that means nothing.
**Today's contribution:** documented the leak, **fixed it**, and stood up the trivial baselines
(persistence + buy-and-hold) that every quant demands — then showed the honest numbers. The leak was
inflating RMSE ~5%; the far bigger finding is that the LSTM loses to a one-line persistence baseline on
all 10 tickers. The resume claim becomes *honest evaluation*, not a fake accuracy.

---

### Files touched
| File | Change |
|------|--------|
| `predictor.py` (was L78–79) | Scaler now fit on the train slice only (`scaler.fit(close[:raw_split])`) |
| `stock_predictor.py` (was L229–230) | Same leak fix in the Streamlit twin |
| `docs/TS_AUDIT.md` | New — full component audit + upgrade plan |
| `docs/DATA_LEAKAGE.md` | New — the bug, the fix, measured impact, regression-guard plan |
| `experiments/day01_baseline.py` | New — reproducible leaky-vs-fixed + baseline harness |
| `results/baseline_metrics.json`, `results/phase1_leakage_comparison.csv` | New — honest metrics |
| `results/plots/*.png`, `results/samples/*.csv` | New — charts + per-ticker sample predictions |

### Setup
- **Compute:** CPU (TensorFlow 2.15, CPU build). 20 LSTM trainings (leaky+fixed × 10 tickers) in ~4 min.
- **Data:** public yfinance daily closes, 10 tickers `AAPL MSFT SPY GOOGL AMZN META NVDA JPM XOM KO`,
  fixed range **2021-01-01 → 2025-01-01** (195 test days per ticker). No fabricated data.
- **Model:** `LSTM(32)→Dropout(0.2)→Dense(16)→Dense(1)`, time_step 30 — identical to `predictor.py`.

---

### Experiment 1 — Scaler leakage: before vs after

**Hypothesis:** fitting `MinMaxScaler` on the full series before the split leaks the test min/max and
inflates RMSE. Fixing it should *raise* the honest error.

**Method:** run the identical LSTM pipeline twice per ticker — (A) leaky `fit_transform(full)`,
(B) fixed `fit(train)`+`transform(all)` — same seed, same split, measure test RMSE in price space.

| Mean over 10 tickers | Leaky (old) | Fixed (honest) | Δ |
|---|---|---|---|
| Test RMSE ($) | 6.58 | **6.98** | **+5.2%** |

Per-ticker inflation: AAPL +23%, NVDA +22%, GOOGL +16%, SPY +13%, MSFT +5%; ~0 / small-negative on
range-bound KO, XOM, JPM, AMZN (noise from CPU non-determinism).

**Interpretation:** the leak is real and biggest on trending names where the test window sets new
extremes — exactly where honesty matters most. But ~5% is a *modest* inflation. The leak alone was never
the reason the model "looked good."

### Experiment 2 — Honest LSTM vs trivial baselines

**Hypothesis:** because the model predicts an absolute price, a persistence baseline `ŷ_t = y_{t-1}`
will be hard to beat, and directional accuracy will be ~coin-flip.

**Method:** on the same 195-day test window, compute persistence RMSE/MAPE, buy-and-hold return, and
next-day directional accuracy for the fixed LSTM vs an always-up baseline.

| Mean over 10 tickers | Value | Verdict |
|---|---|---|
| RMSE — LSTM (fixed, honest) | 6.98 | — |
| RMSE — **persistence** `ŷ_t=y_{t-1}` | **3.80** | **beats LSTM on 10/10** |
| MAPE — LSTM | 2.33% | — |
| MAPE — persistence | 1.17% | ~2× better |
| Directional accuracy — LSTM | **0.48** | below coin flip |
| Directional accuracy — always-up | 0.55 | beats the LSTM |
| Buy-and-hold return (test window) | +20.5% | context reference |

**Interpretation:** the LSTM's honest RMSE is **~1.8× worse than doing nothing** (persistence), on every
single ticker, and it has **no directional edge** (0.48 < 0.50 < 0.55). The visually-tight
"predicted vs actual" curve is an artifact of predicting a slow-moving absolute price — persistence
produces an even tighter one. This is the audit finding in one line.

---

### Head-to-Head leaderboard (Day 1 — price-level, in-sample-style holdout)
| Rank | Method | Mean RMSE ($) | Mean MAPE | Dir. acc | Note |
|------|--------|---------------|-----------|----------|------|
| 1 | Persistence `ŷ_t=y_{t-1}` | **3.80** | **1.17%** | n/a (flat) | trivial, unbeaten |
| 2 | LSTM(32) — leakage-fixed | 6.98 | 2.33% | 0.48 | our honest model |
| — | LSTM(32) — leaky (retired) | 6.58 | — | 0.48 | ✗ inflated, never reported as "ours" |
| ref | Always-up directional | — | — | 0.55 | directional reference |
| ref | Buy-and-hold | — | — | — | +20.5% over test window |

*(Walk-forward folds replace this single-holdout view on Day 2; the trading backtest with costs lands Day 3.)*

---

### Key findings
1. **The scaler leak inflated RMSE ~5% on average** (up to +23% on trending tickers) — the old numbers
   were optimistically biased and are now retired.
2. **The leak was the small story.** The fixed LSTM (RMSE 6.98) loses to a one-line persistence baseline
   (3.80) on **all 10 tickers**, with directional accuracy 0.48 — *below a coin flip.* A price-level
   LSTM has no measurable edge.
3. Predicting an absolute, slow-moving price makes any model *look* accurate. The honest task is
   **returns / direction**, where the edge (if any) actually lives — that's Day 2.

### What didn't work (and why)
- **The LSTM as a price predictor.** Not a training bug — a framing one. Absolute-price targets are
  dominated by the previous price, so a network can't add value over persistence. Fixing requires
  changing the *target*, not the *architecture* (Day 2).

### Sample outputs saved
- `results/baseline_metrics.json` — aggregate + per-ticker honest metrics
- `results/phase1_leakage_comparison.csv` — 10-ticker leaky-vs-fixed-vs-persistence table
- `results/plots/leakage_rmse_before_after.png`, `lstm_vs_persistence_rmse.png`, `directional_accuracy.png`
- `results/samples/<TICKER>_test_predictions.csv` — last-60-day predictions per ticker (all 4 methods)

### Next day (Day 2)
Reframe the target from absolute price → **next-day return / direction**; build the expanding
**walk-forward** backtest harness in `src/backtest/walkforward.py` (refit per window, no peeking); report
RMSE(returns) + directional accuracy across folds vs persistence.

### Code changes
First fix commit (per the hard rule) was the scaler-leak fix alone:
`sprint: fix scaler leakage in predictor.py + stock_predictor.py … scaler now fit on train slice only (Day 1 Phase 1)`.
