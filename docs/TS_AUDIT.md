# StockAI — Time-Series Audit (Day 1)

**Date:** 2026-07-15 · **Scope:** `predictor.py`, `stock_predictor.py`, `historical.py`, `correlation.py`
**Goal of the upgrade:** turn a price-level LSTM with a hidden data leak and no baseline into an
honestly-evaluated forecasting service — walk-forward, cost-aware, and measured against the trivial
competitors any quant would demand.

---

## TL;DR — what's actually here

The repo is presented as an "LSTM Stock Price Predictor." Under audit it is a **single LSTM that
predicts the absolute next-day close price**, trained fresh on every request, reporting RMSE/MAE/MAPE
on the last 20% of one series — **with a data leak in the normalisation, no baseline, and no backtest.**
The headline metrics look good only because (a) predicting an absolute price that barely moves day-to-day
is easy, and (b) the leak nudged them further. Neither means the model has any edge.

---

## Findings

### 1. Scaler leakage (CRITICAL) — `predictor.py:78–79`, `stock_predictor.py:229–230`
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)   # fit on the FULL series...
...
split = int(len(X) * 0.8)                            # ...THEN split into train/test
```
The scaler is `fit` on the **entire** price series *before* the temporal split. Its `min`/`max` — the
normalisation constants the model trains against — are computed using future (test) prices. The test
window's extremes leak into training. **Fixed on Day 1** (see `DATA_LEAKAGE.md`): fit the scaler on the
train slice only, then transform the rest.
*Note:* the 80/20 split itself is sequential/temporal — that part is fine. **Only the scaler was the bug.**

### 2. No baseline, no backtest (CRITICAL)
The model predicts the **absolute close price**, so the one-line **persistence** competitor
`ŷ_t = y_{t-1}` is a near-unbeatable rival, yet it is never computed. There is no **buy-and-hold**
reference and no **trading backtest** with transaction costs. A reported RMSE with no baseline is
theatre — Day 1 adds persistence + buy-and-hold and the numbers are damning (see results below).

### 3. Volatility-only "confidence" (MISLEADING) — `predictor.py:136–141`
```python
if volatility < 2:  confidence = "High", 85
elif volatility < 4: confidence = "Medium", 70
else:                confidence = "Low", 55
```
Confidence is a hardcoded step-function of realised volatility, **completely disconnected from the
model's actual residuals/RMSE.** A calm stock the model predicts badly still shows "High 85%."
Day 5 replaces this with residual-based / conformal prediction intervals.

### 4. Retrains on every request (COST/LATENCY)
`run_prediction()` fits a new LSTM per call — no persistence of weights, no caching of the model.
Every user pays full training latency. (Cache/serving addressed Days 5, 9.)

### 5. Target framing (DESIGN)
Predicting absolute price hides the fact that the model captures almost no *movement*. The honest task
is next-day **return** / **direction**. Reframed on Day 2.

### 6. `stock_predictor.py` duplicates the leak with a bigger net — lines 229–246
Streamlit twin: LSTM(50)→Dropout→LSTM(50)→Dropout→Dense(1), `time_step=60`, **same scaler leak**,
same sequential split. Fixed in the same Day-1 commit.

### 7. Packaging gaps
`requirements.txt` is missing `yfinance`, `streamlit`, `plotly`, `vaderSentiment` (imported but
unpinned/absent). No `tests/`, `docs/`, `data/`, `results/`, `src/`, `models/`. (Fixed across Days 5, 10.)

### Not bugs (verified sound)
- `historical.py:calculate_technical_indicators` — SMA/EMA/RSI/Bollinger/MACD/Sharpe compute correctly.
- `correlation.py` — Pearson matrix + top-5 pairs are fine.
- The 80/20 split ordering is temporal (no shuffling) — correct.

---

## Model architecture (as-is)

| File | Architecture | time_step | Train | Metrics |
|------|-------------|-----------|-------|---------|
| `predictor.py` | LSTM(32)→Dropout(0.2)→Dense(16)→Dense(1) | 30 | 15 ep / batch 16 / ES patience 3 | RMSE/MAE/MAPE on last 20% |
| `stock_predictor.py` | LSTM(50)→Drop→LSTM(50)→Drop→Dense(1) | 60 | 10 ep / batch 32 | same |

---

## Day-1 honest baseline (10 tickers, 2021-01-01 → 2025-01-01, 195 test days each)

Full numbers in `results/baseline_metrics.json` / `results/phase1_leakage_comparison.csv`.

| Metric (mean over 10 tickers) | Value |
|---|---|
| RMSE — leaky scaler (old, optimistic) | 6.58 |
| **RMSE — leakage-fixed (honest)** | **6.98** |
| RMSE inflation from the leak | **+5.2%** |
| RMSE — **persistence** `ŷ_t=y_{t-1}` baseline | **3.80** |
| MAPE — leakage-fixed LSTM | 2.33% |
| MAPE — persistence | 1.17% |
| Directional accuracy — LSTM | **0.48** |
| Directional accuracy — always-up baseline | 0.55 |
| Tickers where LSTM beats persistence (RMSE) | **0 / 10** |

**Two genuine findings:**
1. Removing the leak raised honest RMSE **~5%** on average — the old number was optimistically biased.
   (Per-ticker the leak's size tracks how far the test window pushed new price extremes; on range-bound
   names like KO it's noise, on trending names like AAPL/NVDA it's +20–23%.)
2. **The leak is the small story.** The big one: the LSTM's honest RMSE (6.98) is **~1.8× worse than a
   one-line persistence baseline (3.80)** on *every* ticker, and its next-day direction (0.48) is below
   a coin flip and below always-up (0.55). A price-level LSTM has **no measurable edge**. The resume
   claim is honest evaluation, not a fake 99%.

---

## Upgrade plan (10 days)

| Day | Deliverable |
|-----|-------------|
| 1 ✅ | Audit, **scaler-leak fix**, persistence + buy-and-hold baselines, honest numbers |
| 2 | Reframe target → returns/direction; expanding **walk-forward** harness |
| 3 | Model bake-off (persistence/ARIMA/Prophet/XGBoost/LSTM) + **cost-aware trading backtest** vs buy-and-hold |
| 4 | Feature depth + SHAP; multi-horizon (1/5/20-day) |
| 5 | `src/` refactor; **residual-based prediction intervals**; FastAPI; fix `requirements.txt` |
| 6 | Optuna sweep + failure-mode analysis by regime/vol/sector |
| 7 | Portfolio backtest (sizing, stops, cost sweep) + transformer TS model (PatchTST/N-BEATS) |
| 8 | Frontier LLM forecasting (negative result) + full ablation |
| 9 | Dockerise, Redis cache, MLflow tracking, Streamlit ops dashboard |
| 10 | Tests (leakage/temporal-split/backtest-cost regressions) + README + demo |
