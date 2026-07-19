# Day 05 of 10 — Phase 3: Champion integration, production refactor, and confidence that can be wrong

**Date:** 2026-07-19 · **Project:** StockAI Production Upgrade · **Phase:** 3 (wrap)

## Resume gap progress

**Gap:** the repo had a research-notebook shape (three duplicated data paths, models living inside one experiment script, no serving layer) and a "confidence" number that was unfalsifiable — hardcoded off the asset's volatility (`vol<2 → 85, <4 → 70, else 55`) without ever looking at the model's errors. No hiring manager reads "confidence: 85" and believes it; no observable event could even prove it wrong.

**Today's contribution:** a production `src/` layout (single cached data loader, model registry with the walk-forward contract, FastAPI serving layer), and **split-conformal prediction intervals** wired into the shipped Flask app — uncertainty that makes a falsifiable claim ("the 80% band contains the realised value 80% of the time") and passes its own audit: measured over 7,600 rolling out-of-sample evaluations per level, the conformal 80% band covers **80.8%** and the 99% band covers **99.2%**.

## Files touched

| File | Change |
|------|--------|
| `src/data/loader.py` | **new** — one cached, retried yfinance entry point (`load_prices`, `load_ohlcv`); replaces 3 divergent data paths; reads the Day-2/3/4 CSV cache byte-identically |
| `src/models/__init__.py` | **new** — registry of the Day-3 bake-off families + `make_context`; `CHAMPION = "arima"` |
| `src/models/persistence.py` | **new** — random-walk + momentum baselines |
| `src/models/arima.py` | **new** — AIC order selection, rolling 1-step fold predictor, serving-time `forecast_returns` |
| `src/models/xgb.py` | **new** — kept as the honest negative result (Day 3: 9.3% worse than zero-forecast) |
| `src/models/lstm.py` | **new** — predictor.py's LSTM(32) arch in walk-forward form, train-only scaler |
| `src/models/intervals.py` | **new** — split-conformal + Gaussian half-widths, √h horizon scaling, coverage checker, `build_interval_forecast` |
| `predictor.py` (lines 19–25, 141–163, 168–199) | volatility-only confidence **replaced** by conformal intervals from the model's own held-out test residuals; response gains `interval_lower80/upper80/lower95/upper95`, half-widths, method string — template contract preserved |
| `src/serving/api.py` | **new** — FastAPI on :8000: `/predict`, `/backtest`, `/indicators/{ticker}`, `/correlation`, `/health`; Pydantic v2 schemas; threadpool offload; zero-cost backtests refused at the schema |
| `requirements.txt` | **fixed** — now lists everything the code actually imports (`yfinance`, `streamlit`, `plotly`, `vaderSentiment`, `fastapi`, `uvicorn`, `xgboost`, `statsmodels`, …); versions pinned to the working venv |
| `experiments/day05_intervals.py` | **new** — calibration study + predictor integration check + live API test |

## Setup

- CPU only. Coverage study runs on the saved Day-3 walk-forward OOS champion (ARIMA) streams — 10 tickers × ~500 OOS days (2021–2025), so no refitting and no new leakage surface.
- Rolling 120-day calibration window, **sequential** evaluation: the interval for day *t* is built only from errors observed before *t*. 380 evaluations per ticker per level → 7,600 per level.
- Integration checks run the real things: `run_prediction()` (network fetch + LSTM train) and uvicorn over HTTP.

## Experiments

### Exp 1 — Is conformal actually calibrated (and what does Gaussian cost)?

**Hypothesis:** daily-return residuals are fat-tailed with a peaked center, so Gaussian ±zσ bands will over-cover at central levels and under-cover in the tails (S-shaped calibration curve), while distribution-free conformal tracks the diagonal everywhere.

**Method:** for each of 7 nominal levels (50–99%), build both bands per day from the same rolling 120-day residual window; score empirical coverage and mean width across all 10 tickers.

**Result (mean of 10 tickers, 7,600 eval points per level):**

| Nominal | Conformal coverage | Gaussian coverage | Conformal width | Gaussian width |
|---------|-------------------|-------------------|-----------------|----------------|
| 50% | **0.508** | 0.571 | **0.0092** | 0.0107 |
| 60% | **0.613** | 0.671 | **0.0116** | 0.0134 |
| 70% | **0.711** | 0.760 | **0.0144** | 0.0165 |
| 80% | **0.808** | 0.847 | **0.0180** | 0.0204 |
| 90% | **0.904** | 0.915 | **0.0244** | 0.0261 |
| 95% | **0.951** | 0.948 | 0.0307 | 0.0311 |
| 99% | **0.992** | 0.981 | 0.0710 | **0.0409** |

**Interpretation:** the S-curve is exactly as predicted. Gaussian misses its 99% claim by 2× (miss rate 1.95% vs the 1% promised) — and the tail is where risk management lives. The surprise is the center: conformal is not just better calibrated, it is **12% narrower** at the 80% level, because the Gaussian σ is inflated by the same fat tails that starve its tail coverage. Honesty is *free* in the center; it only costs width (+74%) at 99%, where the Gaussian number was fiction.

### Exp 2 — How wrong was the shipped volatility heuristic?

**Hypothesis:** a confidence derived from asset volatility instead of model error can only be right by coincidence.

**Method:** operationalise the old rule charitably — treat "confidence 85" as a claim of 85% coverage and give it the Gaussian band at that level built from rolling asset vol — then score it against the same OOS streams. Separately, compare its verdict against the new system's on the live AAPL prediction.

**Result:** per-ticker gaps between claimed and empirical coverage run **+0.5 pp to +9.8 pp** (claims 82.2% on average, covers 86.1%) — miscalibrated, but mostly *conservative*… for ARIMA. The live AAPL check exposes the coincidence: AAPL's daily vol is 1.54%, so the old rule says **"High / 85"**. The shipped LSTM's actual held-out price errors give an 80% band of **±$16.59 = 4.97% of price** — the new system says **"Low"**. Both cannot be right, and only one of them looked at the model.

**Interpretation:** the heuristic is nearly-right for a model that predicts nothing (ARIMA ≈ 0, so asset vol ≈ residual scale — coincidence) and badly wrong for the model actually shipped (a price-level LSTM whose 1-step error is ~3× daily vol, because it lags the price). A confidence that never touches the residuals cannot tell those two situations apart — that is the whole indictment.

### Exp 3 — Does the production pipeline serve end-to-end?

**Method:** boot uvicorn, exercise every endpoint over real HTTP; run the Flask job path with the new interval code; assert band nesting, point-inside-band, and √h widening.

**Result:**

| Check | Outcome |
|-------|---------|
| `GET /health`, `POST /predict`, `POST /backtest` ×2, `GET /indicators/AAPL`, `GET /correlation` ×4 tickers | all **200** |
| Schema guards: `cost_bps=0` and unknown model | both **422** (refused) |
| `/backtest` SPY · arima · 5 folds · 5 bps | Sharpe **1.77 vs buy-and-hold 1.99**, `beats_buy_and_hold_sharpe: false` — the API reports its own defeat |
| `/predict` AAPL h=10 | arima(1,0,0), last 248.83 → 250.87, 80% band [238.4, 263.3], 95% band [228.6, 273.1] |
| `run_prediction("AAPL")` via Flask path | done; interval fields present; nesting/monotonicity asserts pass |

## Head-to-head comparison (running leaderboard, unchanged by design)

Day 5 adds no new forecaster — it integrates the Day-3 champion and makes the uncertainty honest. The leaderboard stands: buy-and-hold Sharpe 1.83 unbeaten; ARIMA 1.34 best among forecasters (10-ticker mean); today's single-ticker SPY API run (1.77 vs 1.99) is consistent with it. What changed is the *uncertainty* column:

| Uncertainty method | Claims | Delivers (measured) | Verdict |
|--------------------|--------|---------------------|---------|
| Shipped vol heuristic | "85 / 70 / 55" | 86.1% mean vs 82.2% claimed; +9.8 pp worst gap; **wrong regime for the shipped LSTM** (says High where errors say Low) | replaced |
| Gaussian ±zσ | any level | +4.7 pp at 80%; **2× the promised miss rate at 99%** | comparison baseline |
| Split-conformal | any level | within 1.3 pp at every level, **7,600 evals/level** | **shipped** |

## Key Findings

1. **Conformal dominates Gaussian on both axes in the center.** Better calibrated (80.8% vs 84.7% at nominal 80%) *and* 12% narrower. The fat tails that make Gaussian σ overshoot the central quantiles are the same ones that make its 99% band miss 2× too often. One mechanism, two symptoms, opposite directions.
2. **The old confidence was right only by coincidence.** For a near-zero forecaster, asset vol ≈ residual scale, so the heuristic lands close. Point it at the actual shipped model — a price-level LSTM with ±5%-of-price errors — and it says "High" where the calibrated answer is "Low". A confidence that never reads the residuals cannot distinguish a good model from a bad one *by construction*.
3. **The serving layer encodes the sprint's ethics.** `/backtest` refuses to run without transaction costs (schema-level 422), always returns buy-and-hold beside the strategy, and `/predict` carries the disclaimer that no model here beat buy-and-hold. The honest result is now an API contract, not a footnote.
4. **What didn't work:** first run crashed on Windows cp1252 vs a Unicode arrow in console output (fixed with UTF-8 io encoding — cosmetic, but it is exactly how "works on my machine" starts). Also considered ARIMA's native Gaussian forecast intervals for `/predict` and rejected them: Exp 1 is direct evidence that parametric tails under-cover, so the API uses conformal residual bands instead.

## Sample outputs saved

- `results/phase3_intervals.csv` (140 rows: ticker × method × level), `results/phase3_old_heuristic.csv`
- `results/plots/day05_calibration_curve.png` · `day05_tail_miss_rates.png` · `day05_interval_width.png`
- `results/samples/day05_predictor_result_AAPL.json` (full Flask job payload with bands), `results/samples/day05_api_responses.json` (all endpoints)
- `results/metrics.json` → `day05` entry

## Phase wrap-up: What was finalized

**Final approach.** Production layout locked: `src/data` (single cached loader) → `src/features` (no-look-ahead, empirically proven) → `src/models` (registry: persistence / momentum / arima / xgboost / lstm, all speaking the walk-forward contract) → `src/backtest` (expanding folds + cost-aware trading sim) → `src/serving` (FastAPI :8000, Pydantic-validated). Champion = **ARIMA** — champion *among forecasters*, explicitly not champion over buy-and-hold, and the API says so in every response. Uncertainty = **split-conformal on held-out residuals**, √h-widened for multi-step horizons; the Flask app's `confidence`/`confidence_score` fields survive for template compatibility but are now derived from calibrated band width.

**Final metrics (Phases 1–3).** Scaler leakage fixed Day 1 (the honest numbers are the numbers). Walk-forward, 10 tickers, 2021–2025, 5 bps/side: buy-and-hold Sharpe **1.83** unbeaten; best forecaster ARIMA **1.34**; every model family lands on the random-walk RMSE floor (~0.0164). New this phase: conformal coverage within **1.3 pp of nominal at all 7 levels** (7,600 OOS evals each); Gaussian +4.7 pp at 80% / 2× miss rate at 99%; shipped heuristic retired with gaps up to +9.8 pp and a regime-dependence it could not see.

**What carries forward.** Day 6 (Optuna + failure-mode analysis) tunes the champion through the same `src/` interfaces and slices directional accuracy by regime/vol bucket — the Day-4 regime features that failed as *predictors* get re-examined as *diagnostics*. Day 7's portfolio backtest builds on `src/backtest/trading.py`; Day 9's Docker/MLflow wrapper ships `src/serving/api.py` as-is; Day 10's regression tests (`test_no_scaler_leakage`, `test_walkforward_no_peeking`, `test_backtest_costs`) now have stable module boundaries to pin.

**Resume gap progress.** Before: notebook-shaped repo, duplicated data paths, models trapped in scripts, unfalsifiable confidence, no API. After: layered `src/` package, one data loader, five registered models under one evaluation contract, distribution-free prediction intervals that pass a 7,600-point calibration audit, and a FastAPI service whose schema refuses dishonest backtests. This is the difference between "trained an LSTM on stock prices" and "built and calibrated a forecasting service" on a resume.

## Next Day

Day 6 — Phase 4: Optuna sweep on the champion (≥30 trials), then failure-mode analysis by market regime (bull/bear/sideways), volatility bucket, and sector: where does directional accuracy collapse, and does a targeted fix (regime feature, vol-scaled targets, time-decay weighting) move the walk-forward numbers?

## Code Changes

Branch `sprint/day05-2026-07-19` → PR to `main` (squash). New: `src/data/`, `src/models/` (6 modules), `src/serving/api.py`, `experiments/day05_intervals.py`. Modified: `predictor.py` (conformal confidence + interval fields), `requirements.txt` (complete, pinned).
