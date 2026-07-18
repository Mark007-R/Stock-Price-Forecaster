# StockAI Production Upgrade — Day 4 / 10
## Phase 2c: Feature-engineering depth + multi-horizon forecasting

**Date:** 2026-07-18 · **Project:** StockAI (Time-Series ML / Quant Backtesting) · **Field:** feature engineering, model interpretability, multi-horizon evaluation

---

### Resume gap progress

**Gap:** the shipped project predicts an absolute price with a leaky scaler and a
volatility-only "confidence" number — no honest evaluation, no baseline, no
attribution of *why* the model predicts what it does. Days 1–3 fixed the leakage,
reframed to returns, and showed walk-forward that no model beats buy-and-hold net
of costs. The natural pushback on that result is *"you used the wrong features and
the wrong horizon."*

**Today's contribution:** I built the two features a skeptic would demand —
calendar seasonality and market-regime context — and re-scored the champion
learner (XGBoost) at **1-, 5- and 20-day** horizons under the same no-peeking
walk-forward harness. Then I swapped Day 3's *gain* importance for **SHAP**, a
per-prediction attribution, to test whether any feature carries out-of-sample
signal. Both objections fail, and SHAP explains *why* more cleanly than gain
could. This closes the "did you actually try to make it work?" question with a
measured, interpretable no.

---

### Files touched

| File | Lines | Change |
|------|-------|--------|
| `src/features/engineer.py` | +~95 (constants, `build_feature_frame` signature, `_add_extended_features`, `assert_no_lookahead`) | Added `CALENDAR_COLS` (5) + `REGIME_COLS` (6), `EXTENDED_FEATURE_COLS` (29), `WARMUP_EXT=260`, multi-horizon targets `target_h{1,5,20}`. Backward-compatible: default `extended=False, horizons=(1,)` reproduces the exact Day 1–3 output. `assert_no_lookahead` generalised to prove causality of the extended set too. |
| `experiments/day04_features.py` | new, 380 | Walk-forward XGBoost: base-vs-extended ablation @ h=1; multi-horizon 1/5/20; SHAP + gain importance; horizon-matched baselines (random walk on RMSE, always-up on direction). |
| `results/phase2c_features.csv` | new (200 rows) | per (ticker × fold × config) scores. |
| `results/phase2c_summary.csv`, `results/phase2c_shap_importance.csv` | new | per-config aggregate; SHAP-vs-gain importance table. |
| `results/plots/day04_*.png` (×5), `results/samples/day04_*` (×4) | new | ablation, horizon dir-acc, horizon edge, SHAP importance, SHAP-vs-gain scatter; OOS prediction samples for SPY/NVDA/KO + top-15 SHAP. |
| `results/metrics.json` | +`day04` block | appended. |

---

### Setup

- **Compute:** CPU. Full run (10 tickers × 5 folds × 4 configs + SHAP) = **0.7 min**.
- **Data:** public yfinance daily closes, 2021-01-01 → 2025-01-01, cached from Day 3 (no live calls). 10 tickers: AAPL MSFT SPY GOOGL AMZN META NVDA JPM XOM KO.
- **Features:** *base* = 18 price-derived (lagged returns, SMA/EMA ratios, RSI, %B, MACD, trailing vol/drift). *Extended* = base + **calendar** (cyclic day-of-week, cyclic month, turn-of-month) + **regime** (`trend_up`, `vol_regime_high`, `drawdown_63`, `dist_high_252`, `mom_63`, `vol_63`). All trailing/deterministic; causality proven per ticker via `assert_no_lookahead(extended=True)`.
- **Model:** XGBoost, **identical hyper-parameters to Day 3** — the comparison isolates *features* and *horizon*, not a retune.
- **Split:** expanding-window walk-forward, refit per fold, `test_start == train_end` (no peeking). Multi-horizon train rows are cut at `t ≤ train_end-1-h` so no training target lands in the test span.

---

### Experiment 1 — Feature-set ablation @ h=1: does calendar + regime help?

**Hypothesis:** the base set lacked calendar/regime context; adding it should lift directional edge over the always-up baseline and/or push RMSE below the random-walk floor.

| Feature set | # feat | RMSE / random-walk | Dir. acc | Always-up baseline | **Edge (pp)** | Tickers w/ +edge |
|-------------|-------:|-------------------:|---------:|-------------------:|--------------:|-----------------:|
| base        | 18     | 1.107              | 51.6%    | 54.9%              | **−3.3**      | 2 / 10 |
| **extended**| 29     | **1.305**          | 49.1%    | 54.9%              | **−5.8**      | **1 / 10** |

**Result:** adding the 11 new features made the model **worse on both axes** — RMSE-vs-random-walk rose 1.11 → 1.31, and directional edge fell −3.3 → −5.8 pp. Neither set beats the random-walk RMSE floor (ratio > 1), and the extended set beats the always-up baseline on only **1 of 10** tickers.

**Interpretation:** the failure on Day 3 was not a missing *kind* of feature. More features simply gave the trees more noise to overfit. Objection (1) fails.

---

### Experiment 2 — Multi-horizon: does the edge decay or strengthen?

**Hypothesis:** daily direction is noise, but at 5- or 20-day horizons drift/mean-reversion might give the model something to grip.

| Horizon | RMSE / random-walk | Model dir. acc | Always-up baseline | **Edge (pp)** | Tickers w/ +edge |
|--------:|-------------------:|---------------:|-------------------:|--------------:|-----------------:|
| 1-day   | 1.305              | 49.1%          | 54.9%              | **−5.8**      | 1 / 10 |
| 5-day   | 1.407              | 46.0%          | 60.0%              | **−13.9**     | 1 / 10 |
| 20-day  | 1.388              | 46.8%          | 68.2%              | **−21.5**     | 2 / 10 |

**Result:** the edge **decays** — it gets steadily *more negative* with horizon. The reason is the trap that sinks most "our model is 68% accurate!" claims: the **always-up baseline rises with horizon** (54.9% → 60.0% → 68.2%) purely because equities drift up, so a longer horizon means a *higher* bar, not an easier one. The model's raw accuracy stays stuck near coin-flip while the honest benchmark climbs away from it. RMSE never beats the random walk at any horizon.

**Interpretation:** longer horizons don't unlock signal; they just make the drift-only baseline harder to beat. Objection (2) fails. **A model reporting 46.8% "would look like a respectable 20-day forecaster to anyone who forgot the market was up 68% of those windows anyway."**

---

### Experiment 3 — SHAP vs gain: why the model has nothing to rank

**Hypothesis:** if a feature carried real signal, SHAP (per-prediction attribution) and gain (split-quality) would agree on its importance and SHAP magnitudes would be non-trivial.

| | SHAP top-5 (extended, h=1) | gain rank of that feature |
|--|--|--|
| 1 | `vol_63` (regime) | 3 |
| 2 | `mom_63` (regime) | 2 |
| 3 | `ret_lag_1` (base) | **24** |
| 4 | `vol_21` (base) | 12 |
| 5 | `drawdown_63` (regime) | **1** |

- **SHAP ↔ gain rank correlation = 0.33** — weak. `ret_lag_1` is SHAP-rank 3 but gain-rank 24; gain's #1 (`drawdown_63`) is only SHAP-rank 5. The two instruments substantially disagree.
- **SHAP attribution share by group:** base **63.6%**, regime **29.7%**, calendar **6.8%**. The regime features absorb ~30% of attribution — *the model leans on them* — yet the extended model performs **worse** than base.

**Interpretation:** this is the cleaner diagnosis SHAP was brought in for. When a model has real signal, gain and SHAP agree; here they don't (ρ≈0.33), the fingerprint of trees splitting on noise. And the model devoting 30% of its attribution to features that *reduce* out-of-sample accuracy is textbook overfitting — it is confidently explaining noise. Day 3's flat gain said "nothing stands out"; SHAP says "what it does lean on actively hurts."

---

### Head-to-head comparison (running leaderboard, honest walk-forward)

| Day | Question | Verdict |
|-----|----------|---------|
| 1 | leaky vs honest scaler | leakage inflated the score; honest LSTM loses to persistence on 10/10 tickers |
| 2 | price vs returns target; walk-forward | LSTM-returns RMSE ties the random walk (0.0164); dir-acc 0.519 < always-up 0.549 |
| 3 | 5 model families vs buy-and-hold, net of costs | none beats B&H Sharpe (best ARIMA 1.34 vs 1.83); Sharpe corr 0.971 with *exposure*, not accuracy |
| **4** | **richer features? longer horizon?** | **both make it worse: edge −3.3→−5.8 pp (base→ext); −5.8→−21.5 pp (h=1→h=20). SHAP↔gain ρ=0.33** |

---

### Key findings

1. **Better features didn't help — they hurt.** Calendar + regime context pushed the h=1 directional edge from −3.3 to −5.8 pp and RMSE-ratio from 1.11 to 1.31. The Day 3 failure was never a feature-*kind* gap.
2. **The edge decays with horizon because the baseline rises with drift.** Model accuracy at 20 days (46.8%) *looks* forecaster-ish until you set it against the always-up baseline (68.2%). Reporting raw accuracy without the drift-matched baseline is exactly how a leaky "20-day predictor" gets sold.
3. **SHAP is a sharper noise-detector than gain.** Rank correlation of only 0.33 between the two instruments, plus 30% of attribution flowing to features that lower OOS accuracy, is the interpretability signature of a model with no signal to distribute.

### What didn't work (and why)

- **Calendar features** (`turn_of_month`, day-of-week) — 6.8% of SHAP share, no edge. Any calendar anomaly in liquid US equities over 2021–24 is far smaller than daily noise and is arbitraged; the tree can't separate it from variance.
- **Regime features** — absorbed 30% of attribution but *worsened* OOS accuracy. They're informative about *state* (vol regime, drawdown) but state doesn't predict next-period *direction*; the model overfit them.
- **Longer horizons** — did not surface mean-reversion/drift signal; only raised the honest baseline.

### Sample outputs saved

`results/phase2c_features.csv` (200 rows) · `results/phase2c_summary.csv` · `results/phase2c_shap_importance.csv` · `results/plots/day04_{featureset_ablation,horizon_diracc,horizon_edge,shap_importance,shap_vs_gain}.png` · `results/samples/day04_{SPY,NVDA,KO}_ext_h1_predictions.csv` + `day04_top15_shap_importance.csv` · `results/metrics.json` (`day04`).

### Next day

**Day 5 (Phase 3 — champion integration + refactor + confidence calibration, PHASE-WRAP + POST):** build the `src/` production layout (`data/loader.py`, `features/engineer.py`, `models/{persistence,arima,xgb,lstm}.py`, `backtest/walkforward.py`, `serving/api.py`), **replace the volatility-only confidence in `predictor.py` (lines 136–141) with residual-based prediction intervals (conformal/quantile)**, fix `requirements.txt` (add `yfinance`, `streamlit`, `plotly`, `vaderSentiment`, and now `xgboost`, `statsmodels`, `prophet`, `shap`), and stand up the FastAPI `api.py` (`/predict`, `/backtest`, `/indicators`, `/correlation`).

### Code changes

- `src/features/engineer.py`: extended feature builder + multi-horizon targets, backward-compatible, with the no-look-ahead proof extended to the new columns.
- `experiments/day04_features.py`: the full ablation + multi-horizon + SHAP run.
