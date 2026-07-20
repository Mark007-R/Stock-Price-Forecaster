# Day 06 of 10 — Phase 4: Optuna Sweep + Failure-Mode Analysis

**Date:** 2026-07-20 · **Project:** StockAI Production Upgrade · **Phase 4 (Tuning + failure analysis)**

## Resume gap progress

**Gap:** the repo had never asked the two questions that separate a notebook from an evaluation: *"did you tune it properly?"* and *"where exactly does it fail?"* Days 3–4 established that no model beats buy-and-hold on default settings; the standing objection was that defaults are a strawman.

**Today's contribution:** a 40-trial Optuna sweep run the only honest way a time series allows (inner-validation windows carved off each walk-forward fold's *train* slice — outer test blocks untouched during tuning), a three-axis failure-mode map (market regime × volatility bucket × sector) over 5,000 out-of-sample days, and both remaining targeted fixes from the spec (vol-scaled targets, time-decay sample weighting) re-run through the identical walk-forward. The result is the sprint's cleanest evidence yet that the missing ingredient was never hyperparameters.

## Setup

- **Compute:** CPU only; total wall-clock 7.2 min (40 trials × 25 inner fits + 200 outer fits + failure-mode slicing).
- **Data:** cached yfinance closes, 10 tickers (AAPL MSFT SPY GOOGL AMZN META NVDA JPM XOM KO), 2021-01-01 → 2025-01-01, identical to Days 2–5.
- **Splits:** the exact expanding-window folds of Days 2–5 (`src/backtest/walkforward.py`), `assert_no_peeking` + `assert_no_lookahead` re-verified per ticker.
- **Tuning protocol:** objective = mean RMSE(next-day returns) on the last 120 feature rows of each fold's train slice, averaged over 5 tuning tickers (SPY AAPL NVDA JPM KO) × 5 folds. The other 5 tickers never inform tuning. TPE sampler, seed 42, 40 trials, 8-dim search space (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda, reg_alpha).

## Experiments

### Experiment 1 — Can Optuna rescue XGBoost?

**Hypothesis:** if the Day-3/4 failure was a hyperparameter artifact, 40 trials over an 8-dim space should find a config that beats the random walk on inner validation and converts that into outer-fold edge.

**Method:** sweep as above; the random-walk floor (predict 0) and the default config are scored on the *identical* inner windows for reference. Best config then re-evaluated walk-forward on all 10 tickers with the cost-aware trading backtest (5 bps/side).

**Result:**

| Config (inner validation) | RMSE | mean \|prediction\| |
|---|---|---|
| Default params | 0.017959 | 0.005952 |
| Random walk (predict 0) | 0.016558 | 0 |
| **Optuna best (trial #25)** | **0.016504** | **0.001068** |

Best params: `n_estimators=555, max_depth=3, learning_rate=0.0016, subsample=0.62, colsample_bytree=0.50, min_child_weight=15.2, reg_lambda=7.8, reg_alpha=0.05`.

fANOVA importances: **max_depth 35.8% + learning_rate 32.6%** — the two capacity/shrinkage knobs carry 68% of all objective variance; the seven remaining knobs share the rest.

**Interpretation:** given eight knobs and forty trials, the optimizer's best move was to *remove the model*. The winning config has a learning rate 31× smaller than default and a min_child_weight 15× larger; its mean |prediction| is **5.6× smaller** — about 11 bps of daily "conviction" against a typical daily move of ~165 bps. It edges past the zero floor by 0.33% on inner validation, which is the sound of a model asymptotically approaching "predict nothing." Tuning worked exactly as designed; it just discovered that the loss surface's minimum sits at the origin.

### Experiment 2 — Does the tuned config transfer to the outer folds?

**Method:** default vs tuned, walk-forward on all 10 tickers × 5 folds, plus the trading backtest.

**Result:**

| Config | RMSE ratio vs predict-zero | Dir-acc | mean \|pred\| | Sharpe (net) | vs B&H (1.83) |
|---|---|---|---|---|---|
| xgb_default | 1.107 | 0.5164 | 0.0057 | 1.220 | −0.610 |
| xgb_tuned | **1.0009** | 0.5156 | 0.0011 | 0.950 | −0.880 |

**Interpretation:** tuning bought back RMSE parity with the random walk (1.107 → 1.0009) — and *lowered* the trading Sharpe (1.22 → 0.95). A shrunken model still crosses zero, so it still takes positions (exposure rose 0.54 → 0.74 as tiny positive noise predictions multiplied), but with ~11 bps of conviction its long/flat pattern is noise-gating on the market's drift. Better point forecasts, worse portfolio — the Day-3 exposure lesson (Sharpe tracks time-in-market, not accuracy) reproduced from a new direction.

### Experiment 3 — Failure-mode map: regime × volatility × sector

**Hypothesis (from the spec):** directional accuracy collapses somewhere specific — bear regimes, high-vol periods, or particular sectors — and that pocket picks the targeted fix.

**Method:** the Day-3 concatenated OOS predictions (500 days × 10 tickers, all 6 methods) plus today's tuned stream, sliced by (a) regime from the trailing 63-day return computed through the *previous* close (bull > +5%, bear < −5%, else sideways), (b) trailing 21-day vol terciles per ticker, (c) sector. Inside each bucket, the no-skill baseline is the bucket's up-rate (always-up's accuracy *is* the up-rate, confirmed to 3 decimals — the built-in sanity check).

**Result (directional accuracy; edge = acc − bucket up-rate):**

| Slice | Bucket | n days | Up-rate | ARIMA | XGB tuned | ARIMA edge | XGB-tuned edge |
|---|---|---|---|---|---|---|---|
| regime | bear | 473 | **0.596** | 0.559 | 0.528 | −3.7 pp | −6.9 pp |
| regime | bull | 2,986 | 0.533 | 0.528 | 0.514 | −0.6 pp | −2.0 pp |
| regime | sideways | 1,541 | 0.558 | 0.539 | 0.516 | −1.9 pp | −4.2 pp |
| vol | low | 1,670 | 0.546 | 0.537 | 0.524 | −0.9 pp | −2.2 pp |
| vol | mid | 1,660 | 0.529 | 0.517 | 0.496 | −1.2 pp | −3.3 pp |
| vol | high | 1,670 | 0.567 | 0.549 | 0.527 | −1.8 pp | −3.9 pp |
| sector | financials (JPM) | 500 | 0.570 | 0.524 | 0.472 | −4.6 pp | **−9.8 pp** |
| sector | staples (KO) | 500 | 0.518 | 0.526 | 0.518 | **+0.8 pp** | +0.03 pp |

**Interpretation — two findings, both counter to the script:**

1. **The expected failure mode is inverted.** The textbook story is "long-biased models collapse in bear markets." In this OOS window (2023–24), days labeled *bear* by their trailing 63-day return had the **highest** up-rate of any bucket (59.6% vs 53.3% in bull) — trailing-bear days were mostly V-shaped rebounds. So always-up was *safest* precisely where intuition says it's most exposed. Any fix keyed on "detect the bear regime and de-risk" would have made things worse — consistent with Day 4, where regime *features* hurt.
2. **There is no pocket of skill for a fix to amplify.** Model edge is negative in **every** regime bucket and **every** vol bucket. The only non-negative sector readings (staples +0.8 pp, energy +0.4 pp, ARIMA) are 500-day samples — well inside noise. The failure is unconditional; the aggregate numbers weren't hiding a regime where the models secretly work.

### Experiment 4 — Targeted fixes: vol-scaled targets and time-decay weighting

**Method:** the two spec-listed fixes Day 4 didn't already rule out, applied to the tuned config, full walk-forward + backtest. Vol-scaling divides the training target by trailing 21-day vol (both known at row t) and rescales predictions by the forecast row's own trailing vol; time-decay weights training samples with a 126-day half-life.

**Result:**

| Config | RMSE ratio | Dir-acc | Sharpe (net) | Trades | vs B&H |
|---|---|---|---|---|---|
| xgb_tuned | 1.0009 | 0.5156 | 0.950 | 88 | −0.880 |
| xgb_tuned_volscale | 1.0011 | 0.5146 | 1.083 | 96 | −0.747 |
| **xgb_tuned_decay** | 1.0023 | **0.5313** | 1.186 | 60 | −0.644 |
| *(always-up baseline)* | — | *0.5491* | *1.830* | *1* | *0* |

**Interpretation:** time-decay weighting produced the best directional accuracy of any XGB configuration in the sprint (+1.5 pp over default, +1.6 pp over plain tuned) — weak evidence that whatever faint structure exists is *recent* structure, which a 3-year uniform training window dilutes. It is still 1.8 pp below the one-line always-up baseline and 0.64 Sharpe below buy-and-hold. Vol-scaling was RMSE-neutral: the homoskedastic target is statistically cleaner but there is no signal for it to clean.

## Head-to-Head Comparison (running leaderboard, net of 5 bps/side)

| Method | Dir-acc | Sharpe | vs B&H |
|---|---|---|---|
| buy-and-hold / always-up | 0.549 | **1.830** | 0 |
| ARIMA (Day-3 champion) | 0.534 | 1.340 | −0.49 |
| xgb_default (Day 3) | 0.516 | 1.220 | −0.61 |
| **xgb_tuned_decay (Day 6)** | **0.531** | 1.186 | −0.64 |
| xgb_tuned_volscale (Day 6) | 0.515 | 1.083 | −0.75 |
| xgb_tuned (Day 6) | 0.516 | 0.950 | −0.88 |
| LSTM-returns (Day 3) | 0.519 | 1.108 | −0.72 |

## Key Findings

1. **Given 8 knobs and 40 trials, the optimizer chose to remove the model.** 68% of the objective variance sat in the two shrinkage knobs (max_depth, learning_rate), and the optimum is the shrink-everything corner: 5.6× smaller predictions, RMSE at random-walk parity. Optuna is a search procedure, not a signal source — on a signal-free task, its honest optimum *is* the random walk.
2. **Better forecasts, worse portfolio.** The tuned model improved RMSE ratio from 1.107 to 1.0009 and *lost* 0.27 Sharpe, because near-zero predictions still cross zero and gate exposure on noise. Point-forecast quality and trading quality are different objectives, and improving one degraded the other.
3. **The failure-mode map came back empty — and inverted.** No regime, vol bucket, or sector shows positive model edge beyond noise; and "bear" days (trailing definition) had the *highest* up-rate of the sample, so the intuitive de-risk-in-bear fix is exactly backwards for this window.
4. **Time-decay weighting is the only fix that moved anything** (+1.5 pp dir-acc, best XGB of the sprint) — and it still loses to a baseline with one line of code and one trade.

## What Didn't Work (and WHY)

- **Tuning as rescue:** the search space contained no config with outer-fold edge because the features carry no exploitable signal (established by SHAP on Day 4); tuning can only re-weight what the features know.
- **Vol-scaled targets:** statistically well-motivated (homoskedastic residuals), practically inert — variance normalization can't create predictable mean structure.
- **Regime-conditioned interpretation:** the trailing bear label selects post-drawdown days, which in 2023–24 were rebound days. A regime fix conditioned on that label would have shorted the market's best stretch.

## Sample Outputs Saved

- `results/phase4_optuna_trials.csv` (40 trials, params + inner RMSE + shrinkage telemetry)
- `results/phase4_leaderboard.csv`, `results/phase4_configs_folds.csv` (200 rows), `results/phase4_configs_trading.csv`
- `results/phase4_failure_modes.csv` + `results/samples/day06_failure_mode_table.csv` (13 buckets × 7 methods)
- `results/samples/day06_{SPY,NVDA,KO}_tuned_predictions.csv` (full 500-day OOS streams)
- `results/plots/day06_{optuna_history,param_importance,tuned_vs_default,failure_regime,failure_vol,failure_sector,pred_shrinkage}.png`
- `results/metrics.json` → `day06` block

## Code Changes

- `experiments/day06_optuna_failure.py` (new, ~530 lines): sweep + failure-mode analysis + fixes, all through the shared `src/` harness.
- `src/models/xgb.py`: added `TUNED_PARAMS` (Optuna trial #25), `DECAY_HALF_LIFE`, refactored `_fit_predict` with optional time-decay weights, new `predict_fold_tuned` — default path byte-identical in behavior.
- `src/models/__init__.py`: registered `xgboost_tuned` (making it available to the Day-5 FastAPI `/predict` and `/backtest` endpoints), with an honest one-line description.

## Next Day

Day 7 — Phase 5: multi-ticker **portfolio** backtest (position sizing, stop-loss risk management, transaction-cost sensitivity sweep; Sharpe/Sortino/max-DD vs buy-and-hold across regimes) + a transformer time-series model (PatchTST or N-BEATS) against the Day-3 champion on the identical walk-forward. **[POST day]**
