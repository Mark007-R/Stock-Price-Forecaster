# Day 07 of 10 — Phase 5: Portfolio backtest depth + PatchTST transformer

**Date:** 2026-07-21 · **Project:** Stock-Price-Forecaster upgrade (Project C) · **Role:** Time-Series ML Engineer

## Resume gap progress

**Gap:** the repo predicted single tickers in isolation and had no portfolio-level evaluation, no position sizing, no risk management, no cost-sensitivity analysis — and, for a "modern ML" project, no modern architecture. **Today:** a full multi-ticker portfolio engine (`src/backtest/portfolio.py`: 4 sizing schemes × 3 stop-loss settings × cost sweep × regime slicing, all on frozen out-of-sample walk-forward predictions) and the sprint's first transformer time-series model (`src/models/patchtst.py`, a faithful small-scale PatchTST) measured on the identical folds as every prior model. Both stories are honest negatives — which is the resume claim this project runs on.

## Files touched

| File | Change |
|---|---|
| `src/backtest/portfolio.py` | NEW (~280 lines) — sleeve-based portfolio engine: `equal_sleeve` / `equal_active` / `inv_vol` / `signal_prop` sizing, per-sleeve stop-loss with re-arm, turnover costs, Sharpe/Sortino/max-DD, drifting equal-weight buy-and-hold benchmark, regime slicing |
| `src/models/patchtst.py` | NEW (~120 lines) — PatchTST-style Keras transformer: patch embedding (L=64, patch 8), learned positional embedding, 2 pre-norm encoder blocks, 17.7k params; train-only scaler; same `predict_fold` contract as every registry model |
| `src/models/__init__.py` | registered `patchtst` in the model registry (lines 21, 38–40) |
| `experiments/day07_portfolio_transformer.py` | NEW (~330 lines) — 4 models × 10 tickers × 5 folds OOS generation, bake-off table, 25-config portfolio grid, cost sweep, SPY-regime slicing, 4 plots |

## Setup

- CPU only; total runtime 11.3 min (PatchTST is the long pole at 6.8 s/fit × 50 fits).
- Identical protocol to Days 2–6: 10 tickers (AAPL MSFT SPY GOOGL AMZN META NVDA JPM XOM KO), 2021-01-01→2025-01-01, 5 expanding walk-forward folds, `assert_no_peeking` on every fold set, scaler fit on train only, costs 5 bps/side on turnover, public yfinance data via `src.data.loader`.
- Portfolio inputs are the frozen OOS predictions — nothing is refit inside the backtest. Aligned panel: 479 common OOS days across all 10 names.

## Experiment 1 — PatchTST vs the Day-3 field (identical folds)

**Hypothesis:** a patch-based transformer — the architecture that made transformers competitive on long-horizon TS benchmarks — extracts structure the LSTM(32) misses. Capacity-matched small (17.7k vs ~5k params) so the comparison is about architecture, not size.

| model | RMSE (returns) | RMSE vs zero | dir-acc | mean \|pred\| | per-ticker Sharpe | fit secs |
|---|---|---|---|---|---|---|
| ARIMA (Day-3 champion) | 0.01665 | 1.0006 | **0.535** | 0.0010 | 1.34 | 1.6 |
| XGB tuned+decay (Day 6) | 0.01669 | 1.0026 | 0.531 | 0.0014 | 1.19 | 0.16 |
| LSTM(32) (shipped arch) | 0.01665 | 1.0011 | 0.517 | 0.0010 | 1.03 | 4.9 |
| **PatchTST (new)** | **0.01914** | **1.1592** | 0.515 | **0.0078** | 1.04 | 6.8 |
| always-up baseline | — | — | 0.549 | — | 1.83 (=B&H) | 0 |

**Interpretation:** PatchTST is the worst forecaster of the sprint — **15.9% worse RMSE than predicting zero**, the largest miss of any model tested (the leaky-notebook era excluded). The tell is `mean |pred|` = 0.0078: ~8× bolder than ARIMA/LSTM, on a series whose honest next-day expectation is ≈0. Patching gives attention 8 tokens of 8-day context to attend over, and it finds "patterns" that are noise — the same failure Optuna exposed in XGB on Day 6, but worse because attention has more ways to memorise. The LSTM avoids this only by collapsing to ≈0 predictions. Architecture is not signal; on daily returns there is nothing long-context to attend to.

## Experiment 2 — Portfolio grid: sizing × stop-loss (25 configs, 5 bps/side)

**Hypothesis:** even a weak per-name signal can win at book level if sizing and risk management (diversification, vol budgets, stops) do the heavy lifting.

Signals: `xgb_tuned_decay` (best XGB dir-acc, Day 6) and `patchtst`. Benchmark: equal-weight buy-and-hold with drift (one entry cost, never touched). Selected rows:

| config | Sharpe | Sortino | max DD | total ret | turnover/day | Δ Sharpe vs B&H |
|---|---|---|---|---|---|---|
| **equal-weight buy-and-hold** | **2.620** | 4.123 | −14.7% | **+121.3%** | 0 | — |
| xgb · equal_active · stop 5% | **2.835** | **4.818** | −11.3% | +118.7% | 0.37 | **+0.215** |
| xgb · equal_active · stop 10% | 2.661 | 4.319 | −11.5% | +106.6% | 0.32 | +0.040 |
| xgb · equal_active · no stop | 2.490 | 3.871 | −11.5% | +93.3% | 0.29 | −0.130 |
| xgb · equal_sleeve · no stop | 2.351 | 3.553 | −9.6% | +65.1% | 0.12 | −0.269 |
| xgb · inv_vol · no stop | 2.068 | 3.007 | −7.0% | +45.7% | 0.16 | −0.552 |
| xgb · signal_prop · no stop | 1.927 | 3.204 | −16.9% | +110.8% | 0.40 | −0.693 |
| patchtst · best (equal_sleeve) | 2.202 | 3.472 | −5.3% | +42.0% | 0.48 | −0.418 |
| patchtst · worst (signal_prop) | 1.084 | 1.673 | −19.3% | +50.8% | 1.36 | −1.536 |

**Interpretation:** 23 of 25 configs lose to buy-and-hold on Sharpe. The two that win are adjacent cells of the same family (equal_active + stop), and the best one still has **lower total return than B&H** (+118.7% vs +121.3%) — its Sharpe edge is entirely vol reduction, and picking the best cell of a 25-cell grid after the fact is exactly the selection bias a walk-forward exists to prevent. We report it as what it is: one grid cell, not an edge. What the grid *does* show honestly: (a) timing out of the market reliably cuts drawdown (−14.7% → −5% to −11%) at the price of return; (b) stop-losses helped only the concentrated scheme (equal_active 2.49→2.84) and *hurt* every diversified one — a stop on an already-small sleeve just locks in noise losses; (c) `signal_prop` sizing by forecast magnitude was the worst idea on both signals — forecast magnitude carries no information here, so sizing by it is sizing by noise (PatchTST signal_prop: 1.36 turns/day, Sharpe 1.08).

## Experiment 3 — Transaction-cost sensitivity (best active scheme, no stop)

| cost (bps/side) | active Sharpe | B&H Sharpe | Δ |
|---|---|---|---|
| 0 | 2.802 | 2.620 | +0.181 |
| 1 | 2.738 | 2.620 | +0.118 |
| 2 | 2.676 | 2.620 | +0.056 |
| 5 | 2.490 | 2.620 | −0.130 |
| 10 | 2.190 | 2.620 | −0.431 |
| 20 | 1.619 | 2.620 | −1.001 |

**Interpretation:** the break-even is ≈**3 bps/side**. In a fantasy zero-cost world the active book edges ahead; at the realistic retail 5 bps it is already behind, and at 20 bps it has given up a full Sharpe point. Cost sensitivity is the difference between a backtest and a brochure.

## Experiment 4 — Regime slicing (SPY trailing 63-day return, shifted)

| regime | days | active Sharpe | B&H Sharpe |
|---|---|---|---|
| bull | 277 | **1.51** | 0.67 |
| sideways | 192 | 3.24 | **4.69** |
| bear | 10 | 0.37 | 1.05 |

**Interpretation:** the active book's whole (apparent) merit lives in the bull regime, where concentrating in signalled names beat the drifting 1/N book. In the sideways regime — where a timing signal should shine — B&H crushed it, because 2024's "sideways" was chop that repeatedly shook the signal out at 5 bps a turn. Only 10 bear days in the span, so the bear column is anecdote, not evidence.

## Head-to-head leaderboard (running, walk-forward, net of 5 bps)

Per-ticker Sharpe: B&H 1.83 > ARIMA 1.34 > XGB tuned 1.19 > LSTM 1.03 ≈ PatchTST 1.04 > momentum 0.90 > persistence 0.00. Portfolio level: B&H 2.62 > best defensible active ≈2.49. Directional accuracy: nothing has beaten always-up (0.549) in seven days of trying.

## Key findings

1. **Diversification is the best "model" in the repo.** Going from single tickers to a 1/N book lifted buy-and-hold Sharpe 1.83 → 2.62 — a bigger jump than any forecasting model, feature set, tuning run, or architecture delivered in six days of ML work. It costs one line of arithmetic and zero API calls.
2. **PatchTST is a genuine negative result:** 15.9% worse RMSE than predicting zero with 8× bolder predictions — attention over patch tokens manufactures conviction from noise on a series with ~no day-scale memory. Worth knowing before anyone puts "transformer" on this resume line: the honest phrasing is "evaluated PatchTST, rejected it with evidence."
3. **The one grid cell that beats B&H is selection bias wearing a stop-loss.** 2 of 25 configs win on Sharpe, they're neighbours, the winner loses on total return, and the win evaporates above ~3 bps/side costs.
4. **Stop-losses are not free risk management:** they helped only the concentrated scheme and hurt every diversified one — with 1/N sleeves the stop mostly converts temporary noise into realised losses (28 stop-outs at 5%).
5. **What didn't work and why:** `signal_prop` sizing (forecast magnitude is noise → sizing by it maximises turnover, 1.36 turns/day on PatchTST); `inv_vol` (down-weights the high-vol names that carried the 2023–24 market — risk parity is a bear-market technology tested on a bull tape).

## Sample outputs saved

- `results/phase5_transformer.csv`, `results/phase5_portfolio.csv`, `results/phase5_cost_sweep.csv`, `results/phase5_regimes.csv`
- `results/plots/day07_portfolio_equity.png`, `day07_cost_sweep.png`, `day07_regime_sharpe.png`, `day07_transformer_bakeoff.png`
- `results/samples/day07_oos_predictions.csv` (all 4 models × 479 aligned OOS days × 10 tickers), `day07_patchtst_sample_preds.csv`
- `results/metrics.json` → `day07` entry

## Next Day

Day 8 — Phase 6: frontier comparison (the designed negative result — Claude Opus 4.6 / GPT-5.4 next-day direction on 200 samples vs the specialized stack, expecting ≈coin-flip at ~2s and real cost) + the full ablation (persistence → returns target → features → champion → tuning). **[POST · PHASE-WRAP]**

## Code changes

New: `src/backtest/portfolio.py`, `src/models/patchtst.py`, `experiments/day07_portfolio_transformer.py`. Modified: `src/models/__init__.py` (registry + patchtst entry).
