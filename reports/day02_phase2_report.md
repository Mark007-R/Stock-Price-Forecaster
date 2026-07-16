# StockAI Production Upgrade — Day 2 / 10
## Phase 2a: Target Reframing (price → return) + Walk-Forward Harness

**Date:** 2026-07-16 · **Field:** Time-Series ML / Quant Backtesting

---

### Resume gap progress
**Gap:** the project scored its LSTM on a single arbitrary 80/20 holdout, in
absolute-price space — a setup that flatters any model and can't tell signal
from a slow-moving price level.
**Today's contribution:** (1) built a reusable **expanding-window walk-forward**
harness (`src/backtest/walkforward.py`) that refits per fold with a structural
no-peeking guarantee — the evaluation backbone every later day reuses; and
(2) **reframed the target from absolute price to next-day return** and scored
both framings on identical folds. The reframing works exactly as theory
predicts — and in doing so it exposes that the shipped price-level accuracy was
an illusion of scale. The honest finding: *next-day returns are essentially a
random walk here.*

---

### Files touched
| File | Change |
|------|--------|
| `src/backtest/walkforward.py` | **New** — `Fold`, `expanding_window_folds`, `rolling_window_folds`, `assert_no_peeking`, return-space metrics, `walk_forward_predict` orchestrator |
| `src/backtest/__init__.py`, `src/__init__.py` | **New** — package exports |
| `experiments/day02_walkforward.py` | **New** — reframing experiment: LSTM-price vs LSTM-returns vs 3 baselines across folds |
| `results/walkforward.csv` | **New** — 250 rows (10 tickers × 5 folds × 5 methods) |
| `results/phase2a_walkforward_summary.csv`, `results/metrics.json` | **New** — method-level aggregate + append-only metrics log |
| `results/plots/day02_*.png` (×4), `results/samples/day02_*_walkforward.csv` (×10) | **New** |

### Setup
- **Compute:** CPU (TensorFlow 2.15). 100 LSTM refits (2 framings × 10 tickers ×
  5 folds), ~11.5 min wall-clock. EarlyStopping (patience 3) on each fold.
- **Data:** public yfinance daily closes, same fixed 10 tickers
  (`AAPL MSFT SPY GOOGL AMZN META NVDA JPM XOM KO`), **2021-01-01 → 2025-01-01**
  (~1005 days/ticker). No fabricated data.
- **Folds:** 5 expanding windows, first train ~505 days, test block ~100 days
  each, refit per fold. `test_start == train_end` on every fold (asserted).
- **Model:** `LSTM(32)→Dropout(0.2)→Dense(16)→Dense(1)`, time_step 30 — identical
  to `predictor.py`. Only the **target** differs between the two LSTM runs.
- **Scoring space:** next-day simple returns for *all* methods.

---

### Experiment 1 — Does the price-level LSTM survive being scored as a return forecaster?

**Hypothesis:** the tight "predicted vs actual" price curve is an artifact of a
slow-moving absolute level. Convert its price forecast to a return and it should
collapse toward — or below — a zero-return random walk.

**Method:** run the shipped price-LSTM through the walk-forward harness; convert
each predicted price to a predicted return `p̂_t / p_{t-1} − 1`; score RMSE and
directional accuracy in return space against the random walk `r̂ = 0`.

| Mean over 10 tickers × 5 folds | RMSE (returns) | Directional acc |
|---|---|---|
| **LSTM-price** (as returns) | 0.0307 | 0.489 |
| Random walk (`r̂ = 0`) | **0.0164** | — |

**Interpretation:** as a return forecaster the price-LSTM has **~1.9× the random
walk's RMSE** and directional accuracy **below a coin flip** — worse on every
single ticker. The impressive price chart was scale, not skill. This is Day 1's
finding made rigorous across folds.

### Experiment 2 — Reframe to returns: does an honest target buy an edge?

**Hypothesis:** predicting the return directly is the correct task; it should
drop RMSE to the random-walk floor. Whether it *beats* that floor (real edge) is
the open question.

**Method:** same architecture, target = next-day return (StandardScaler fit on
train returns only). Same folds. Compare to random walk, naive momentum
(`r̂_t = r_{t−1}`), and an always-up directional baseline.

| Mean over 10 tickers × 5 folds | RMSE (returns) | Directional acc | dir-acc σ across folds |
|---|---|---|---|
| **LSTM-returns** | **0.01640** | 0.519 | 0.052 |
| Random walk (`r̂ = 0`) | 0.01638 | — | — |
| Momentum (`r̂_t = r_{t−1}`) | 0.0231 | 0.511 | 0.045 |
| Always-up | — | **0.549** | 0.044 |

**Interpretation:** the reframing lands the RMSE **exactly on the random-walk
floor** (0.01640 vs 0.01638 — a 0.1% *loss*), beating it on only **3/10 tickers**
by 4th-decimal margins. That is the signature of a network that has correctly
learned the MSE-optimal forecast of a near-zero-mean series: **predict ≈ 0.** On
direction it clears a coin flip on 32/50 folds (0.519) — but **cannot beat the
trivial always-up baseline (0.549)**, which wins on 8/10 tickers because the
market simply drifted up over 2021–2025. And the fold-to-fold σ (±0.05) is as
large as the entire edge over 0.50, so the "edge" is noise.

---

### Head-to-Head leaderboard (Day 2 — walk-forward, return space)
| Rank | Method | RMSE (returns) | Dir. acc | Note |
|------|--------|----------------|----------|------|
| 1 | Random walk `r̂=0` | **0.01638** | n/a | unbeaten on RMSE |
| 2 | **LSTM-returns** | 0.01640 | 0.519 | ties RW; can't beat always-up on direction |
| 3 | Always-up | — | **0.549** | best directional (market drift) |
| 4 | Momentum `r̂_t=r_{t-1}` | 0.0231 | 0.511 | worse RMSE, ~coin-flip |
| 5 | LSTM-price (as returns) | 0.0307 | 0.489 | ✗ scale illusion, sub-coin-flip |

*(Day 3 adds ARIMA / Prophet / XGBoost-on-features and a transaction-cost-aware
trading backtest vs buy-and-hold on these exact folds.)*

### Key findings
1. **The reframing did its job — and that's the bad news.** RMSE fell from 0.0307
   (price framing) to 0.0164 (returns) — a 47% drop that lands *precisely* on the
   zero-return random walk. The LSTM-returns essentially learned to output zero,
   which is the MSE-optimal forecast for near-unpredictable returns.
2. **No directional edge that survives scrutiny.** 0.519 is above a coin flip but
   below the always-up baseline (0.549) and swamped by ±0.05 fold-to-fold noise.
   A model that can't beat "bet up every day" has no demonstrated skill.
3. **Walk-forward changes the verdict's confidence, not its direction.** Five
   independent out-of-sample windows all say the same thing, so this isn't one
   unlucky holdout — it's the honest state of next-day equity prediction.

### What didn't work (and why)
- **LSTM-returns as an alpha source.** Not a tuning failure — a fact about the
  target. Daily equity returns are ~0-mean, heavy-tailed, and near-serially-
  uncorrelated, so MSE training drives the output to zero and there is little
  linear direction to exploit. Day 3 tests whether *engineered features*
  (lagged returns, RSI/MACD, vol) or a different learner extracts anything the
  raw-price LSTM cannot — measured net of transaction costs, where the bar is
  higher still.

### Sample outputs saved
- `results/walkforward.csv` — full 250-row per-(ticker,fold,method) table
- `results/phase2a_walkforward_summary.csv` — method-level aggregate
- `results/metrics.json` — append-only log (`day02` entry)
- `results/plots/day02_return_rmse.png`, `day02_directional_accuracy.png`,
  `day02_dir_acc_per_fold.png`, `day02_rmse_per_ticker.png`
- `results/samples/day02_<TICKER>_walkforward.csv` — last-60-day return
  predictions per method (all 10 tickers)

### Next day (Day 3)
Model bake-off on the same folds — persistence vs ARIMA vs Prophet vs
XGBoost-on-features (indicators from `historical.py:calculate_technical_indicators`)
vs LSTM(returns) — plus a **transaction-cost-aware trading backtest**
(cumulative return, Sharpe, max drawdown) vs buy-and-hold. Pick a champion.

### Code changes
`sprint: add expanding walk-forward harness (src/backtest/walkforward.py) + reframe target price→return (experiments/day02_walkforward.py) — LSTM-returns RMSE 0.0164 ties the random walk, dir-acc 0.519 < always-up 0.549 (Day 2 Phase 2a)`
