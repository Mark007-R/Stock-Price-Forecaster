# StockAI Production Upgrade — Day 3 / 10
## Phase 2b: Model Bake-Off + Transaction-Cost-Aware Trading Backtest

**Date:** 2026-07-17 · **Field:** Time-Series ML / Quant Backtesting

---

### Resume gap progress
**Gap:** the project could not say whether its model made money — it had no
competitor models, no trading simulation, no transaction costs, and no
buy-and-hold benchmark. "Predicts stock prices" was a claim about a chart.
**Today's contribution:** ran **seven methods across five model families** on the
identical walk-forward folds, then turned every forecast into a **P&L net of 5
bps/side costs** and set it against **buy-and-hold on the same days**. This
closes the objection Day 2 left open — *maybe the LSTM was just the wrong
learner* — and replaces it with a measured answer: **no**, the learner was never
the problem. The resume claim is now a cost-aware multi-model backtest with a
baseline, which is what a quant actually asks for.

---

### Files touched
| File | Change |
|------|--------|
| `src/features/engineer.py` | **New** (L1–178) — `build_feature_frame` (18 stationary features: 5 lagged returns + indicators **imported from `historical.py:calculate_technical_indicators`**, not re-implemented), `assert_no_lookahead` — an empirical causality proof, not a comment |
| `src/backtest/trading.py` | **New** (L1–163) — `backtest_long_flat` (turnover-charged), `backtest_buy_and_hold`, `BacktestResult`; Sharpe / max-DD / cost-drag / exposure |
| `src/backtest/__init__.py` | Export the trading sim alongside the Day-2 walk-forward harness |
| `src/features/__init__.py` | **New** — package exports |
| `experiments/day03_bakeoff.py` | **New** (L1–470) — 7 methods × 10 tickers × 5 folds + trading sim + `aggregate()` + 6 plots |
| `.gitignore` | Ignore cached market-data pulls (`data/eval/prices_*.csv`), `mlruns/`, model weights |
| `results/phase2b_models.csv`, `phase2b_trading.csv`, `leaderboard.csv`, `phase2b_xgb_gain.csv`, `metrics.json` | **New/updated** — 350 per-fold rows, 80 per-ticker backtests, leaderboard |
| `results/plots/day03_*.png` (×6), `results/samples/day03_*_oos_predictions.csv` (×10) | **New** |

### Setup
- **Compute:** CPU. 350 model fits (7 methods × 10 tickers × 5 folds) in **7.1 min**
  wall-clock. Per-fit cost: LSTM 4.54 s, XGBoost 1.61 s, ARIMA 1.59 s,
  Prophet 0.24 s, baselines ~0 s.
- **Data:** public yfinance daily closes, same fixed 10 tickers
  (`AAPL MSFT SPY GOOGL AMZN META NVDA JPM XOM KO`), **2021-01-01 → 2025-01-01**
  (~1005 days/ticker). Cached to `data/eval/` for reproducibility (gitignored).
  No fabricated data; no costs hidden.
- **Folds:** identical to Day 2 — 5 expanding windows, train 505→905, test 100
  each, refit per fold, `test_start == train_end` asserted.
- **Out-of-sample span:** the 5 test blocks concatenate into one continuous
  **500-day** stream per ticker (2023-01-05 → 2024-12-31) — that is what the
  trading sim runs on.
- **Trading rule:** long when predicted return > 0, else flat. Costs **5 bps per
  side** on every position change (commission + spread + slippage, retail
  large-cap). Benchmark: buy-and-hold over the identical days.

---

### Experiment 1 — Is the failure the LSTM, or the task?

**Hypothesis:** Day 2's null result might be an artifact of one architecture fed
one representation. A classical linear model (ARIMA), a structural model
(Prophet), or gradient-boosted trees on real engineered features should extract
*something* the raw-window LSTM cannot — if there is anything to extract.

**Method:** all five families on identical folds, scored in return space against
the zero-return random walk (`r̂ = 0`).

| Method | RMSE (returns) | vs random walk | Beats RW (tickers) | Dir. acc | σ across folds |
|---|---|---|---|---|---|
| Random walk `r̂=0` | **0.016380** | — | — | n/a | — |
| ARIMA (AIC-selected) | **0.016377** | **−0.02%** | 5/10 | 0.534 | 0.051 |
| LSTM-returns | 0.016394 | +0.09% | 4/10 | 0.519 | 0.055 |
| Prophet | 0.017060 | +4.15% | 0/10 | 0.501 | 0.059 |
| XGBoost (18 features) | 0.017907 | **+9.32%** | 0/10 | 0.516 | 0.060 |
| Momentum `r̂_t=r_{t−1}` | 0.023088 | +40.95% | 0/10 | 0.511 | 0.046 |
| Always-up | — | — | — | **0.549** | 0.045 |

**Interpretation:** the objection is dead. **Every family converges on the same
random-walk floor or lands worse.** ARIMA's "win" is −0.02% — the 4th decimal,
on 5/10 tickers, i.e. a coin flip about which side of the floor it falls. More
telling: **XGBoost, the only model given real engineered features, is 9.3% WORSE
than predicting zero, on 0/10 tickers** — the features did not add signal, they
added variance. And **Prophet's directional accuracy is 0.5006**, a literal coin
flip (23/50 folds above 0.50), which is what a trend-and-seasonality model should
score on a series with neither. Not one model beats the trivial always-up
baseline (0.549) on direction.

### Experiment 2 — Does any of it make money after costs?

**Hypothesis:** statistical scores and profitability are different questions. A
model near the random-walk floor might still ride the drift into a decent P&L —
or the turnover might eat it alive.

**Method:** concatenate each model's 5 folds into one 500-day OOS stream per
ticker; long when predicted up; charge 5 bps/side on turnover; compare to
buy-and-hold on the same days. Means across 10 tickers.

| Method | Cum. return (net) | Sharpe | Max DD | Trades | Exposure | Cost drag | Beats B&H (tickers) |
|---|---|---|---|---|---|---|---|
| **Buy & hold** | **+178.7%** | **1.83** | −17.9% | 1 | 100% | 0.1 pp | — |
| Always-up | +178.7% | 1.83 | −17.9% | 1 | 100% | 0.1 pp | 0/10 |
| ARIMA | +118.1% | 1.34 | −18.0% | 53 | 84.5% | 5.5 pp | 1/10 |
| XGBoost | +84.1% | 1.22 | −15.9% | 185 | 53.6% | **18.2 pp** | 1/10 |
| LSTM-returns | +79.6% | 1.11 | −19.2% | 56 | 71.6% | 5.3 pp | 1/10 |
| Momentum | +54.6% | 0.90 | −16.7% | 244 | 54.8% | **20.0 pp** | 1/10 |
| Prophet | +75.2% | 0.87 | −19.2% | 61 | 53.6% | 6.3 pp | 0/10 |
| Random walk `r̂=0` | 0.0% | 0.00 | 0.0% | 0 | 0% | 0 pp | 0/10 |

**Interpretation:** **nothing beats doing nothing.** The best forecasting model
(ARIMA) trails buy-and-hold by **0.49 Sharpe** and leaves **60 points of return**
on the table; each model wins on exactly **1/10 tickers**, which is noise. The
random walk earns 0.00% — correctly: it forecasts no edge, so it takes no
position, and a model honest enough to predict nothing bets nothing.

---

### The genuine insight — these are not strategies, they are diluted buy-and-hold

Rank the table by Sharpe and it reproduces the ranking by **exposure**, not by
accuracy. Measured across all methods:

> **corr(market exposure, Sharpe) = 0.971**

Sharpe is **97% explained by how many days the strategy simply held the stock.**
It is not explained by directional accuracy: XGBoost (0.516) outranks
LSTM (0.519) on Sharpe, and Prophet (0.501 — a coin flip) beats momentum (0.511).
The causal chain is not *forecast → edge → profit*. It is **exposure → drift →
profit, minus a tax for every trade.** These models are buy-and-hold with random
holes punched in it, and each hole costs 5 bps and a slice of the drift. That is
why the two highest-turnover methods (momentum 244 trades, XGBoost 185) surrender
**20.0 and 18.2 points** of return to costs while buy-and-hold surrenders **0.1**.

**Two independent checks confirm the finding rather than a bug:**
1. **Backtest identity check (passed):** `always_up` is long every day, so it
   *must* reproduce buy-and-hold exactly. It does — Sharpe gap **4.44e-16**,
   floating-point dust. The simulator's cost and turnover arithmetic is sound.
2. **XGBoost gain is flat:** across 18 features, gain spans **0.043–0.063**
   against a uniform-if-meaningless **1/18 = 0.0556**. The top feature
   (`mean_ret_5`, 0.0628) barely separates from the bottom (`ret_lag_0`, 0.0433)
   — a 1.45× spread where a genuine predictor would tower over the rest. No
   feature dominates because none carries signal — the trees are splitting on
   noise, which is exactly why the model's RMSE is *worse* than predicting zero.
   A wry detail: the five **raw lagged returns occupy the five lowest gain
   slots** (0.043–0.052), below every smoothed indicator. The model preferred
   averaged features precisely because averaging suppresses the noise it was
   otherwise forced to chew on.

---

### Head-to-Head leaderboard (Days 1–3, cumulative)
| Rank | Method | RMSE (returns) | Dir. acc | Sharpe (net) | Verdict |
|------|--------|----------------|----------|--------------|---------|
| 1 | **Buy & hold** | n/a | n/a | **1.83** | **unbeaten — the champion is no model at all** |
| 2 | Always-up (≡ buy & hold) | n/a | 0.549 | 1.83 | identity check for the backtest |
| 3 | **ARIMA** | **0.016377** | 0.534 | 1.34 | champion *among forecasting models*; still −0.49 Sharpe |
| 4 | XGBoost (18 feats) | 0.017907 | 0.516 | 1.22 | ✗ RMSE worse than zero-forecast; features add variance |
| 5 | LSTM-returns (Day 2) | 0.016394 | 0.519 | 1.11 | ties RW; no edge |
| 6 | Momentum | 0.023088 | 0.511 | 0.90 | ✗ turnover eats 20 pp |
| 7 | Prophet | 0.017060 | 0.501 | 0.87 | ✗ literal coin flip on direction |
| 8 | Random walk `r̂=0` | 0.016380 | n/a | 0.00 | RMSE floor; takes no position |
| — | LSTM-price (retired Day 2) | 0.0307 | 0.489 | — | ✗ scale illusion |

*(Day 4 adds feature-importance depth + multi-horizon; the frontier/LLM
comparison lands Day 8.)*

### Key findings
1. **The learner was never the problem — the task is.** Five model families,
   one conclusion: ARIMA, Prophet, XGBoost-on-features and the LSTM all land on
   the random-walk floor. Day 2 could be dismissed as "wrong architecture"; after
   Day 3 that dismissal is unavailable.
2. **Sharpe is 97% correlated with exposure, not accuracy.** The apparent
   "performance" of every strategy is market drift it failed to opt out of.
   Trading is a tax on that drift: 244 trades cost 20 points of return.
3. **Feature engineering made the forecast worse, and said so.** XGBoost is 9.3%
   worse than predicting zero with gain spread uniformly across 18 features —
   the fingerprint of fitting noise. Engineering more features is not the fix.
4. **Nothing beats buy-and-hold net of costs** (best: 1.34 vs 1.83, winning on
   1/10 tickers). The honest headline for this project is a measured negative
   result with a validated simulator — not a Sharpe.

### What didn't work (and why)
- **XGBoost on engineered features** — the day's biggest disappointment and its
  most useful result. Daily equity returns are ~0-mean and near-serially-
  uncorrelated; the indicators are all deterministic transforms of the same past
  prices, so they add no *independent* information. Trees then fit sample noise,
  which shows up as RMSE 9.3% above the zero-forecast and flat gain. Boosting
  cannot manufacture signal that is not in the inputs.
- **Prophet** — a structural mismatch, not a tuning failure. It decomposes trend +
  seasonality; daily returns are (by construction) the *differenced* series with
  the trend removed and no calendar seasonality. Fitting it to returns asks it to
  model the one thing it has no machinery for. Dir. acc 0.5006 is the honest score.
- **Momentum** — 41% worse RMSE than the random walk confirms daily returns are
  not positively autocorrelated; betting on yesterday repeating trades 244 times
  to lose 20 points to costs.

### Sample outputs saved
- `results/phase2b_models.csv` — 350 rows, per (ticker, fold, method): RMSE, dir-acc, fit time
- `results/phase2b_trading.csv` — 80 rows, per (ticker, method): return, Sharpe, max-DD, trades, exposure, cost drag
- `results/leaderboard.csv` — method-level aggregate + Δ vs buy-and-hold
- `results/phase2b_xgb_gain.csv` — mean feature gain (the flat-gain evidence)
- `results/metrics.json` — append-only (`day03` entry, incl. identity check + exposure correlation)
- `results/plots/day03_spy_equity_curves.png` — **the headline picture**: buy-and-hold on top, every model below
- `results/plots/day03_{sharpe_vs_buyhold,cumulative_return,rmse_by_model,directional_accuracy,cost_drag}.png`
- `results/samples/day03_<TICKER>_oos_predictions.csv` — full 500-day OOS predictions, all 7 methods × 10 tickers

### Next day (Day 4)
Phase 2c — feature-engineering depth + multi-horizon. Expand the feature set
(calendar effects, regime flags), measure importance properly with **SHAP** (today's
gain analysis says there is nothing to find — SHAP tests that claim with a better
instrument), and ask whether the edge decays or strengthens at **1-day vs 5-day vs
20-day** horizons. Day 3's null result at h=1 makes the horizon question the most
promising remaining lead: noise averages out as the horizon grows.

### Code changes
`sprint: add cost-aware trading backtest (src/backtest/trading.py) + no-look-ahead features (src/features/engineer.py) + 5-family model bake-off (experiments/day03_bakeoff.py) — nothing beats buy-and-hold net of costs (best ARIMA Sharpe 1.34 vs 1.83); Sharpe correlates 0.971 with exposure, not accuracy (Day 3 Phase 2b)`
