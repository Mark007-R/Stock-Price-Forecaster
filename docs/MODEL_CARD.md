# Model card — StockAI forecasting stack

**Champion model:** ARIMA (order by AIC per training window, grid
{(1,0,0),(0,0,1),(1,0,1),(2,0,2)}) on next-day simple returns, with
split-conformal prediction intervals from its own held-out 1-step errors.
Served by `src/serving/api.py` (`/predict`, `/backtest`); registry in
`src/models/__init__.py`.

## Intended use

- Demonstration of honest time-series evaluation methodology: walk-forward
  splits, persistence + buy-and-hold baselines, cost-aware backtests,
  calibrated uncertainty.
- Educational/portfolio use. **Not investment advice, and not an edge:** the
  champion measurably loses to buy-and-hold net of costs (Sharpe 1.34 vs 1.83
  across the 10-ticker walk-forward). Every API response says so in its
  `disclaimer` field.

## Training & evaluation data

- Public daily OHLCV via yfinance (`auto_adjust=True`), fixed universe of 10
  liquid US large-caps + SPY (AAPL, MSFT, SPY, GOOGL, AMZN, META, NVDA, JPM,
  XOM, KO), fixed range 2021-01-01 → 2025-01-01 (~1,005 trading days each,
  ~5,000 out-of-sample days pooled).
- No survivorship-safe point-in-time universe: tickers were chosen in 2026 and
  are all survivors. Any absolute performance number inherits that bias —
  another reason the only claims made are *relative* (vs persistence, vs
  buy-and-hold on the same days).

## Evaluation protocol

- Expanding-window walk-forward, 5 folds, test blocks contiguous and
  non-overlapping (`src/backtest/walkforward.py`; invariants enforced at
  runtime by `assert_no_peeking` and pinned by `tests/`).
- Scored in return space: RMSE(returns) + directional accuracy vs the
  persistence floor (r̂ = 0) and the always-up base rate.
- Trading evaluation: long/flat with 5 bps/side charged on turnover vs
  buy-and-hold on identical days (`src/backtest/trading.py`). Zero-cost
  backtests are refused at the API schema level.

## Key results (walk-forward, 10 tickers, net of 5 bps costs)

| Metric | Value |
|---|---|
| ARIMA RMSE(returns) vs persistence | 0.016377 vs 0.016380 (−0.02%) |
| ARIMA directional accuracy | 0.534 (always-up baseline: 0.549) |
| ARIMA Sharpe vs buy-and-hold | 1.34 vs **1.83** |
| Conformal 80% / 95% band coverage | 80.8% / 99.2% (nominal 80 / 95) |
| Full ablation (5 rungs, 5,000 days) | dir-acc 0.517 → 0.535, all < 0.549 |

## Limitations & failure modes

- **No demonstrated edge.** The model's point forecasts sit at the random-walk
  floor; its value is calibrated uncertainty, not alpha.
- Day-6 failure-mode analysis: the signal's sign flips across trend regimes
  (mean-reversion patterns inverted in 2023–24); directional accuracy is not
  stable in any bucket.
- Intervals assume exchangeable 1-step errors; √h widening for multi-step
  horizons additionally assumes independent errors. A volatility-regime break
  between calibration and serving degrades coverage until the window refills.
- US large-cap daily bars only. Nothing here transfers to intraday,
  small-caps, or other asset classes without re-measurement.

## Ethical / responsible-use notes

- Public market data only; no personal data anywhere in the pipeline.
- The serving layer deliberately couples every forecast to its disclaimer and
  every backtest to its benchmark — removing those couplings would misrepresent
  the system and is the one modification the design actively resists.
