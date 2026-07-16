# Target Reframing: Absolute Price → Next-Day Return

**Day 2 · Phase 2a** — companion to [`DATA_LEAKAGE.md`](DATA_LEAKAGE.md) and
[`TS_AUDIT.md`](TS_AUDIT.md).

## The problem with predicting price

`predictor.py` and `stock_predictor.py` both train the LSTM to predict the
**absolute next-day close**. Absolute equity prices are near-integrated: today's
price is ~99% of the information about tomorrow's. A model that simply echoes the
last price (persistence) therefore looks excellent, and any network trained on
MSE learns a slightly-smoothed echo. The result is a visually tight
"predicted vs actual" curve that encodes **no forecasting skill** — only the
autocorrelation of the price level.

Day 1 showed this on a single holdout (LSTM RMSE 6.98 vs persistence 3.80 in
price space). Day 2 proves it is not an artifact of one window.

## The fix: score in return space

The economically meaningful, near-unpredictable quantity is the **return**
`r_t = p_t / p_{t-1} − 1`. We evaluate every model in return space:

- **RMSE(returns)** — how far the point forecast is from the realised return.
  The benchmark is the zero-return random walk (`r̂ = 0`), whose RMSE equals the
  realised volatility. Beating it means genuine point-forecast skill.
- **Directional accuracy** — fraction of days the predicted return *sign* matches
  the realised sign (flat days excluded). Benchmarks: coin flip (0.50), naive
  momentum, and always-up (which is strong in an up-trending market).

Two LSTM framings are compared on identical walk-forward folds:

| Framing | Target | Scored as |
|---|---|---|
| **LSTM-price** (shipped) | next-day price | convert `p̂` → `r̂ = p̂/p_{t-1} − 1` |
| **LSTM-returns** (reframed) | next-day return | directly |

## What the reframing revealed (walk-forward, 10 tickers × 5 folds)

| Method | RMSE(returns) | Dir. acc |
|---|---|---|
| LSTM-price (as returns) | 0.0307 | 0.489 |
| **LSTM-returns** | **0.0164** | 0.519 |
| Random walk `r̂=0` | 0.0164 | — |
| Always-up | — | 0.549 |

- Reframing cut RMSE **47%** — but only down to the **random-walk floor**, not
  below it (beats RW on just 3/10 tickers, by 4th-decimal margins). The network
  learned the MSE-optimal forecast of a ~0-mean series: **predict ≈ 0.**
- Directional accuracy (0.519) beats a coin flip but **loses to always-up
  (0.549)** and is within the ±0.05 fold-to-fold noise band.

**Conclusion:** the honest task is returns, and on the honest task a vanilla LSTM
has no demonstrated edge over trivial baselines. This is the correct baseline for
the Day-3 model bake-off + cost-aware trading backtest — a model earns its place
only by beating the random walk on RMSE *and* buy-and-hold net of costs.

## Regression guard (planned Day 10)

`tests/test_temporal_split.py` and `tests/test_walkforward_no_peeking.py` assert
that (a) folds are contiguous with `test_start == train_end`, and (b) scalers and
models are fit strictly on `prices[:train_end]` — no test day ever enters
training or normalisation.
