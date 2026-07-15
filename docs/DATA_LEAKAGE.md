# Data Leakage — The Scaler Bug and Its Fix

**Component:** `MinMaxScaler` normalisation in `predictor.py` and `stock_predictor.py`
**Severity:** Critical (silent metric inflation) · **Status:** Fixed Day 1

---

## The bug

Both prediction pipelines normalised the close-price series like this:

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)   # (1) fit on the FULL series
...
X, y = create_dataset(scaled_data, time_step)
split = int(len(X) * 0.8)                            # (2) THEN split train/test
X_train, X_test = X[:split], X[split:]
```

`MinMaxScaler.fit` learns two constants from its input: `data_min_` and `data_max_`. Here it sees the
**entire** series — including the test window — so those constants encode information about **future
prices the model is not supposed to have seen at training time.** Every training sample is then scaled
using knowledge of the test period's high and low. That is textbook look-ahead leakage.

### Why it inflates the score
The scaled target the model learns to predict is `(price - min) / (max - min)`. When `max`/`min` are
computed with the test set included, the model trains on a normalisation that is already "aware" of the
test range, so predictions on the test set land in a pre-calibrated band and error looks smaller than it
honestly is. The effect is largest exactly when the test window sets **new extremes** (a strong trend),
which is precisely when an honest model would struggle most.

---

## The fix

Choose the temporal split on the **raw** price series, fit the scaler on the **train slice only**, then
transform the whole series with those train-derived constants:

```python
raw_split = int(len(close_prices) * 0.8)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(close_prices[:raw_split])     # train-only fit — no peeking
scaled_data = scaler.transform(close_prices)
```

The test set can now legitimately fall outside `[0, 1]` after scaling — that is correct and expected;
in production you genuinely do not know tomorrow's high/low. The windowed 80/20 split is unchanged (it
was already temporal and sound); **only the scaler fit moved.** Function signatures are preserved so the
Flask app and Streamlit app keep working.

Applied at:
- `predictor.py` — was lines 78–79
- `stock_predictor.py` — was lines 229–230

---

## Measured impact (10 tickers, 2021-01-01 → 2025-01-01, 195 test days each)

| Mean over 10 tickers | Leaky (old) | Fixed (honest) | Change |
|---|---|---|---|
| Test RMSE (price $) | 6.58 | **6.98** | **+5.2%** |

Per-ticker the inflation ranges from ~0% on range-bound names (KO, XOM) to **+20–23%** on strongly
trending names (AAPL +23%, NVDA +22%, GOOGL +16%, SPY +13%) — exactly where the test window pushed new
extremes. A few tickers show a small negative delta; that is LSTM training stochasticity (fixed random
seed per config, but CPU non-determinism remains), not the leak reversing. **In aggregate the leaky
number is optimistic, and the honest number is what we report from now on.**

## The bigger honest finding

The leak is real but modest here (~5%). The dominant result is that the leakage-fixed LSTM
(RMSE 6.98) is **beaten by a one-line persistence baseline `ŷ_t = y_{t-1}` (RMSE 3.80) on all 10
tickers**, with next-day directional accuracy of 0.48 — below a coin flip. See `TS_AUDIT.md` and
`results/baseline_metrics.json`. **Fixing the leak is table stakes; the real story is that a price-level
LSTM has no measurable edge, and the upgrade's value is proving that honestly.**

## Regression guard

Day 10 adds `tests/test_no_scaler_leakage.py`: asserts the scaler's `data_min_`/`data_max_` equal those
of a scaler fit on the **train indices only**, so this bug can never silently return.
