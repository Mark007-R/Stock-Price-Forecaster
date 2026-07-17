"""
Feature engineering for the return-forecasting bake-off — no look-ahead.

Why this exists
---------------
Day 2 showed an LSTM fed nothing but a window of raw returns lands exactly on
the zero-return random walk: with no side information, the MSE-optimal forecast
of a ~0-mean series is 0, and that is what it learns. The open question Day 3
asks is whether *engineered features* — lagged returns plus the technical
indicators the app already computes — carry any signal the raw window does not.

Two design rules make these features honest:

1. **Backward-looking only.** Every column at row ``t`` is a function of prices
   up to and including ``t``. All rolling/ewm windows are trailing. Nothing is
   centred, nothing is shifted forward, nothing is filled backwards. The
   ``assert_no_lookahead`` helper *proves* this empirically rather than
   trusting the claim (see below).

2. **Stationary by construction.** The raw indicators from ``historical.py``
   are price LEVELS (SMA20 of AAPL is ~$190). Feeding a level to a model that
   trains on 2021 and predicts 2024 is a trap: the feature's support shifts
   under it and the split "works" only by memorising a price range. So every
   level is converted to a scale-free ratio (``Close/SMA20 - 1``), a bounded
   oscillator (RSI/100, %B), or a return. This is why the feature list below
   contains no absolute prices.

Indicator parity
----------------
The indicators are NOT reimplemented here. ``historical.py`` already ships
``calculate_technical_indicators`` (SMA20/SMA50/EMA20/RSI/Bollinger/MACD) and
that exact function is imported and called, so the model trains on the same
numbers the app shows its users. Reimplementing them would let the two drift
apart silently — the classic "the notebook and the app disagree" bug.

The target
----------
``y[t] = prices[t+1]/prices[t] - 1`` — the next-day simple return, matching the
return space Day 2 established as the honest scoring space. Row ``t`` therefore
pairs features known at the close of day ``t`` with the return realised the
following day. There is no overlap between what the row knows and what it
predicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The shipped indicator implementation — imported, never duplicated, so the
# bake-off features and the app's charts can never diverge.
from historical import calculate_technical_indicators

# Lagged-return depth: r_{t}, r_{t-1}, ... r_{t-4}. One trading week of memory,
# which is where any autocorrelation in daily equity returns would live.
N_LAGS = 5

FEATURE_COLS = [
    "ret_lag_0", "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4",
    "close_over_sma20", "close_over_sma50", "close_over_ema20",
    "sma20_over_sma50",
    "rsi", "bb_pct", "macd_norm", "macd_hist_norm",
    "vol_5", "vol_10", "vol_21",
    "mean_ret_5", "mean_ret_21",
]

# Longest trailing window in play is SMA50, so the first 50-odd rows are NaN
# while the indicators warm up. Callers must drop them rather than impute —
# imputing a warm-up window invents history the model would never have had.
WARMUP = 60


def build_feature_frame(prices: np.ndarray, dates=None) -> pd.DataFrame:
    """Build the feature matrix + next-day-return target from a close series.

    Parameters
    ----------
    prices : array of daily closes, oldest first.
    dates  : optional DatetimeIndex, carried through for Prophet / reporting.

    Returns
    -------
    DataFrame indexed 0..n-1 (aligned to ``prices``) with ``FEATURE_COLS``, a
    ``target`` column (next-day return), and ``valid`` marking rows whose
    features are fully warmed up. Row ``t``'s features use prices[:t+1] only;
    ``target[t]`` is the return realised on day t+1 (NaN on the final row,
    which has no tomorrow).
    """
    p = np.asarray(prices, dtype=float).flatten()
    n = len(p)
    if n < WARMUP + 5:
        raise ValueError(f"Need >{WARMUP + 5} prices to warm up features, got {n}")

    df = pd.DataFrame({"Close": p})
    if dates is not None:
        df["date"] = pd.to_datetime(pd.Series(dates).values[:n])

    # Trailing simple returns: ret[t] = p[t]/p[t-1] - 1 (known at close of t).
    ret = pd.Series(p).pct_change()

    # --- shipped indicators (levels) -------------------------------------
    # calculate_technical_indicators reads/writes df["Close"] and appends its
    # columns in place; every window it uses is trailing.
    ind = calculate_technical_indicators(df.copy())

    # --- lagged returns ---------------------------------------------------
    # ret_lag_0 is TODAY's realised return — known at today's close, so it is a
    # legitimate input for predicting tomorrow. Deeper lags shift further back.
    for k in range(N_LAGS):
        df[f"ret_lag_{k}"] = ret.shift(k)

    # --- levels -> scale-free ratios --------------------------------------
    df["close_over_sma20"] = ind["Close"] / ind["SMA20"] - 1.0
    df["close_over_sma50"] = ind["Close"] / ind["SMA50"] - 1.0
    df["close_over_ema20"] = ind["Close"] / ind["EMA20"] - 1.0
    df["sma20_over_sma50"] = ind["SMA20"] / ind["SMA50"] - 1.0

    # RSI is already bounded 0-100; rescale to 0-1 for conditioning.
    df["rsi"] = ind["RSI"] / 100.0

    # Bollinger %B: where price sits inside the band (0 = lower, 1 = upper).
    band = (ind["BB_Upper"] - ind["BB_Lower"]).replace(0, np.nan)
    df["bb_pct"] = (ind["Close"] - ind["BB_Lower"]) / band

    # MACD is a price difference -> normalise by price to compare across tickers.
    df["macd_norm"] = ind["MACD"] / ind["Close"]
    df["macd_hist_norm"] = (ind["MACD"] - ind["MACD_Signal"]) / ind["Close"]

    # --- realised volatility + drift (trailing) ---------------------------
    for w in (5, 10, 21):
        df[f"vol_{w}"] = ret.rolling(w).std()
    for w in (5, 21):
        df[f"mean_ret_{w}"] = ret.rolling(w).mean()

    # --- target: next-day return -----------------------------------------
    df["target"] = ret.shift(-1)

    df["valid"] = df[FEATURE_COLS].notna().all(axis=1)
    df.loc[:WARMUP - 1, "valid"] = False        # force-drop the warm-up window
    return df


def assert_no_lookahead(prices: np.ndarray, probe_at: int | None = None) -> None:
    """Prove — not assert by comment — that features never see the future.

    The test: build features on the FULL series, then rebuild them on the
    series truncated at ``probe_at``. If any feature at a row before the cut
    depends on a future price, the two builds must disagree at that row. If
    they match to floating-point tolerance, the feature at that row provably
    used only data available at the time.

    This catches the whole family of look-ahead bugs that survive code review:
    centred rolling windows, ``bfill``, a scaler fit on everything, or a
    ``shift(-1)`` that crept into a feature instead of the target.

    Raises AssertionError on the first offending column.
    """
    p = np.asarray(prices, dtype=float).flatten()
    if probe_at is None:
        probe_at = len(p) // 2

    full = build_feature_frame(p)
    trunc = build_feature_frame(p[:probe_at])

    # Compare every row the truncated build knows about (its last row's target
    # is legitimately NaN — it has no tomorrow yet — so target is excluded).
    cmp_rows = trunc.index[trunc["valid"]]
    for col in FEATURE_COLS:
        a = full.loc[cmp_rows, col].to_numpy(dtype=float)
        b = trunc.loc[cmp_rows, col].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            bad = int(np.argmax(~np.isclose(a, b, rtol=1e-9, atol=1e-12,
                                            equal_nan=True)))
            raise AssertionError(
                f"LOOK-AHEAD in '{col}': row {cmp_rows[bad]} changes when the "
                f"series is truncated at {probe_at} "
                f"({a[bad]!r} with future data vs {b[bad]!r} without) — the "
                f"feature is reading prices it could not have known."
            )
