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

# ── Day 4 extension: calendar + regime features and multi-horizon targets ────
# Day 3's XGBoost split its gain almost perfectly uniformly across the 18
# base features — the fingerprint of trees finding no signal. Two objections
# survive that result: (1) maybe the features were the wrong *kind* — no
# calendar or regime context — and (2) maybe daily returns are simply the
# wrong *horizon* and signal lives further out. Day 4 adds the features those
# objections ask for and re-scores at 1-, 5-, and 20-day horizons.
#
# Every added column obeys the same two rules as the base set: backward-looking
# only, and stationary/scale-free. Calendar columns are deterministic functions
# of the date at row t (a Tuesday in March is knowable at the close of that
# day). Regime columns are trailing rolling statistics — nothing is centred,
# nothing peeks forward. ``assert_no_lookahead(..., extended=True)`` proves it.

# Calendar seasonality — encoded cyclically so Friday(4)→Monday(0) is a small
# step, not a jump from 4 to 0, and December→January likewise wraps smoothly.
CALENDAR_COLS = ["dow_sin", "dow_cos", "month_sin", "month_cos", "turn_of_month"]

# Market-state / regime context — all trailing, all scale-free.
REGIME_COLS = ["trend_up", "vol_regime_high", "drawdown_63",
               "dist_high_252", "mom_63", "vol_63"]

EXTENDED_FEATURE_COLS = FEATURE_COLS + CALENDAR_COLS + REGIME_COLS

# The longest trailing window in the extended set is 252 days (one trading
# year: the 52-week high and the volatility-regime median). The warm-up must
# cover it or those columns leak a partially-formed window into early rows.
WARMUP_EXT = 260


def build_feature_frame(prices: np.ndarray, dates=None, extended: bool = False,
                        horizons=(1,)) -> pd.DataFrame:
    """Build the feature matrix + forward-return target(s) from a close series.

    Parameters
    ----------
    prices : array of daily closes, oldest first.
    dates  : optional DatetimeIndex, carried through for Prophet / reporting.
             REQUIRED when ``extended=True`` (calendar features need it).
    extended : if True, also append ``CALENDAR_COLS`` + ``REGIME_COLS`` and a
             ``valid_ext`` mask warmed up over ``WARMUP_EXT``. Default False so
             every Day 1–3 caller keeps its exact previous output.
    horizons : iterable of forecast horizons (in trading days) to build targets
             for. For each ``h`` a ``target_h{h}`` column holds the cumulative
             forward return ``p[t+h]/p[t]-1``. ``target`` (the 1-day return) is
             always present for backward compatibility.

    Returns
    -------
    DataFrame indexed 0..n-1 (aligned to ``prices``) with ``FEATURE_COLS``, a
    ``target`` column (next-day return), and ``valid`` marking rows whose base
    features are fully warmed up. Row ``t``'s features use prices[:t+1] only;
    ``target[t]`` is the return realised on day t+1 (NaN on the final row,
    which has no tomorrow). With ``extended=True`` the frame also carries the
    calendar/regime columns and a ``valid_ext`` mask.
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

    # --- targets: multi-horizon forward returns --------------------------
    # target_h{h}[t] = p[t+h]/p[t] - 1, realised strictly AFTER the close of t.
    # ``target`` is kept as the 1-day return under its original name so every
    # Day 1–3 caller is untouched. Horizons >1 overlap between consecutive rows
    # (row t and t+1 share h-1 days); callers must treat the effective sample as
    # smaller than the row count — see day04_features.py.
    p_ser = pd.Series(p)
    df["target"] = ret.shift(-1)
    for h in horizons:
        df[f"target_h{h}"] = p_ser.shift(-h) / p_ser - 1.0

    df["valid"] = df[FEATURE_COLS].notna().all(axis=1)
    df.loc[:WARMUP - 1, "valid"] = False        # force-drop the warm-up window

    if extended:
        if dates is None:
            raise ValueError("extended=True requires `dates` for calendar features")
        _add_extended_features(df, ind, ret, p)
        df["valid_ext"] = df[EXTENDED_FEATURE_COLS].notna().all(axis=1)
        df.loc[:WARMUP_EXT - 1, "valid_ext"] = False
    return df


def _add_extended_features(df: pd.DataFrame, ind: pd.DataFrame,
                           ret: pd.Series, p: np.ndarray) -> None:
    """Append calendar + regime columns in place. All causal (see module doc).

    ``ind`` is the shipped-indicator frame (levels), ``ret`` the trailing
    return series, ``p`` the raw close array. Mutates ``df``.
    """
    dts = pd.to_datetime(df["date"])
    p_ser = pd.Series(p)

    # --- calendar seasonality (deterministic from the date at row t) ------
    dow = dts.dt.dayofweek                       # 0=Mon .. 4=Fri
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    month = dts.dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    # Turn-of-month effect: the first 3 / last 2 trading-ish days of the month,
    # where a well-documented (if small) equity drift concentrates.
    dom = dts.dt.day
    dim = dts.dt.days_in_month
    df["turn_of_month"] = ((dom <= 3) | (dom >= dim - 2)).astype(float)

    # --- regime context (trailing, scale-free) ----------------------------
    # Trend state: is the short average above the long? A 0/1 regime flag, the
    # classic golden/death-cross partition. NaN during warm-up -> comparison
    # yields False there, but those rows are force-dropped by valid_ext.
    df["trend_up"] = (ind["SMA20"] > ind["SMA50"]).astype(float)

    # Volatility regime: is current 21-day vol above its own trailing 1-year
    # median? A relative (per-ticker, per-era) high/low-vol flag that does not
    # bake in any absolute vol level.
    vol_med = df["vol_21"].rolling(252, min_periods=60).median()
    df["vol_regime_high"] = (df["vol_21"] > vol_med).astype(float)

    # Drawdown from the trailing 63-day (≈ quarter) high: 0 at a new high,
    # negative below it. A bounded, scale-free proxy for "how stressed".
    df["drawdown_63"] = p_ser / p_ser.rolling(63, min_periods=20).max() - 1.0

    # Distance below the trailing 252-day (52-week) high — the long-horizon
    # analogue, a feature momentum strategies lean on.
    df["dist_high_252"] = p_ser / p_ser.rolling(252, min_periods=60).max() - 1.0

    # Medium-term momentum + volatility the base set (which tops out at 21
    # days) does not carry.
    df["mom_63"] = p_ser.pct_change(63)
    df["vol_63"] = ret.rolling(63).std()


def assert_no_lookahead(prices: np.ndarray, probe_at: int | None = None,
                        dates=None, extended: bool = False) -> None:
    """Prove — not assert by comment — that features never see the future.

    The test: build features on the FULL series, then rebuild them on the
    series truncated at ``probe_at``. If any feature at a row before the cut
    depends on a future price, the two builds must disagree at that row. If
    they match to floating-point tolerance, the feature at that row provably
    used only data available at the time.

    This catches the whole family of look-ahead bugs that survive code review:
    centred rolling windows, ``bfill``, a scaler fit on everything, or a
    ``shift(-1)`` that crept into a feature instead of the target.

    With ``extended=True`` the same proof is run over the calendar + regime
    columns (``dates`` is then required, and is truncated alongside prices so
    the calendar features are rebuilt from the same days).

    Raises AssertionError on the first offending column.
    """
    p = np.asarray(prices, dtype=float).flatten()
    if probe_at is None:
        probe_at = len(p) // 2

    cols = EXTENDED_FEATURE_COLS if extended else FEATURE_COLS
    valid_col = "valid_ext" if extended else "valid"

    dt_full = None if dates is None else pd.DatetimeIndex(pd.Series(dates).values[:len(p)])
    dt_trunc = None if dt_full is None else dt_full[:probe_at]

    full = build_feature_frame(p, dt_full, extended=extended)
    trunc = build_feature_frame(p[:probe_at], dt_trunc, extended=extended)

    # Compare every row the truncated build knows about (its last row's target
    # is legitimately NaN — it has no tomorrow yet — so target is excluded).
    cmp_rows = trunc.index[trunc[valid_col]]
    for col in cols:
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
