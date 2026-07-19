"""
Market-data loader — one cached, retried yfinance entry point for everything.

Why this exists
---------------
Before Day 5 the repo had THREE independent data paths: ``predictor.py``
(in-memory dict cache, 1h TTL), ``experiments/day03_bakeoff.py`` (CSV cache in
``data/eval``), and ``correlation.py`` / ``historical.py`` (no cache at all).
Three code paths means three ways to disagree about what "the close price of
AAPL on 2024-03-01" is. This module is the single source: every experiment,
the FastAPI service, and future tests pull prices through here.

Design choices
--------------
* **Disk cache, keyed by (ticker, start, end).** Walk-forward experiments are
  re-run many times while iterating; a disk cache makes them reproducible and
  keeps us polite to the free yfinance endpoint. An explicit date range (not
  ``period="1y"``) keeps cached files stable — "1y" means a different window
  every day you run it.
* **auto_adjust=True** everywhere, so splits/dividends can never differ
  between two components of the app.
* **Retries with backoff** because yfinance flakes; a failed fetch raises
  instead of returning an empty frame (never log success on failure).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE = os.path.join(ROOT, "data", "eval")

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(cache_dir: str, ticker: str, start: str, end: str, kind: str) -> str:
    return os.path.join(cache_dir, f"{kind}_{ticker}_{start}_{end}.csv")


def load_ohlcv(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | None = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Daily OHLCV for ``ticker`` over [start, end), cached to disk.

    Returns a DataFrame indexed by date with columns Open/High/Low/Close/Volume,
    sorted ascending, NaNs dropped. Raises RuntimeError if no data can be
    fetched — callers must not treat an empty result as success.
    """
    ticker = ticker.upper().strip()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = _cache_path(cache_dir, ticker, start, end, "ohlcv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            return df

    import yfinance as yf  # imported lazily so offline cache-hits need no network stack

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                out = df[OHLCV_COLS].dropna().sort_index()
                out.index = pd.to_datetime(out.index)
                out.index.name = "date"
                out = out.astype(float)
                if cache_dir:
                    out.to_csv(path)
                return out
        except Exception as e:  # noqa: BLE001 — yfinance raises many types
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(
        f"No data for '{ticker}' [{start}..{end}) after {retries} attempts"
        + (f": {last_err}" if last_err else "")
    )


def load_prices(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | None = DEFAULT_CACHE,
    retries: int = 3,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Close-price array + dates — the shape every walk-forward model consumes.

    Reads the same ``prices_*`` CSV cache the Day-2/3/4 experiments wrote, so
    re-running them through this loader is byte-identical with their history.
    """
    ticker = ticker.upper().strip()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        legacy = _cache_path(cache_dir, ticker, start, end, "prices")
        if os.path.exists(legacy):
            df = pd.read_csv(legacy, parse_dates=["date"])
            return df["close"].to_numpy(dtype=float), pd.DatetimeIndex(df["date"])

    ohlcv = load_ohlcv(ticker, start, end, cache_dir=cache_dir, retries=retries)
    if cache_dir:
        pd.DataFrame({"date": ohlcv.index, "close": ohlcv["Close"].values}).to_csv(
            _cache_path(cache_dir, ticker, start, end, "prices"), index=False)
    return ohlcv["Close"].to_numpy(dtype=float), pd.DatetimeIndex(ohlcv.index)


def clear_cache(cache_dir: str = DEFAULT_CACHE) -> int:
    """Delete cached price files; returns how many were removed."""
    if not os.path.isdir(cache_dir):
        return 0
    n = 0
    for f in os.listdir(cache_dir):
        if f.startswith(("prices_", "ohlcv_")) and f.endswith(".csv"):
            os.remove(os.path.join(cache_dir, f))
            n += 1
    return n
