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
* **Optional Redis in front of the disk cache (Day 9).** A single process is
  served perfectly well by the local CSV cache; Redis exists for the Docker
  deployment, where the API and the dashboard are separate containers with
  separate filesystems and would otherwise each hit yfinance for the same
  series. Redis is opt-in via ``REDIS_URL`` and degrades silently to the
  disk-then-network path — an unreachable cache must never take the service
  down with it.
"""
from __future__ import annotations

import io
import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE = os.path.join(ROOT, "data", "eval")

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Historical daily bars for an explicit [start, end) range are effectively
# immutable, but ranges whose end is still in the future fill in as days pass —
# a finite TTL lets those refresh instead of freezing at first fetch.
REDIS_TTL_SECONDS = 7 * 24 * 3600

_redis_client = None
_redis_failed = False   # one bad connection disables Redis for this process


def _redis():
    """Redis client from ``REDIS_URL``, or None (unset / previously failed).

    The client is created lazily and verified with a ping so that a container
    started before its Redis dependency simply falls back to disk instead of
    erroring on every request.
    """
    global _redis_client, _redis_failed
    url = os.environ.get("REDIS_URL")
    if not url or _redis_failed:
        return None
    if _redis_client is None:
        try:
            import redis  # imported lazily — not required outside Docker
            _redis_client = redis.Redis.from_url(
                url, socket_connect_timeout=2, socket_timeout=2)
            _redis_client.ping()
            logger.info("price cache: Redis connected (%s)", url)
        except Exception as e:  # noqa: BLE001 — any failure means "no Redis"
            logger.warning("price cache: Redis unavailable (%s) — disk only", e)
            _redis_client, _redis_failed = None, True
            return None
    return _redis_client


def redis_status() -> str:
    """'connected' / 'unavailable' / 'disabled' — surfaced by /health."""
    if not os.environ.get("REDIS_URL"):
        return "disabled"
    return "connected" if _redis() is not None else "unavailable"


def _redis_get(key: str) -> pd.DataFrame | None:
    r = _redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
    except Exception:  # noqa: BLE001
        return None
    if raw is None:
        return None
    return pd.read_csv(io.BytesIO(raw), parse_dates=["date"], index_col="date")


def _redis_put(key: str, df: pd.DataFrame) -> None:
    r = _redis()
    if r is None:
        return
    try:
        buf = io.StringIO()
        df.to_csv(buf)
        r.set(key, buf.getvalue().encode(), ex=REDIS_TTL_SECONDS)
    except Exception:  # noqa: BLE001 — cache writes are best-effort
        pass


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
    rkey = f"stockai:ohlcv:{ticker}:{start}:{end}"

    cached = _redis_get(rkey)
    if cached is not None:
        return cached

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = _cache_path(cache_dir, ticker, start, end, "ohlcv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            _redis_put(rkey, df)   # promote, so sibling containers skip yfinance
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
                _redis_put(rkey, out)
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
