"""
Shared fixtures for the StockAI test suite.

Every test runs OFFLINE: prices are synthetic geometric random walks with a
fixed seed, or pre-written CSVs in a tmp cache dir. No test may hit yfinance —
a suite that needs the network is a suite that flakes in CI and in interviews.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root on sys.path so `src.*` and root modules (historical.py) import the
# same way the experiments and the API import them.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_prices(n: int = 500, seed: int = 7, drift: float = 0.0004,
                vol: float = 0.012, p0: float = 100.0) -> np.ndarray:
    """Deterministic geometric random walk — positive prices, realistic vol."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, size=n - 1)
    return p0 * np.concatenate([[1.0], np.cumprod(1.0 + ret)])


@pytest.fixture
def prices() -> np.ndarray:
    return make_prices(500)


@pytest.fixture
def long_prices() -> np.ndarray:
    """Long enough to warm up the extended (252-day) feature windows."""
    return make_prices(450, seed=11)


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", periods=500)


@pytest.fixture
def long_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", periods=450)
