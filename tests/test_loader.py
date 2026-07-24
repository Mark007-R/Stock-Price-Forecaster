"""
Tests for the unified market-data loader (src/data/loader.py) — offline only.

The loader's contract: cache-first, never-silently-empty, Redis strictly
optional. All tests run against tmp-dir CSV caches; none may touch yfinance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import clear_cache, load_ohlcv, load_prices, redis_status


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path)


def _write_prices_csv(cache_dir, ticker="TEST", start="2021-01-01", end="2022-01-01"):
    dates = pd.bdate_range(start, periods=100)
    closes = np.linspace(100.0, 120.0, 100)
    pd.DataFrame({"date": dates, "close": closes}).to_csv(
        f"{cache_dir}/prices_{ticker}_{start}_{end}.csv", index=False)
    return closes, dates


class TestDiskCache:
    def test_load_prices_reads_cache_without_network(self, cache_dir):
        closes, dates = _write_prices_csv(cache_dir)
        prices, idx = load_prices("TEST", "2021-01-01", "2022-01-01",
                                  cache_dir=cache_dir, retries=0)
        np.testing.assert_allclose(prices, closes)
        assert len(idx) == 100

    def test_ticker_is_normalised_to_uppercase(self, cache_dir):
        _write_prices_csv(cache_dir, ticker="TEST")
        prices, _ = load_prices("  test ", "2021-01-01", "2022-01-01",
                                cache_dir=cache_dir, retries=0)
        assert len(prices) == 100

    def test_missing_data_raises_instead_of_returning_empty(self, cache_dir, monkeypatch):
        # Never log success on failure: an unfetchable ticker must raise.
        import yfinance as yf
        monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())
        with pytest.raises(RuntimeError, match="No data"):
            load_ohlcv("NOPE", "2021-01-01", "2022-01-01",
                       cache_dir=cache_dir, retries=1)

    def test_clear_cache_counts_removals(self, cache_dir):
        _write_prices_csv(cache_dir, ticker="A")
        _write_prices_csv(cache_dir, ticker="B")
        assert clear_cache(cache_dir) == 2
        assert clear_cache(cache_dir) == 0


class TestRedisTier:
    def test_disabled_without_redis_url(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        assert redis_status() == "disabled"
