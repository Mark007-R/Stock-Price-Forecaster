"""
FastAPI endpoint tests — golden paths + the guardrails the API promises.

All market data is monkeypatched to deterministic synthetic series, so the
suite runs offline and asserts on structure and invariants (not on live
prices): /backtest must refuse zero-cost runs, must always return buy-and-hold
beside the strategy, every forecast must carry the honest disclaimer, and the
Day-9 regression (xgboost_tuned 422) must stay fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.serving.api as api
from tests.conftest import make_prices


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Synthetic prices instead of yfinance; telemetry into a tmp dir; a small
    # calibration window so /predict's sequential ARIMA loop stays fast.
    def fake_load_prices(ticker, start, end, **kwargs):
        n = 400
        return make_prices(n, seed=5), pd.bdate_range("2021-01-04", periods=n)

    monkeypatch.setattr(api, "load_prices", fake_load_prices)
    monkeypatch.setattr(api, "CALIBRATION_DAYS", 40)
    monkeypatch.setattr(api, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(api, "TELEMETRY_PATH", tmp_path / "requests.jsonl")
    return TestClient(api.app)


class TestHealth:
    def test_reports_models_champion_and_cache_status(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["champion"] == "arima"
        assert "persistence" in body["models"]
        assert body["price_cache_redis"] in ("disabled", "connected", "unavailable")


class TestPredict:
    def test_golden_path_returns_bands_and_disclaimer(self, client):
        r = client.post("/predict", json={"ticker": "AAPL", "horizon": 5})
        assert r.status_code == 200
        body = r.json()
        assert len(body["pred_prices"]) == 5
        assert body["model"].startswith("arima")
        # 95% band contains the 80% band contains the point path.
        for k in range(5):
            assert (body["lower95"][k] <= body["lower80"][k]
                    <= body["pred_prices"][k]
                    <= body["upper80"][k] <= body["upper95"][k])
        # The honest-evaluation notice is not optional.
        assert "buy-and-hold" in body["disclaimer"]

    def test_bands_widen_with_horizon(self, client):
        body = client.post("/predict", json={"ticker": "AAPL", "horizon": 10}).json()
        widths = [u - l for u, l in zip(body["upper80"], body["lower80"])]
        assert widths[-1] > widths[0]

    def test_rejects_horizon_beyond_schema(self, client):
        assert client.post("/predict",
                           json={"ticker": "AAPL", "horizon": 31}).status_code == 422

    def test_rejects_malformed_ticker(self, client):
        assert client.post("/predict",
                           json={"ticker": "AA PL; DROP", "horizon": 5}).status_code == 422


class TestBacktest:
    def test_golden_path_always_reports_buy_and_hold(self, client):
        r = client.post("/backtest", json={
            "ticker": "AAPL", "model": "persistence", "n_folds": 3})
        assert r.status_code == 200
        body = r.json()
        assert "buy_and_hold" in body and "strategy" in body
        assert "beats_buy_and_hold_sharpe" in body
        # persistence forecasts no edge -> takes no positions -> zero return.
        assert body["strategy"]["total_return"] == 0.0
        assert body["strategy"]["n_trades"] == 0

    def test_zero_cost_backtests_are_refused_by_schema(self, client):
        r = client.post("/backtest", json={
            "ticker": "AAPL", "model": "persistence", "cost_bps": 0.0})
        assert r.status_code == 422, "a backtest without costs is theater"

    def test_unknown_model_is_a_422_with_the_valid_names(self, client):
        r = client.post("/backtest", json={"ticker": "AAPL", "model": "oracle"})
        assert r.status_code == 422
        assert "persistence" in r.json()["detail"]

    def test_day9_regression_xgboost_tuned_gets_its_features(self, client):
        # Fixed on Day 9: `with_features` was gated on == "xgboost", so the
        # tuned variant 422'd for 17 days. Never again.
        r = client.post("/backtest", json={
            "ticker": "AAPL", "model": "xgboost_tuned", "n_folds": 2})
        assert r.status_code == 200, r.json()
        assert r.json()["oos_days"] > 0

    def test_oos_days_equal_the_sum_of_fold_test_spans(self, client):
        body = client.post("/backtest", json={
            "ticker": "AAPL", "model": "momentum", "n_folds": 4}).json()
        # 400 samples -> default test_size (400//2)//4 = 50 per fold.
        assert body["oos_days"] == 200
        assert body["n_folds"] == 4


class TestIndicators:
    def test_serves_the_flask_apps_indicator_math(self, client, monkeypatch):
        n = 300
        prices = make_prices(n, seed=9)
        df = pd.DataFrame(
            {"Open": prices, "High": prices * 1.01, "Low": prices * 0.99,
             "Close": prices, "Volume": np.full(n, 1e6)},
            index=pd.bdate_range("2021-01-04", periods=n))
        monkeypatch.setattr(api, "load_ohlcv", lambda *a, **k: df)

        body = client.get("/indicators/AAPL").json()
        assert body["ticker"] == "AAPL"
        assert body["n_days"] == n
        assert 0.0 <= body["rsi14"] <= 100.0
        assert body["bb_lower"] <= body["close"] * 1.05


class TestCorrelation:
    def test_needs_at_least_two_valid_tickers(self, client, monkeypatch):
        def one_good(ticker, start, end, **kw):
            if ticker != "AAPL":
                raise RuntimeError("no data")
            return make_prices(200), pd.bdate_range("2021-01-04", periods=200)
        monkeypatch.setattr(api, "load_prices", one_good)
        r = client.get("/correlation", params={"tickers": "AAPL,NOPE"})
        assert r.status_code == 422

    def test_matrix_is_symmetric_with_unit_diagonal(self, client, monkeypatch):
        def two_series(ticker, start, end, **kw):
            seed = {"AAPL": 1, "MSFT": 2}.get(ticker, 3)
            return make_prices(200, seed=seed), pd.bdate_range("2021-01-04", periods=200)
        monkeypatch.setattr(api, "load_prices", two_series)
        body = client.get("/correlation", params={"tickers": "AAPL,MSFT"}).json()
        m = body["matrix"]
        assert m["AAPL"]["AAPL"] == pytest.approx(1.0)
        assert m["AAPL"]["MSFT"] == m["MSFT"]["AAPL"]
