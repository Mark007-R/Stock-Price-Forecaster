"""
FastAPI serving layer — the walk-forward pipeline behind four endpoints.

Runs SEPARATELY from the Flask UI (port 8000 vs 5000): Flask keeps serving the
templates it always served; this service exposes the honest pipeline —
champion forecasts with conformal intervals, walk-forward backtests with
transaction costs and a buy-and-hold benchmark — as validated JSON.

Design stances (all downstream of Days 1–4):
* **/predict serves ARIMA**, the Day-3 champion among forecasting models, and
  every response carries the disclaimer field: the champion did NOT beat
  buy-and-hold net of costs. An API that hides that is marketing.
* **Uncertainty is conformal, not Gaussian**: intervals come from the model's
  own recent 1-step errors (split-conformal, distribution-free) — the same
  machinery that replaced predictor.py's volatility-only "confidence".
* **/backtest refuses to run without costs** (cost_bps > 0 enforced by the
  schema) and always returns buy-and-hold beside the strategy. A backtest
  without a baseline is theater.
* Endpoints are async; the numeric work is offloaded to the threadpool so the
  event loop never blocks on an ARIMA fit.

Day 9 (production wrapper) additions:
* **Per-request telemetry** — one JSON line per request (path, status,
  latency ms) appended to ``logs/requests.jsonl``; the ops dashboard tails it.
* **/backtest can record itself to MLflow** (``log_mlflow: true``) via the
  same ``src.tracking`` machinery the CLI uses, so API-triggered runs and
  scripted runs land in one file store.
* **/health reports the Redis price-cache status** — "disabled" outside
  Docker is the expected, correct answer, not a failure.

Run:  uvicorn src.serving.api:app --port 8000
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_prices, load_ohlcv, redis_status       # noqa: E402
from src.models import MODELS, CHAMPION, get_model, make_context        # noqa: E402
from src.models.arima import forecast_returns                            # noqa: E402
from src.models.intervals import conformal_halfwidth                     # noqa: E402
from src.backtest.walkforward import (                                   # noqa: E402
    expanding_window_folds, assert_no_peeking, walk_forward_predict,
)
from src.backtest.trading import backtest_long_flat, backtest_buy_and_hold  # noqa: E402

DEFAULT_START, DEFAULT_END = "2021-01-01", "2025-01-01"
CALIBRATION_DAYS = 120           # recent 1-step errors used for conformal bands
DISCLAIMER = (
    "Honest-evaluation notice: across a 10-ticker walk-forward with 5 bps/side "
    "costs (Days 2-4), no forecasting model in this repo beat buy-and-hold "
    "(best: ARIMA Sharpe 1.34 vs 1.83). Point forecasts are near the random-"
    "walk floor; the intervals are the informative part of this response."
)

app = FastAPI(
    title="StockAI honest forecasting API",
    version="0.9.0",
    description="Walk-forward-evaluated forecasts with conformal intervals, "
                "cost-aware backtests, indicators and correlations.",
)

# ─────────────────────────────────────────────────────────────────────────────
# Per-request telemetry — append-only JSONL, tailed by the ops dashboard.
# ─────────────────────────────────────────────────────────────────────────────
TELEMETRY_DIR = Path(os.environ.get("STOCKAI_LOG_DIR", ROOT / "logs"))
TELEMETRY_PATH = TELEMETRY_DIR / "requests.jsonl"


@app.middleware("http")
async def telemetry(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        try:
            TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
            with open(TELEMETRY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                }) + "\n")
        except OSError:
            pass   # telemetry must never take a request down with it


# ─────────────────────────────────────────────────────────────────────────────
# Schemas (Pydantic v2)
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.\-]+$")
    horizon: int = Field(5, ge=1, le=30, description="Trading days ahead")
    start: str = Field(DEFAULT_START, description="History start (YYYY-MM-DD)")
    end: str = Field(DEFAULT_END, description="History end, exclusive")


class PredictResponse(BaseModel):
    ticker: str
    model: str
    last_price: float
    last_date: str
    horizon: int
    pred_returns: list[float]
    pred_prices: list[float]
    lower80: list[float]
    upper80: list[float]
    lower95: list[float]
    upper95: list[float]
    interval_method: str
    calibration_days: int
    disclaimer: str


class BacktestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.\-]+$")
    model: str = Field(CHAMPION, description=f"One of {sorted(MODELS)}")
    n_folds: int = Field(5, ge=2, le=10)
    cost_bps: float = Field(5.0, gt=0.0, le=100.0,
                            description="Per-side costs; zero-cost backtests are refused")
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    log_mlflow: bool = Field(
        False, description="Record this run (params/metrics/artifacts) to MLflow")


class BacktestResponse(BaseModel):
    ticker: str
    model: str
    n_folds: int
    cost_bps: float
    oos_days: int
    rmse_ret_mean: float
    dir_acc_mean: float | None
    strategy: dict
    buy_and_hold: dict
    beats_buy_and_hold_sharpe: bool
    mlflow_run_id: str | None = None
    disclaimer: str


# ─────────────────────────────────────────────────────────────────────────────
# Blocking cores (executed via threadpool)
# ─────────────────────────────────────────────────────────────────────────────
def _predict_core(req: PredictRequest) -> dict:
    prices, dates = load_prices(req.ticker, req.start, req.end)
    if len(prices) < CALIBRATION_DAYS + 60:
        raise ValueError(
            f"Need >= {CALIBRATION_DAYS + 60} days of history, got {len(prices)}")

    ret = prices[1:] / prices[:-1] - 1.0

    # Calibration residuals: the champion's actual recent 1-step errors.
    # ARIMA's own forecasts hug zero, so its residuals ≈ realised returns —
    # we still compute them properly (error = forecast − realised) so the
    # bands stay honest for any future champion that forecasts more.
    calib_train_ret = ret[:-CALIBRATION_DAYS]
    calib_actual = ret[-CALIBRATION_DAYS:]
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    from src.models.arima import select_order
    order = select_order(calib_train_ret)
    res = _ARIMA(calib_train_ret, order=order).fit()
    calib_pred = []
    for k in range(CALIBRATION_DAYS):
        calib_pred.append(float(res.forecast(steps=1)[0]))
        res = res.append([calib_actual[k]], refit=False)
    residuals = np.asarray(calib_pred) - calib_actual

    q80 = conformal_halfwidth(residuals, alpha=0.20)
    q95 = conformal_halfwidth(residuals, alpha=0.05)

    pred_ret = forecast_returns(prices, req.horizon)
    last_price = float(prices[-1])
    pred_prices = last_price * np.cumprod(1.0 + pred_ret)
    h = np.arange(1, req.horizon + 1)
    # Return-space conformal band, √h-widened, compounded onto the price path.
    lo80 = pred_prices * (1.0 - q80 * np.sqrt(h))
    hi80 = pred_prices * (1.0 + q80 * np.sqrt(h))
    lo95 = pred_prices * (1.0 - q95 * np.sqrt(h))
    hi95 = pred_prices * (1.0 + q95 * np.sqrt(h))

    r4 = lambda a: [round(float(v), 4) for v in a]  # noqa: E731
    return {
        "ticker": req.ticker.upper(),
        "model": f"arima{order}".replace(" ", ""),
        "last_price": round(last_price, 4),
        "last_date": str(dates[-1].date()),
        "horizon": req.horizon,
        "pred_returns": [round(float(v), 6) for v in pred_ret],
        "pred_prices": r4(pred_prices),
        "lower80": r4(lo80), "upper80": r4(hi80),
        "lower95": r4(lo95), "upper95": r4(hi95),
        "interval_method": "split-conformal on last "
                           f"{CALIBRATION_DAYS} one-step errors, sqrt(h)-widened",
        "calibration_days": CALIBRATION_DAYS,
        "disclaimer": DISCLAIMER,
    }


def _backtest_core(req: BacktestRequest) -> dict:
    model_fn = get_model(req.model)
    prices, dates = load_prices(req.ticker, req.start, req.end)

    folds = expanding_window_folds(len(prices), n_folds=req.n_folds)
    assert_no_peeking(folds, n_samples=len(prices))
    # startswith, not ==: "xgboost_tuned" (Day 6) needs the feature frame too —
    # the exact-match version 422'd every tuned-model backtest.
    ctx = make_context(prices, dates,
                       with_features=req.model.startswith("xgboost"))

    fold_results = walk_forward_predict(
        prices, lambda p, f: model_fn(p, f, ctx), folds)

    pred_all = np.concatenate([r["pred_ret"] for r in fold_results])
    actual_all = np.concatenate([r["actual_ret"] for r in fold_results])

    bt = backtest_long_flat(pred_all, actual_all, cost_bps=req.cost_bps)
    bh = backtest_buy_and_hold(actual_all, cost_bps=req.cost_bps)

    dir_accs = [r["dir_acc"] for r in fold_results if not np.isnan(r["dir_acc"])]

    # Optional MLflow record. It re-runs the same deterministic pipeline via
    # src.tracking (identical folds/costs), so the logged run matches this
    # response; the small duplicate compute buys one code path for CLI + API.
    mlflow_run_id = None
    if req.log_mlflow:
        from src.tracking.mlflow_runs import track_walkforward_run
        summary = track_walkforward_run(
            req.ticker, req.model, n_folds=req.n_folds,
            cost_bps=req.cost_bps, start=req.start, end=req.end)
        mlflow_run_id = summary["run_id"]

    return {
        "ticker": req.ticker.upper(),
        "model": req.model,
        "n_folds": len(fold_results),
        "cost_bps": req.cost_bps,
        "oos_days": int(len(actual_all)),
        "rmse_ret_mean": round(float(np.mean([r["rmse_ret"] for r in fold_results])), 6),
        "dir_acc_mean": round(float(np.mean(dir_accs)), 4) if dir_accs else None,
        "strategy": {k: round(float(v), 6) for k, v in bt.as_dict().items()},
        "buy_and_hold": {k: round(float(v), 6) for k, v in bh.as_dict().items()},
        "beats_buy_and_hold_sharpe": bool(bt.sharpe > bh.sharpe),
        "mlflow_run_id": mlflow_run_id,
        "disclaimer": DISCLAIMER,
    }


def _indicators_core(ticker: str, start: str, end: str) -> dict:
    # Reuse the Flask app's own indicator math — one implementation, two servers.
    from historical import calculate_technical_indicators

    df = load_ohlcv(ticker, start, end)
    df = calculate_technical_indicators(df.copy())
    last = df.iloc[-1]

    def val(col):
        v = last.get(col)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)

    returns = df["Close"].pct_change().dropna()
    sharpe = (float(returns.mean()) / float(returns.std()) * np.sqrt(252)
              if len(returns) > 1 and returns.std() > 0 else 0.0)
    return {
        "ticker": ticker.upper(),
        "as_of": str(df.index[-1].date()),
        "n_days": int(len(df)),
        "close": val("Close"),
        "sma20": val("SMA20"), "sma50": val("SMA50"), "ema20": val("EMA20"),
        "rsi14": val("RSI"),
        "bb_upper": val("BB_Upper"), "bb_middle": val("BB_Middle"),
        "bb_lower": val("BB_Lower"),
        "macd": val("MACD"), "macd_signal": val("MACD_Signal"),
        "daily_return_pct": val("Daily_Return"),
        "annualised_sharpe": round(sharpe, 4),
    }


def _correlation_core(tickers: list[str], start: str, end: str) -> dict:
    import pandas as pd

    closes, failed = {}, []
    for tk in tickers:
        try:
            p, d = load_prices(tk, start, end)
            closes[tk.upper()] = pd.Series(p, index=d)
        except Exception:  # noqa: BLE001
            failed.append(tk.upper())
    frame = pd.DataFrame(closes).dropna()
    if frame.shape[1] < 2 or frame.empty:
        raise ValueError(
            f"Need >= 2 tickers with overlapping data (failed: {failed or 'none'})")

    corr = frame.corr()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({"ticker1": cols[i], "ticker2": cols[j],
                          "correlation": round(float(corr.iloc[i, j]), 4)})
    pairs.sort(key=lambda p: -abs(p["correlation"]))
    return {
        "tickers": cols,
        "failed": failed,
        "n_days": int(len(frame)),
        "matrix": {c: {k: round(float(corr.loc[c, k]), 4) for k in cols} for c in cols},
        "top_pairs": pairs[:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "models": sorted(MODELS),
        "champion": CHAMPION,
        "price_cache_redis": redis_status(),
        "telemetry": str(TELEMETRY_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    try:
        return PredictResponse(**await run_in_threadpool(_predict_core, req))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/backtest", response_model=BacktestResponse)
async def backtest(req: BacktestRequest) -> BacktestResponse:
    try:
        return BacktestResponse(**await run_in_threadpool(_backtest_core, req))
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e.args[0]))
    except ImportError as e:
        # The slim serving image ships without TensorFlow on purpose (the
        # champion is ARIMA). Say so, instead of a bare 500.
        raise HTTPException(
            status_code=501,
            detail=f"Model '{req.model}' needs an optional dependency not in "
                   f"this serving image ({e.name}). Deep models are research-"
                   "only here — use the full requirements.txt environment.")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/indicators/{ticker}")
async def indicators(
    ticker: str,
    start: str = Query(DEFAULT_START),
    end: str = Query(DEFAULT_END),
) -> dict:
    try:
        return await run_in_threadpool(_indicators_core, ticker, start, end)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/correlation")
async def correlation(
    tickers: str = Query(..., description="Comma-separated, e.g. AAPL,MSFT,SPY"),
    start: str = Query(DEFAULT_START),
    end: str = Query(DEFAULT_END),
) -> dict:
    names = [t.strip() for t in tickers.split(",") if t.strip()]
    try:
        return await run_in_threadpool(_correlation_core, names, start, end)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
