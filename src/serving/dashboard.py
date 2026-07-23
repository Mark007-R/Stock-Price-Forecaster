"""
Ops dashboard — the sprint's honest-evaluation story, live on one page.

This is NOT another prediction UI (Flask on :5000 and stock_predictor.py
already do that). It is the operator's view of the pipeline built on Days 1–8:

* the Day-1 leakage fix, shown as before/after RMSE per ticker — the reason
  every number after it can be trusted;
* a walk-forward explorer: pick ticker × model, get the cost-aware equity
  curve vs buy-and-hold and per-fold directional accuracy vs always-up,
  computed live through the exact ``src`` modules the API serves;
* conformal-interval coverage from Day 5 (nominal vs empirical);
* the MLflow run table (Day 9) and the API's request telemetry tail.

Data honesty: every chart is either computed through the same walk-forward
code path the API uses, or read from the append-only ``results/`` CSVs the
daily reports cite. Nothing here is a mock.

Run:  streamlit run src/serving/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_prices, redis_status                    # noqa: E402
from src.models import MODELS, CHAMPION, get_model, make_context         # noqa: E402
from src.backtest.walkforward import (                                   # noqa: E402
    expanding_window_folds, walk_forward_predict,
)
from src.backtest.trading import backtest_long_flat, backtest_buy_and_hold  # noqa: E402

RESULTS = ROOT / "results"
LOGS = ROOT / "logs" / "requests.jsonl"
TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "KO"]
# LSTM/PatchTST need TensorFlow + minutes of fit time — wrong tool for a
# dashboard request; the CSVs from Days 2/7 already tell their story.
FAST_MODELS = ["arima", "xgboost", "xgboost_tuned", "momentum", "persistence"]
START, END = "2021-01-01", "2025-01-01"
COST_BPS = 5.0

st.set_page_config(page_title="StockAI ops dashboard", layout="wide")

st.title("StockAI — honest-evaluation ops dashboard")
st.caption(
    "Walk-forward, cost-aware, benchmark-first. Headline finding of this sprint: "
    "**no model in this repo beats buy-and-hold net of costs** (best forecaster: "
    "ARIMA, Sharpe 1.34 vs 1.83). This dashboard exists to keep that comparison "
    "in view, not to hide it."
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner="Running walk-forward backtest…")
def run_walkforward(ticker: str, model_name: str) -> dict:
    """Same pipeline as the API's /backtest — folds, costs and all."""
    prices, dates = load_prices(ticker, START, END)
    folds = expanding_window_folds(len(prices), n_folds=5)
    ctx = make_context(prices, dates,
                       with_features=model_name.startswith("xgboost"))
    model_fn = get_model(model_name)
    fold_results = walk_forward_predict(
        prices, lambda p, f: model_fn(p, f, ctx), folds)

    pred = np.concatenate([r["pred_ret"] for r in fold_results])
    actual = np.concatenate([r["actual_ret"] for r in fold_results])
    test_start = folds[0].test_start
    oos_dates = pd.DatetimeIndex(dates[test_start:folds[-1].test_end])

    cost = COST_BPS / 10_000.0
    position = (pred > 0).astype(float)
    prev = np.concatenate([[0.0], position[:-1]])
    net = position * actual - np.abs(position - prev) * cost
    bh_net = actual.copy()
    bh_net[0] -= cost

    bt = backtest_long_flat(pred, actual, cost_bps=COST_BPS)
    bh = backtest_buy_and_hold(actual, cost_bps=COST_BPS)
    return {
        "dates": oos_dates,
        "equity_strat": np.cumprod(1 + net),
        "equity_bh": np.cumprod(1 + bh_net),
        "folds": [{k: r[k] for k in ("fold", "train_days", "test_days",
                                     "rmse_ret", "dir_acc")} for r in fold_results],
        "always_up": float(np.mean(actual > 0)),
        "strat": bt.as_dict(),
        "bh": bh.as_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1 · The leakage fix (Day 1) — why the numbers are trustworthy
# ─────────────────────────────────────────────────────────────────────────────
st.header("1 · Scaler-leakage fix — before vs after (Day 1)")
leak = load_csv("phase1_leakage_comparison.csv")
if leak is None:
    st.info("results/phase1_leakage_comparison.csv not found.")
else:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_bar(name="leaky scaler (fit on full series)",
                    x=leak["ticker"], y=leak["rmse_leaky"])
        fig.add_bar(name="fixed scaler (fit on train only)",
                    x=leak["ticker"], y=leak["rmse_fixed"])
        fig.update_layout(barmode="group", height=360,
                          yaxis_title="LSTM price RMSE ($)",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Median RMSE inflation hidden by the leak",
                  f"{leak['rmse_inflation_pct'].median():.1f}%")
        st.metric("Tickers where LSTM beat persistence (after fix)",
                  f"{int(leak['lstm_beats_persistence_rmse'].sum())} / {len(leak)}")
        st.markdown(
            "Fitting `MinMaxScaler` on the full series let the test window's "
            "min/max leak into training. The fix (fit on the train slice only) "
            "is the first commit of this sprint — every chart below inherits it."
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Walk-forward explorer — equity curve vs buy-and-hold
# ─────────────────────────────────────────────────────────────────────────────
st.header("2 · Walk-forward explorer (live, cost-aware)")
cc1, cc2 = st.columns(2)
ticker = cc1.selectbox("Ticker", TICKERS, index=0)
model_name = cc2.selectbox(
    "Model", FAST_MODELS, index=0,
    format_func=lambda m: f"{m} — {MODELS[m][1]}" if m in MODELS else m)

wf = run_walkforward(ticker, model_name)

fig = go.Figure()
fig.add_scatter(x=wf["dates"], y=wf["equity_strat"], name="strategy (net of costs)")
fig.add_scatter(x=wf["dates"], y=wf["equity_bh"], name="buy & hold (net)",
                line=dict(dash="dash"))
fig.update_layout(height=380, yaxis_title="equity (start = 1.0)",
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Strategy Sharpe (net)", f"{wf['strat']['sharpe']:.2f}",
          delta=f"{wf['strat']['sharpe'] - wf['bh']['sharpe']:+.2f} vs B&H",
          delta_color="normal")
m2.metric("Buy & hold Sharpe", f"{wf['bh']['sharpe']:.2f}")
m3.metric("Strategy total return", f"{wf['strat']['total_return']:+.1%}")
m4.metric("Buy & hold total return", f"{wf['bh']['total_return']:+.1%}")

folds_df = pd.DataFrame(wf["folds"])
figf = go.Figure()
figf.add_bar(x=folds_df["fold"], y=folds_df["dir_acc"], name="model dir-acc")
figf.add_hline(y=wf["always_up"], line_dash="dot",
               annotation_text=f"always-up {wf['always_up']:.3f}")
figf.add_hline(y=0.5, line_dash="dot", line_color="grey",
               annotation_text="coin flip")
figf.update_layout(height=300, xaxis_title="walk-forward fold",
                   yaxis_title="directional accuracy",
                   yaxis_range=[0.3, 0.75])
st.plotly_chart(figf, use_container_width=True)
st.caption(
    "Every fold fits strictly on the past and scores on the following block "
    "(`assert_no_peeking` enforced). The dotted line is the always-up baseline "
    "— the number a model must clear before its directional signal means anything."
)

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Conformal intervals (Day 5) — coverage, not confidence theater
# ─────────────────────────────────────────────────────────────────────────────
st.header("3 · Conformal prediction intervals — nominal vs empirical coverage")
intervals = load_csv("phase3_intervals.csv")
if intervals is None:
    st.info("results/phase3_intervals.csv not found.")
else:
    st.dataframe(intervals, use_container_width=True, height=240)
    st.caption(
        "predictor.py used to print 85/70/55% 'confidence' from volatility "
        "buckets, tied to nothing. These split-conformal bands come from the "
        "model's own recent one-step errors — distribution-free, and the "
        "empirical coverage sits on the nominal level."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4 · MLflow runs (Day 9) + request telemetry
# ─────────────────────────────────────────────────────────────────────────────
st.header("4 · Tracked runs & service telemetry")
c1, c2 = st.columns(2)

with c1:
    st.subheader("MLflow walk-forward runs")
    mlruns = ROOT / "mlruns"
    if not mlruns.exists():
        st.info("No mlruns/ store yet — run `python -m src.tracking.mlflow_runs`.")
    else:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlruns.as_uri())
            runs = mlflow.search_runs(
                experiment_names=["walkforward"],
                order_by=["start_time DESC"], max_results=50)
            cols = [c for c in [
                "tags.mlflow.runName", "params.ticker", "params.model",
                "metrics.dir_acc_mean", "metrics.strat_sharpe_net",
                "metrics.bh_sharpe_net", "metrics.beats_bh_sharpe",
                "start_time"] if c in runs.columns]
            st.dataframe(runs[cols], use_container_width=True, height=300)
            if "metrics.beats_bh_sharpe" in runs.columns and len(runs):
                beat = int(runs["metrics.beats_bh_sharpe"].sum())
                st.caption(f"{beat} of {len(runs)} tracked runs beat their "
                           "buy-and-hold benchmark on Sharpe.")
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not read MLflow store: {e}")

with c2:
    st.subheader("API request telemetry")
    st.caption(f"Price-cache Redis: **{redis_status()}**")
    if not LOGS.exists():
        st.info("No logs/requests.jsonl yet — hit the FastAPI service first.")
    else:
        lines = LOGS.read_text(encoding="utf-8").strip().splitlines()[-200:]
        tel = pd.DataFrame([json.loads(ln) for ln in lines])
        st.dataframe(tel.tail(15).iloc[::-1], use_container_width=True, height=300)
        ok = tel[tel["status"] < 400]
        if len(ok):
            st.caption(
                f"last {len(tel)} requests — p50 {ok['latency_ms'].median():.0f} ms, "
                f"p95 {ok['latency_ms'].quantile(0.95):.0f} ms, "
                f"error rate {(tel['status'] >= 400).mean():.1%}")
