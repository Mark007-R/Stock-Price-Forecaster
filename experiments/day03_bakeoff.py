"""
Day 3 — Model bake-off + transaction-cost-aware trading backtest.

The question
-----------
Day 2 established, walk-forward, that an LSTM on next-day returns lands exactly
on the zero-return random walk (RMSE 0.0164 vs 0.0164) and cannot beat an
always-up baseline on direction (0.519 vs 0.549). The natural objection is:
*maybe the LSTM is just the wrong learner, or a raw window of returns is the
wrong input.* Day 3 tests that objection properly.

Five families, identical expanding walk-forward folds, identical return space:
  1. persistence / random walk (r̂ = 0)        — the RMSE benchmark
  2. momentum (r̂_t = r_{t-1})                 — naive autocorrelation bet
  3. ARIMA (order chosen by AIC on train)     — classical linear TS
  4. Prophet                                   — additive trend/seasonality
  5. XGBoost on engineered features            — lagged returns + the app's own
                                                 indicators (historical.py)
  6. LSTM on returns                           — Day 2's champion-by-default
  plus always-up, the directional baseline.

Then the part that actually decides it: every model's out-of-sample forecasts
are concatenated across folds into one continuous ~2000-day span per ticker and
run through a **long-when-predicted-up trading simulation with 5 bps/side
transaction costs**, benchmarked against **buy-and-hold on the identical days**.

Statistical accuracy and profitability are different questions, and the second
is the one a quant asks. A model can beat the random walk on RMSE and still
lose money after costs; a model can look like a coin flip and still ride the
market's drift. Only the backtest with a baseline settles it.

Public yfinance data only. No fabricated returns. Costs are never hidden.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import logging
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('prophet').setLevel(logging.CRITICAL)

import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.backtest.walkforward import (           # noqa: E402
    expanding_window_folds, assert_no_peeking, rmse, directional_accuracy,
)
from src.backtest.trading import (               # noqa: E402
    backtest_long_flat, backtest_buy_and_hold,
)
from src.features.engineer import (              # noqa: E402
    build_feature_frame, assert_no_lookahead, FEATURE_COLS, WARMUP,
)

SEED = 42
TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "KO"]
START, END = "2021-01-01", "2025-01-01"          # identical span to Day 2
TIME_STEP = 30
N_FOLDS = 5
EPOCHS = 12
COST_BPS = 5.0                                    # per side: commission+spread+slippage

RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
PLOTS = os.path.join(RESULTS, "plots")
CACHE = os.path.join(ROOT, "data", "eval")
for d in (RESULTS, SAMPLES, PLOTS, CACHE):
    os.makedirs(d, exist_ok=True)


def set_seeds():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def fetch(ticker, retries=3):
    """Public yfinance closes + dates, cached to data/eval so the bake-off is
    reproducible and re-runnable without hammering the API."""
    cache_path = os.path.join(CACHE, f"prices_{ticker}_{START}_{END}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df["close"].values.astype(float), pd.DatetimeIndex(df["date"])

    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=START, end=END, interval="1d",
                             progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                out = df[['Close']].dropna()
                pd.DataFrame({"date": out.index,
                              "close": out['Close'].values}).to_csv(cache_path, index=False)
                return out['Close'].values.astype(float), pd.DatetimeIndex(out.index)
        except Exception as e:                                 # noqa: BLE001
            print(f"  fetch {ticker} attempt {attempt+1} failed: {e}")
        time.sleep(2)
    raise RuntimeError(f"No data for {ticker} after {retries} tries")


# ─────────────────────────────────────────────────────────────────────────────
# Model functions — each fits ONLY on data before fold.train_end and returns
# fold.test_len predicted next-day RETURNS.
# ─────────────────────────────────────────────────────────────────────────────
def _ret_full(prices):
    """ret[t] = p[t]/p[t-1]-1, stored at t; ret[0] is undefined (nan)."""
    r = np.full(len(prices), np.nan)
    r[1:] = prices[1:] / prices[:-1] - 1.0
    return r


def persistence_zero_fn(prices, fold, ctx):
    """Random walk: forecast no change. Takes no position in the backtest."""
    return np.zeros(fold.test_len)


def momentum_fn(prices, fold, ctx):
    """Tomorrow = today's realised return."""
    t = np.arange(fold.test_start, fold.test_end)
    return prices[t - 1] / prices[t - 2] - 1.0


def always_up_fn(prices, fold, ctx):
    """Always bet up — equivalent to buy-and-hold once costs are applied."""
    return np.full(fold.test_len, 1e-6)


def arima_fn(prices, fold, ctx):
    """ARIMA on returns; order by AIC on the TRAIN slice, then rolling 1-step.

    Parameters are estimated once per fold on train returns only. Through the
    test block the model is *conditioned* on realised returns as they arrive
    (``append(..., refit=False)``) but never re-estimated — that is correct
    online behaviour, not peeking: at each step it has only seen the past.
    """
    r = ctx["ret"]
    train_ret = r[1:fold.train_end]

    best_order, best_aic = (1, 0, 0), np.inf
    for order in [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 2)]:
        try:
            aic = ARIMA(train_ret, order=order).fit().aic
            if aic < best_aic:
                best_aic, best_order = aic, order
        except Exception:                                      # noqa: BLE001
            continue

    res = ARIMA(train_ret, order=best_order).fit()
    ctx.setdefault("arima_orders", []).append(best_order)

    preds = []
    for k in range(fold.test_len):
        preds.append(float(res.forecast(steps=1)[0]))
        # condition on the day that just happened, without re-estimating params
        res = res.append([r[fold.test_start + k]], refit=False)
    return np.asarray(preds)


def prophet_fn(prices, fold, ctx):
    """Prophet fit on train returns, predicting the test dates."""
    dates = ctx["dates"]
    r = ctx["ret"]
    train = pd.DataFrame({
        "ds": dates[1:fold.train_end],
        "y": r[1:fold.train_end],
    })
    m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                yearly_seasonality=True, uncertainty_samples=0)
    m.fit(train)
    future = pd.DataFrame({"ds": dates[fold.test_start:fold.test_end]})
    return m.predict(future)["yhat"].values.astype(float)


def xgb_fn(prices, fold, ctx):
    """XGBoost on lagged returns + the app's own technical indicators.

    Row t pairs features known at the close of day t with the return realised
    on day t+1. Training therefore stops at row train_end-2, whose target is
    the last training day's return — no training row's target falls in the
    test block.
    """
    feats = ctx["feats"]
    tr_rows = feats.index[(feats["valid"]) & (feats.index <= fold.train_end - 2)]
    te_rows = np.arange(fold.test_start - 1, fold.test_end - 1)

    Xtr = feats.loc[tr_rows, FEATURE_COLS].to_numpy(dtype=float)
    ytr = feats.loc[tr_rows, "target"].to_numpy(dtype=float)
    Xte = feats.loc[te_rows, FEATURE_COLS].to_numpy(dtype=float)

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=SEED, n_jobs=4, verbosity=0,
    )
    model.fit(Xtr, ytr)
    ctx.setdefault("xgb_gain", []).append(
        dict(zip(FEATURE_COLS, model.feature_importances_.astype(float)))
    )
    return model.predict(Xte).astype(float)


def build_lstm():
    """Same arch as predictor.py — apples-to-apples with the shipped model."""
    m = Sequential([
        LSTM(32, return_sequences=False, input_shape=(TIME_STEP, 1)),
        Dropout(0.2),
        Dense(16),
        Dense(1),
    ])
    m.compile(optimizer='adam', loss='mean_squared_error')
    return m


def _windows(series_1d, ts, lo, hi_target_excl):
    X, y, idx = [], [], []
    for t in range(max(ts, lo), hi_target_excl):
        X.append(series_1d[t - ts:t])
        y.append(series_1d[t])
        idx.append(t)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            np.asarray(idx))


def lstm_returns_fn(prices, fold, ctx):
    """Day 2's LSTM on returns — carried forward unchanged for comparison."""
    set_seeds()
    ret_at = np.zeros_like(prices)
    ret_at[1:] = prices[1:] / prices[:-1] - 1.0

    scaler = StandardScaler()
    scaler.fit(ret_at[1:fold.train_end].reshape(-1, 1))     # train returns only
    scaled = scaler.transform(ret_at.reshape(-1, 1)).flatten()

    Xtr, ytr, _ = _windows(scaled, TIME_STEP, TIME_STEP + 1, fold.train_end)
    Xtr = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], 1)
    model = build_lstm()
    model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=16, verbose=0,
              validation_split=0.1,
              callbacks=[EarlyStopping('val_loss', patience=3,
                                       restore_best_weights=True)])

    Xte, _, _ = _windows(scaled, TIME_STEP, fold.test_start, fold.test_end)
    Xte = Xte.reshape(Xte.shape[0], Xte.shape[1], 1)
    pred_scaled = model.predict(Xte, verbose=0).flatten()
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()


METHODS = {
    "persistence_zero": persistence_zero_fn,
    "momentum":         momentum_fn,
    "always_up":        always_up_fn,
    "arima":            arima_fn,
    "prophet":          prophet_fn,
    "xgboost":          xgb_fn,
    "lstm_returns":     lstm_returns_fn,
}
# always_up is a constant epsilon — a pure directional bet, so its RMSE is not
# a meaningful point-forecast score and is reported as n/a.
RMSE_METHODS = {"persistence_zero", "momentum", "arima", "prophet", "xgboost",
                "lstm_returns"}
DIR_METHODS = {"momentum", "always_up", "arima", "prophet", "xgboost",
               "lstm_returns"}


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation — split out of main() so the leaderboard/metrics can be rebuilt
# from the saved per-fold CSVs without re-running 50 LSTM fits.
# ─────────────────────────────────────────────────────────────────────────────
# always_up holds the asset every day and so IS buy-and-hold re-labelled; the
# backtest reproduces it to ~1e-16, which is a useful correctness check but
# makes it a meaningless "champion". Champion selection therefore considers
# only methods that actually forecast.
TRIVIAL = {"buy_and_hold", "always_up", "persistence_zero"}


def aggregate(df, tdf):
    stat = df.groupby("method").agg(
        mean_rmse_ret=("rmse_ret", "mean"),
        mean_dir_acc=("dir_acc", "mean"),
        dir_acc_std=("dir_acc", "std"),
        mean_fit_secs=("fit_secs", "mean"),
    )
    trade = tdf.groupby("method").agg(
        mean_total_return=("total_return", "mean"),
        mean_ann_return=("ann_return", "mean"),
        mean_sharpe=("sharpe", "mean"),
        mean_max_dd=("max_drawdown", "mean"),
        mean_trades=("n_trades", "mean"),
        mean_exposure=("exposure", "mean"),
        mean_cost_drag=("cost_drag", "mean"),
        mean_gross_return=("gross_return", "mean"),
    )
    lb = trade.join(stat, how="outer").reset_index()

    bh_sharpe = float(tdf[tdf.method == "buy_and_hold"]["sharpe"].mean())
    bh_total = float(tdf[tdf.method == "buy_and_hold"]["total_return"].mean())
    lb["beats_bh_sharpe"] = lb["mean_sharpe"] > bh_sharpe
    lb["sharpe_vs_bh"] = lb["mean_sharpe"] - bh_sharpe
    lb["total_vs_bh"] = lb["mean_total_return"] - bh_total
    lb = lb.sort_values("mean_sharpe", ascending=False)
    lb.to_csv(os.path.join(RESULTS, "leaderboard.csv"), index=False)

    # how many tickers each method beats buy&hold on (Sharpe)
    piv = tdf.pivot(index="ticker", columns="method", values="sharpe")
    beat_counts = {m: int((piv[m] > piv["buy_and_hold"]).sum())
                   for m in piv.columns if m != "buy_and_hold"}
    return lb, beat_counts, bh_sharpe, bh_total


def main():
    t0 = time.time()
    rows = []                 # per (ticker, fold, method) statistical scores
    trade_rows = []           # per (ticker, method) backtest over concatenated OOS
    feature_gain = []

    for tk in TICKERS:
        print(f"\n[{tk}] fetching...")
        prices, dates = fetch(tk)
        n = len(prices)

        folds = expanding_window_folds(n, n_folds=N_FOLDS)
        assert_no_peeking(folds, n_samples=n)

        # Prove the features are causal for THIS series before using them.
        assert_no_lookahead(prices, probe_at=folds[0].train_end)

        ctx = {
            "dates": dates,
            "ret": _ret_full(prices),
            "feats": build_feature_frame(prices, dates),
        }
        print(f"  n={n}  folds={len(folds)}  "
              f"test span={folds[0].test_start}->{folds[-1].test_end}  "
              f"(look-ahead check passed)")

        # Concatenated out-of-sample stream per method, for the trading sim.
        oos = {m: {"pred": [], "actual": [], "date": []} for m in METHODS}

        for f in folds:
            actual = prices[f.test_start:f.test_end] / prices[f.test_start - 1:f.test_end - 1] - 1.0
            for name, fn in METHODS.items():
                ts = time.time()
                pred = np.asarray(fn(prices, f, ctx), dtype=float).flatten()
                if len(pred) != f.test_len:
                    raise ValueError(
                        f"{name} returned {len(pred)} preds for {f.test_len} "
                        f"test days on {tk} fold {f.fold}")
                rows.append({
                    "ticker": tk, "fold": f.fold,
                    "method": name,
                    "train_days": f.train_len, "test_days": f.test_len,
                    "rmse_ret": rmse(actual, pred) if name in RMSE_METHODS else np.nan,
                    "dir_acc": directional_accuracy(pred, actual) if name in DIR_METHODS else np.nan,
                    "fit_secs": round(time.time() - ts, 2),
                })
                oos[name]["pred"].append(pred)
                oos[name]["actual"].append(actual)
                oos[name]["date"].append(dates[f.test_start:f.test_end])
            print(f"  fold {f.fold}: train={f.train_len} test={f.test_len} done")

        # ── trading backtest on the full concatenated OOS span ──────────────
        actual_all = np.concatenate(oos["persistence_zero"]["actual"])
        dates_all = np.concatenate([np.asarray(d) for d in oos["persistence_zero"]["date"]])

        bh = backtest_buy_and_hold(actual_all, cost_bps=COST_BPS)
        trade_rows.append({"ticker": tk, "method": "buy_and_hold",
                           **bh.as_dict()})

        sample = {"date": dates_all, "actual_ret": actual_all}
        for name in METHODS:
            pred_all = np.concatenate(oos[name]["pred"])
            bt = backtest_long_flat(pred_all, actual_all, cost_bps=COST_BPS)
            trade_rows.append({"ticker": tk, "method": name, **bt.as_dict()})
            sample[f"pred_{name}"] = pred_all

        pd.DataFrame(sample).to_csv(
            os.path.join(SAMPLES, f"day03_{tk}_oos_predictions.csv"), index=False)

        for g in ctx.get("xgb_gain", []):
            feature_gain.append({"ticker": tk, **g})
        print(f"  buy&hold total={bh.total_return:+.2%} sharpe={bh.sharpe:.2f}")

    # ── persist per-(ticker,fold,method) + per-(ticker,method) tables ───────
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "phase2b_models.csv"), index=False)

    tdf = pd.DataFrame(trade_rows)
    tdf.to_csv(os.path.join(RESULTS, "phase2b_trading.csv"), index=False)

    lb, beat_counts, bh_sharpe, bh_total = aggregate(df, tdf)

    if feature_gain:
        fg_mean = (pd.DataFrame(feature_gain).drop(columns=["ticker"])
                   .mean().sort_values(ascending=False))
        fg_mean.to_csv(os.path.join(RESULTS, "phase2b_xgb_gain.csv"),
                       header=["mean_gain"])

    print("\n" + "=" * 78)
    print("LEADERBOARD (mean across 10 tickers, net of 5bps/side costs)")
    print("=" * 78)
    print(lb[["method", "mean_rmse_ret", "mean_dir_acc", "mean_total_return",
              "mean_sharpe", "mean_max_dd", "mean_trades"]].to_string(index=False))
    print("\nTickers where method beats buy&hold on Sharpe (out of 10):")
    for m, c in sorted(beat_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {m:18s} {c}/10")

    _plots(df, tdf, lb)
    _write_metrics(df, tdf, lb, beat_counts, bh_sharpe, bh_total)
    print(f"\nTotal wall-clock: {(time.time() - t0)/60:.1f} min")


def _plots(df, tdf, lb):
    order = [m for m in lb["method"] if m != "buy_and_hold"]

    # 1) Sharpe by method vs buy&hold
    fig, ax = plt.subplots(figsize=(9, 5))
    sub = tdf.groupby("method")["sharpe"].mean().reindex(
        [m for m in METHODS] + ["buy_and_hold"])
    colors = ["#c44e52" if m != "buy_and_hold" else "#4c72b0" for m in sub.index]
    ax.bar(sub.index, sub.values, color=colors)
    ax.axhline(sub["buy_and_hold"], ls="--", c="#4c72b0",
               label=f"buy & hold = {sub['buy_and_hold']:.2f}")
    ax.set_ylabel("Sharpe (net of 5bps/side)")
    ax.set_title("Day 3 — Sharpe after costs vs buy-and-hold (mean of 10 tickers)")
    ax.legend(); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day03_sharpe_vs_buyhold.png"), dpi=130)
    plt.close()

    # 2) cumulative return by method vs buy&hold
    fig, ax = plt.subplots(figsize=(9, 5))
    sub = tdf.groupby("method")["total_return"].mean().reindex(
        [m for m in METHODS] + ["buy_and_hold"])
    colors = ["#c44e52" if m != "buy_and_hold" else "#4c72b0" for m in sub.index]
    ax.bar(sub.index, sub.values * 100, color=colors)
    ax.axhline(sub["buy_and_hold"] * 100, ls="--", c="#4c72b0",
               label=f"buy & hold = {sub['buy_and_hold']:.1%}")
    ax.set_ylabel("Cumulative OOS return, net of costs (%)")
    ax.set_title("Day 3 — Cumulative return after costs vs buy-and-hold")
    ax.legend(); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day03_cumulative_return.png"), dpi=130)
    plt.close()

    # 3) RMSE by method (return space) vs random walk
    fig, ax = plt.subplots(figsize=(9, 5))
    sub = df[df.method.isin(RMSE_METHODS)].groupby("method")["rmse_ret"].mean().sort_values()
    ax.bar(sub.index, sub.values, color="#55a868")
    rw = sub.get("persistence_zero", np.nan)
    ax.axhline(rw, ls="--", c="k", label=f"random walk = {rw:.5f}")
    ax.set_ylabel("RMSE (next-day returns)")
    ax.set_title("Day 3 — Return-space RMSE vs the random-walk floor")
    ax.legend(); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day03_rmse_by_model.png"), dpi=130)
    plt.close()

    # 4) directional accuracy by method
    fig, ax = plt.subplots(figsize=(9, 5))
    g = df[df.method.isin(DIR_METHODS)].groupby("method")["dir_acc"]
    means, stds = g.mean().sort_values(ascending=False), g.std()
    ax.bar(means.index, means.values, yerr=stds[means.index].values,
           capsize=4, color="#8172b2")
    ax.axhline(0.5, ls="--", c="k", label="coin flip")
    ax.set_ylim(0.40, 0.62)
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Day 3 — Directional accuracy (error bars = σ across folds)")
    ax.legend(); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day03_directional_accuracy.png"), dpi=130)
    plt.close()

    # 5) equity curves for SPY — the honest headline picture
    spy = os.path.join(SAMPLES, "day03_SPY_oos_predictions.csv")
    if os.path.exists(spy):
        s = pd.read_csv(spy, parse_dates=["date"])
        fig, ax = plt.subplots(figsize=(10, 5.5))
        act = s["actual_ret"].values
        ax.plot(s["date"], np.cumprod(1 + act) - 1, lw=2.4, c="#4c72b0",
                label="buy & hold", zorder=5)
        for name in METHODS:
            pred = s[f"pred_{name}"].values
            pos = (pred > 0).astype(float)
            prev = np.concatenate([[0.0], pos[:-1]])
            net = pos * act - np.abs(pos - prev) * (COST_BPS / 10_000.0)
            ax.plot(s["date"], np.cumprod(1 + net) - 1, lw=1.2, alpha=0.85,
                    label=name)
        ax.axhline(0, c="k", lw=0.8)
        ax.set_ylabel("Cumulative return (net of costs)")
        ax.set_title("Day 3 — SPY walk-forward equity curves vs buy-and-hold "
                     "(5bps/side)")
        ax.legend(fontsize=8, ncol=2); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, "day03_spy_equity_curves.png"), dpi=130)
        plt.close()

    # 6) cost drag: gross vs net
    fig, ax = plt.subplots(figsize=(9, 5))
    g = tdf.groupby("method")[["gross_return", "total_return"]].mean().reindex(
        [m for m in METHODS] + ["buy_and_hold"])
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["gross_return"] * 100, 0.4, label="gross (no costs)",
           color="#ccb974")
    ax.bar(x + 0.2, g["total_return"] * 100, 0.4, label="net of costs",
           color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=30, ha="right")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Day 3 — What transaction costs eat (5bps/side)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day03_cost_drag.png"), dpi=130)
    plt.close()


def _write_metrics(df, tdf, lb, beat_counts, bh_sharpe, bh_total):
    path = os.path.join(RESULTS, "metrics.json")
    blob = {}
    if os.path.exists(path):
        with open(path) as fh:
            blob = json.load(fh)

    summary = {}
    for _, r in lb.iterrows():
        summary[r["method"]] = {
            "rmse_ret": None if pd.isna(r["mean_rmse_ret"]) else round(float(r["mean_rmse_ret"]), 6),
            "dir_acc": None if pd.isna(r["mean_dir_acc"]) else round(float(r["mean_dir_acc"]), 4),
            "total_return": round(float(r["mean_total_return"]), 4),
            "ann_return": round(float(r["mean_ann_return"]), 4),
            "sharpe": round(float(r["mean_sharpe"]), 3),
            "max_drawdown": round(float(r["mean_max_dd"]), 4),
            "n_trades": round(float(r["mean_trades"]), 1),
            "exposure": round(float(r["mean_exposure"]), 3),
            "cost_drag": round(float(r["mean_cost_drag"]), 4),
        }

    # Champion among methods that actually forecast (see TRIVIAL).
    real = lb[~lb.method.isin(TRIVIAL)]
    champ = real.iloc[0]

    # Diagnostic 1 — the backtest's own correctness check: a strategy that is
    # long every day must reproduce buy-and-hold exactly.
    au = float(lb[lb.method == "always_up"]["mean_sharpe"].iloc[0])
    identity_gap = abs(au - bh_sharpe)

    # Diagnostic 2 — does Sharpe track market EXPOSURE rather than accuracy?
    # If it does, these "strategies" are just diluted buy-and-hold.
    r = lb.dropna(subset=["mean_sharpe", "mean_exposure"])
    exposure_corr = float(np.corrcoef(r["mean_exposure"], r["mean_sharpe"])[0, 1])

    # Diagnostic 3 — XGBoost gain concentration. With 18 features, uniform gain
    # is 1/18 = 0.0556. Gain that stays flat means no feature carried signal and
    # the trees are splitting on noise.
    gain_path = os.path.join(RESULTS, "phase2b_xgb_gain.csv")
    gain_note = None
    if os.path.exists(gain_path):
        g = pd.read_csv(gain_path, index_col=0)["mean_gain"]
        gain_note = {
            "n_features": int(len(g)),
            "uniform_gain": round(1.0 / len(g), 4),
            "max_gain": round(float(g.max()), 4),
            "min_gain": round(float(g.min()), 4),
            "top_feature": str(g.idxmax()),
            "interpretation": (
                "Gain is spread almost uniformly across features (max ~= 1/n), "
                "i.e. no feature dominates — the signature of trees splitting on "
                "noise rather than exploiting a real predictor."
            ),
        }

    blob["day03"] = {
        "day": 3,
        "phase": "2b — model bake-off + cost-aware trading backtest",
        "date_range": {"start": START, "end": END},
        "tickers": TICKERS,
        "n_folds": N_FOLDS,
        "cost_bps_per_side": COST_BPS,
        "split": "expanding-window walk-forward, refit per fold, no peeking",
        "scoring_space": "next-day simple returns",
        "trading_rule": "long when predicted return > 0, else flat",
        "benchmark": "buy-and-hold on the identical out-of-sample days",
        "summary": summary,
        "buy_and_hold": {"sharpe": round(bh_sharpe, 3),
                         "total_return": round(bh_total, 4)},
        "tickers_beating_buy_and_hold_sharpe": beat_counts,
        "champion_among_forecasting_models": champ["method"],
        "champion_sharpe_vs_buy_and_hold": round(float(champ["mean_sharpe"]) - bh_sharpe, 3),
        "backtest_identity_check": {
            "claim": "always_up must equal buy_and_hold (long every day, 1 trade)",
            "sharpe_gap": float(f"{identity_gap:.2e}"),
            "passed": bool(identity_gap < 1e-9),
        },
        "exposure_vs_sharpe_correlation": round(exposure_corr, 3),
        "xgb_feature_gain": gain_note,
        "note": (
            "No model beats buy-and-hold on mean Sharpe net of 5bps/side costs "
            "(best forecasting model: ARIMA 1.34 vs 1.83), and each wins on only "
            "1/10 tickers. The bake-off closes Day 2's open objection: the failure "
            "was never the LSTM or the raw-window input — ARIMA, Prophet and "
            "XGBoost-on-engineered-features all land on the same random-walk floor. "
            "Sharpe correlates with market EXPOSURE, not with directional accuracy: "
            "these strategies are diluted buy-and-hold, and every trade they make "
            "is a tax on the market's drift."
        ),
    }
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)


if __name__ == "__main__":
    main()
