"""
Day 7 — Phase 5: multi-ticker portfolio backtest + PatchTST transformer.

Two questions, one honest harness
---------------------------------
1. **Portfolio depth.** Per-ticker, no strategy in this sprint has beaten
   buy-and-hold net of costs. Does portfolio machinery — sizing schemes,
   inverse-vol budgets, stop-losses — rescue the book, or does it just add
   turnover? Every configuration runs on the SAME frozen out-of-sample
   walk-forward predictions over the SAME aligned dates, against equal-weight
   buy-and-hold of the same ten names. Plus a transaction-cost sensitivity
   sweep (0-20 bps/side) to find where, if anywhere, the strategy breaks even.
2. **The transformer.** PatchTST — the architecture that made transformers
   competitive on TS benchmarks — vs the Day-3 field (LSTM, tuned XGB, ARIMA)
   on the identical expanding walk-forward. Capacity-matched small (17.7k
   params vs LSTM's ~5k), so the comparison is about architecture.

Protocol: identical tickers/span/folds as Days 2-6; scaler fit on train only;
signals decided at the prior close; costs charged on turnover; public
yfinance data via src.data.loader. Nothing is refit inside the backtest.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

import io
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.data.loader import load_prices                    # noqa: E402
from src.backtest.walkforward import (                     # noqa: E402
    expanding_window_folds, assert_no_peeking, rmse, directional_accuracy,
)
from src.backtest.trading import (                         # noqa: E402
    backtest_long_flat, backtest_buy_and_hold,
)
from src.backtest.portfolio import (                       # noqa: E402
    SCHEMES, align_panel, portfolio_backtest, buy_and_hold_portfolio,
    regime_slice,
)
from src.models import make_context                        # noqa: E402
from src.models import arima as m_arima                    # noqa: E402
from src.models import lstm as m_lstm                      # noqa: E402
from src.models import patchtst as m_patchtst              # noqa: E402
from src.models import xgb as m_xgb                        # noqa: E402

SEED = 42
TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "KO"]
START, END = "2021-01-01", "2025-01-01"        # identical span to Days 2-6
N_FOLDS = 5
COST_BPS = 5.0
COST_SWEEP = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
STOPS = [None, 0.05, 0.10]

RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
PLOTS = os.path.join(RESULTS, "plots")
for d in (RESULTS, SAMPLES, PLOTS):
    os.makedirs(d, exist_ok=True)

MODELS = {
    "patchtst": m_patchtst.predict_fold,
    "lstm_returns": m_lstm.predict_fold,
    "xgb_tuned_decay": m_xgb.predict_fold_tuned,
    "arima": m_arima.predict_fold,
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — out-of-sample predictions for every model x ticker x fold
# ─────────────────────────────────────────────────────────────────────────────
def generate_oos():
    """One long frame: (model, ticker, date, pred, actual) over all OOS days."""
    rows, per_fit = [], []
    for tk in TICKERS:
        prices, dates = load_prices(tk, START, END)
        folds = expanding_window_folds(len(prices), n_folds=N_FOLDS)
        assert_no_peeking(folds, n_samples=len(prices))
        ret = np.zeros_like(prices)
        ret[1:] = prices[1:] / prices[:-1] - 1.0
        ctx = make_context(prices, dates, with_features=True)

        for name, fn in MODELS.items():
            for f in folds:
                t0 = time.time()
                pred = np.asarray(fn(prices, f, ctx), dtype=float)
                secs = time.time() - t0
                actual = ret[f.test_start:f.test_end]
                assert len(pred) == len(actual) == f.test_len
                per_fit.append({"model": name, "ticker": tk, "fold": f.fold,
                                "fit_secs": secs})
                for i in range(f.test_len):
                    rows.append({
                        "model": name, "ticker": tk, "fold": f.fold,
                        "date": dates[f.test_start + i],
                        "pred": pred[i], "actual": actual[i],
                    })
            print(f"  [{tk}] {name}: {len(folds)} folds done")
    return pd.DataFrame(rows), pd.DataFrame(per_fit)


def trailing_panels(oos_model: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aligned pred/actual/vol21 panels for ONE model's OOS frame.

    vol21 is the trailing 21-day stdev of realised returns, shifted one day
    so the sizing weight for day t is knowable at the close of t-1.
    """
    per_tk = {}
    for tk, g in oos_model.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["vol21"] = g["actual"].rolling(21).std().shift(1)
        per_tk[tk] = g[["date", "pred", "actual", "vol21"]].dropna()
    return align_panel(per_tk)


def spy_regimes() -> pd.Series:
    """bull/bear/sideways from SPY's trailing 63-day return, shifted 1 day —
    the same definition as the Day-6 failure-mode analysis."""
    prices, dates = load_prices("SPY", START, END)
    s = pd.Series(prices, index=dates)
    r63 = s.pct_change(63).shift(1)
    return pd.Series(
        np.select([r63 > 0.05, r63 < -0.05], ["bull", "bear"], "sideways"),
        index=dates, name="regime",
    )[r63.notna()]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — transformer bake-off table (per-ticker walk-forward + trading)
# ─────────────────────────────────────────────────────────────────────────────
def bakeoff_table(oos: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in oos.groupby("model"):
        per_tk = []
        for tk, h in g.groupby("ticker"):
            h = h.sort_values("date")
            a, p = h["actual"].to_numpy(), h["pred"].to_numpy()
            tr = backtest_long_flat(p, a, cost_bps=COST_BPS)
            bh = backtest_buy_and_hold(a, cost_bps=COST_BPS)
            per_tk.append({
                "rmse": rmse(a, p),
                "rmse_vs_zero": rmse(a, p) / rmse(a, np.zeros_like(a)),
                "dir_acc": directional_accuracy(p, a),
                "abs_pred": float(np.abs(p).mean()),
                "sharpe": tr.sharpe, "total_return": tr.total_return,
                "sharpe_vs_bh": tr.sharpe - bh.sharpe,
            })
        t = pd.DataFrame(per_tk).mean()
        rows.append({
            "model": name,
            "mean_rmse_ret": t["rmse"], "mean_rmse_vs_zero": t["rmse_vs_zero"],
            "mean_dir_acc": t["dir_acc"], "mean_abs_pred": t["abs_pred"],
            "mean_sharpe": t["sharpe"], "mean_total_return": t["total_return"],
            "mean_sharpe_vs_bh": t["sharpe_vs_bh"],
            "mean_fit_secs": fits[fits["model"] == name]["fit_secs"].mean(),
        })
    return pd.DataFrame(rows).sort_values("mean_dir_acc", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — portfolio grid, cost sweep, regimes
# ─────────────────────────────────────────────────────────────────────────────
def portfolio_grid(panels_by_model: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    bench = buy_and_hold_portfolio(next(iter(panels_by_model.values())),
                                   cost_bps=COST_BPS)
    rows.append({"signal": "—", **bench.as_row()})
    curves["buy_and_hold"] = bench.net_daily
    for sig, panels in panels_by_model.items():
        for scheme in SCHEMES:
            for stop in STOPS:
                r = portfolio_backtest(panels, scheme=scheme, stop_loss=stop,
                                       cost_bps=COST_BPS)
                rows.append({"signal": sig, **r.as_row()})
                if stop is None:
                    curves[f"{sig}:{scheme}"] = r.net_daily
    df = pd.DataFrame(rows)
    df["sharpe_vs_bh"] = df["sharpe"] - bench.sharpe
    return df, curves


def cost_sweep(panels: dict, scheme: str) -> pd.DataFrame:
    rows = []
    for bps in COST_SWEEP:
        r = portfolio_backtest(panels, scheme=scheme, cost_bps=bps)
        b = buy_and_hold_portfolio(panels, cost_bps=bps)
        rows.append({"cost_bps": bps, "scheme": scheme,
                     "sharpe": r.sharpe, "total_return": r.total_return,
                     "bh_sharpe": b.sharpe, "bh_total": b.total_return,
                     "sharpe_vs_bh": r.sharpe - b.sharpe})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plots + outputs
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(curves, dates, grid, sweep, regs, bake):
    # 1) Equity curves vs buy-and-hold
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, net in curves.items():
        eq = np.cumprod(1.0 + net)
        lw, z = (2.6, 5) if label == "buy_and_hold" else (1.3, 3)
        ax.plot(dates, eq, label=label, linewidth=lw, zorder=z)
    ax.set_title("Day 7 — Portfolio equity (net of 5 bps/side) vs equal-weight buy-and-hold")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "day07_portfolio_equity.png"), dpi=130)

    # 2) Cost-sensitivity sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["cost_bps"], sweep["sharpe"], "o-", label="best active scheme")
    ax.plot(sweep["cost_bps"], sweep["bh_sharpe"], "s--", label="buy-and-hold")
    ax.set_title("Day 7 — Sharpe vs transaction cost (bps/side)")
    ax.set_xlabel("cost (bps per side)")
    ax.set_ylabel("annualised Sharpe")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "day07_cost_sweep.png"), dpi=130)

    # 3) Regime slice
    piv = regs.pivot(index="regime", columns="strategy", values="sharpe").reindex(
        ["bull", "sideways", "bear"])
    fig, ax = plt.subplots(figsize=(8, 5))
    piv.plot.bar(ax=ax, rot=0)
    ax.set_title("Day 7 — Sharpe by market regime (SPY trailing 63-day return)")
    ax.set_ylabel("annualised Sharpe")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "day07_regime_sharpe.png"), dpi=130)

    # 4) Transformer bake-off: dir-acc + RMSE ratio
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    b = bake.sort_values("mean_dir_acc")
    axes[0].barh(b["model"], b["mean_dir_acc"])
    axes[0].axvline(0.549, color="k", linestyle="--", label="always-up 0.549")
    axes[0].set_title("Directional accuracy (walk-forward mean)")
    axes[0].legend()
    axes[1].barh(b["model"], b["mean_rmse_vs_zero"])
    axes[1].axvline(1.0, color="k", linestyle="--", label="random walk = 1.0")
    axes[1].set_title("RMSE vs predicting zero (lower is better)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Day 7 — PatchTST vs the Day-3 field, identical folds")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "day07_transformer_bakeoff.png"), dpi=130)
    plt.close("all")


def main():
    np.random.seed(SEED)
    t_start = time.time()

    print("Stage 1 — generating out-of-sample predictions (4 models x 10 tickers x 5 folds)")
    oos, fits = generate_oos()
    oos.to_csv(os.path.join(SAMPLES, "day07_oos_predictions.csv"), index=False)

    print("Stage 2 — transformer bake-off")
    bake = bakeoff_table(oos, fits)
    bake.to_csv(os.path.join(RESULTS, "phase5_transformer.csv"), index=False)
    print(bake.to_string(index=False))

    print("Stage 3 — portfolio grid")
    panels_by_model = {
        sig: trailing_panels(oos[oos["model"] == sig])
        for sig in ("xgb_tuned_decay", "patchtst")
    }
    n_aligned = len(next(iter(panels_by_model.values()))["pred"])
    print(f"  aligned OOS days: {n_aligned}")
    grid, curves = portfolio_grid(panels_by_model)
    grid.to_csv(os.path.join(RESULTS, "phase5_portfolio.csv"), index=False)
    print(grid.drop(columns=["ann_vol", "cost_drag"]).to_string(index=False))

    print("Stage 4 — cost sweep on the best active scheme")
    active = grid[grid["signal"] != "—"]
    best = active.loc[active["sharpe"].idxmax()]
    best_panels = panels_by_model[best["signal"]]
    sweep = cost_sweep(best_panels, best["scheme"])
    sweep.to_csv(os.path.join(RESULTS, "phase5_cost_sweep.csv"), index=False)
    print(sweep.to_string(index=False))

    print("Stage 5 — regime slicing")
    regimes = spy_regimes()
    dates = panels_by_model["xgb_tuned_decay"]["pred"].index
    best_r = portfolio_backtest(best_panels, scheme=best["scheme"],
                                stop_loss=None, cost_bps=COST_BPS)
    bench = buy_and_hold_portfolio(best_panels, cost_bps=COST_BPS)
    regs = []
    for label, net in (("best_active", best_r.net_daily),
                       ("buy_and_hold", bench.net_daily)):
        r = regime_slice(net, dates, regimes)
        r["strategy"] = label
        regs.append(r)
    regs = pd.concat(regs, ignore_index=True)
    regs.to_csv(os.path.join(RESULTS, "phase5_regimes.csv"), index=False)
    print(regs.to_string(index=False))

    make_plots(curves, dates, grid, sweep, regs, bake)

    # Samples: 10 PatchTST prediction days per ticker
    samp = (oos[oos["model"] == "patchtst"].groupby("ticker")
            .head(10).reset_index(drop=True))
    samp.to_csv(os.path.join(SAMPLES, "day07_patchtst_sample_preds.csv"), index=False)

    # metrics.json — append-only protocol
    mpath = os.path.join(RESULTS, "metrics.json")
    with open(mpath, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    metrics["day07"] = {
        "phase": "5",
        "date_range": {"start": START, "end": END}, "tickers": TICKERS,
        "aligned_oos_days": int(n_aligned),
        "transformer_bakeoff": bake.to_dict(orient="records"),
        "portfolio_grid": grid.to_dict(orient="records"),
        "cost_sweep": sweep.to_dict(orient="records"),
        "regime_slice": regs.to_dict(orient="records"),
        "best_active": {"signal": str(best["signal"]),
                        "scheme": str(best["scheme"]),
                        "sharpe": float(best["sharpe"]),
                        "sharpe_vs_bh": float(best["sharpe_vs_bh"])},
        "runtime_secs": round(time.time() - t_start, 1),
    }
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    print(f"\nDone in {round((time.time() - t_start) / 60, 1)} min")


if __name__ == "__main__":
    main()
