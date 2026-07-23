"""
MLflow tracking for walk-forward runs — every backtest becomes a queryable record.

Why this exists
---------------
Days 2–8 wrote their walk-forward numbers into ad-hoc CSVs under ``results/``.
That was right for a sprint (append-only, diffable in a PR), but it does not
answer operational questions: "what were the exact params of the run that
produced this Sharpe?", "how did dir-acc move across the last N runs of the
champion?", "which artifact belongs to which run?". MLflow's file store gives
all of that for free — params + metrics + artifacts keyed by run, browsable
with ``mlflow ui`` — without standing up any server.

What gets logged per run
------------------------
* **params**: ticker, model, folds, cost_bps, date range, code SHA if available
* **metrics**: per-fold ``rmse_ret`` / ``dir_acc`` (as steps), aggregate means,
  strategy Sharpe/return/drawdown NET of costs, and the buy-and-hold benchmark
  on the identical days — the benchmark is logged as a first-class metric so
  no MLflow view can show a strategy number without its baseline sitting
  next to it.
* **artifacts**: per-fold CSV + the equity-curve PNG (strategy vs B&H).

The tracking URI defaults to ``<repo>/mlruns`` (file store, gitignored). In
Docker the compose file mounts the same directory into every container, so
API-triggered runs and CLI runs land in one place.

Run from the CLI:
    python -m src.tracking.mlflow_runs --tickers AAPL,MSFT --models arima
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_prices                                  # noqa: E402
from src.models import CHAMPION, get_model, make_context                 # noqa: E402
from src.backtest.walkforward import (                                   # noqa: E402
    expanding_window_folds, assert_no_peeking, walk_forward_predict,
)
from src.backtest.trading import backtest_long_flat, backtest_buy_and_hold  # noqa: E402

DEFAULT_TRACKING_URI = (ROOT / "mlruns").as_uri()
DEFAULT_EXPERIMENT = "walkforward"
DEFAULT_START, DEFAULT_END = "2021-01-01", "2025-01-01"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — tracking must work without git too
        return None


def _equity_png(pred: np.ndarray, actual: np.ndarray, cost_bps: float,
                title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost = cost_bps / 10_000.0
    position = (pred > 0).astype(float)
    prev = np.concatenate([[0.0], position[:-1]])
    net = position * actual - np.abs(position - prev) * cost
    bh = actual.copy()
    bh[0] -= cost

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.cumprod(1 + net), label="strategy (net of costs)", lw=1.6)
    ax.plot(np.cumprod(1 + bh), label="buy & hold (net)", lw=1.6, ls="--")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("out-of-sample trading day")
    ax.set_ylabel("equity (start = 1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def track_walkforward_run(
    ticker: str,
    model_name: str = CHAMPION,
    n_folds: int = 5,
    cost_bps: float = 5.0,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> dict:
    """Run one walk-forward backtest and record it as an MLflow run.

    Returns a summary dict (run_id + headline metrics). Raises on any model or
    data failure — a failed run must never be logged as a finished one, so the
    MLflow run is only created after the numeric work has succeeded.
    """
    model_fn = get_model(model_name)
    prices, dates = load_prices(ticker, start, end)

    t0 = time.perf_counter()
    folds = expanding_window_folds(len(prices), n_folds=n_folds)
    assert_no_peeking(folds, n_samples=len(prices))
    ctx = make_context(prices, dates,
                       with_features=model_name.startswith("xgboost"))
    fold_results = walk_forward_predict(
        prices, lambda p, f: model_fn(p, f, ctx), folds)
    elapsed = time.perf_counter() - t0

    pred_all = np.concatenate([r["pred_ret"] for r in fold_results])
    actual_all = np.concatenate([r["actual_ret"] for r in fold_results])
    bt = backtest_long_flat(pred_all, actual_all, cost_bps=cost_bps)
    bh = backtest_buy_and_hold(actual_all, cost_bps=cost_bps)
    dir_accs = [r["dir_acc"] for r in fold_results if not np.isnan(r["dir_acc"])]

    import mlflow

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"{model_name}_{ticker.upper()}") as run:
        mlflow.log_params({
            "ticker": ticker.upper(), "model": model_name,
            "n_folds": len(fold_results), "cost_bps": cost_bps,
            "start": start, "end": end, "n_days": len(prices),
            "split": "expanding_walkforward",
            "git_sha": _git_sha() or "unknown",
        })
        for r in fold_results:
            mlflow.log_metric("fold_rmse_ret", r["rmse_ret"], step=r["fold"])
            if not np.isnan(r["dir_acc"]):
                mlflow.log_metric("fold_dir_acc", r["dir_acc"], step=r["fold"])
        always_up = float(np.mean(actual_all > 0))
        mlflow.log_metrics({
            "rmse_ret_mean": float(np.mean([r["rmse_ret"] for r in fold_results])),
            "dir_acc_mean": float(np.mean(dir_accs)) if dir_accs else float("nan"),
            "dir_acc_always_up": always_up,
            "oos_days": float(len(actual_all)),
            "fit_seconds": elapsed,
            "strat_sharpe_net": bt.sharpe,
            "strat_total_return_net": bt.total_return,
            "strat_max_drawdown": bt.max_drawdown,
            "strat_exposure": bt.exposure,
            "strat_n_trades": float(bt.n_trades),
            "bh_sharpe_net": bh.sharpe,
            "bh_total_return_net": bh.total_return,
            "bh_max_drawdown": bh.max_drawdown,
            "beats_bh_sharpe": float(bt.sharpe > bh.sharpe),
        })

        with tempfile.TemporaryDirectory() as tmp:
            folds_csv = os.path.join(tmp, "folds.csv")
            pd.DataFrame([{k: r[k] for k in
                           ("fold", "train_days", "test_days", "rmse_ret", "dir_acc")}
                          for r in fold_results]).to_csv(folds_csv, index=False)
            png = os.path.join(tmp, "equity_curve.png")
            _equity_png(pred_all, actual_all, cost_bps,
                        f"{ticker.upper()} — {model_name} vs buy & hold "
                        f"({len(actual_all)} OOS days, {cost_bps} bps/side)", png)
            mlflow.log_artifact(folds_csv)
            mlflow.log_artifact(png)

        run_id = run.info.run_id

    return {
        "run_id": run_id, "ticker": ticker.upper(), "model": model_name,
        "n_folds": len(fold_results), "oos_days": int(len(actual_all)),
        "dir_acc_mean": round(float(np.mean(dir_accs)), 4) if dir_accs else None,
        "dir_acc_always_up": round(always_up, 4),
        "strat_sharpe_net": round(bt.sharpe, 4),
        "bh_sharpe_net": round(bh.sharpe, 4),
        "beats_buy_and_hold_sharpe": bool(bt.sharpe > bh.sharpe),
        "fit_seconds": round(elapsed, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MLflow-tracked walk-forward runs")
    ap.add_argument("--tickers", default="AAPL", help="comma-separated")
    ap.add_argument("--models", default=CHAMPION, help="comma-separated")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    args = ap.parse_args()

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for tk in [t.strip() for t in args.tickers.split(",") if t.strip()]:
            summary = track_walkforward_run(
                tk, model, n_folds=args.folds, cost_bps=args.cost_bps,
                start=args.start, end=args.end)
            rows.append(summary)
            print(f"[mlflow] {summary['model']:>14} {summary['ticker']:>5} "
                  f"dir_acc={summary['dir_acc_mean']} "
                  f"sharpe={summary['strat_sharpe_net']} "
                  f"(B&H {summary['bh_sharpe_net']}) run={summary['run_id'][:8]}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
