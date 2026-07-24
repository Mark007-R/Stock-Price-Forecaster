"""
StockAI — the 60-second honest-evaluation demo.

Runs the whole sprint story end-to-end, live, offline (cached AAPL prices),
in four acts:

  1. THE BUG      — the scaler leakage predictor.py shipped with, demonstrated
                    on real prices in three lines.
  2. WALK-FORWARD — ARIMA (the champion) vs the random walk across expanding
                    walk-forward folds. No single lucky holdout.
  3. THE BACKTEST — long/flat trading with 5 bps costs vs buy-and-hold.
                    The champion loses. That is the finding.
  4. THE INTERVAL — a 5-day forecast whose uncertainty comes from the model's
                    own errors (split-conformal), not from vibes.

Usage:  python demo.py            (venv with requirements.txt)
Talk-track: docs/DEMO.md
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.backtest.trading import backtest_buy_and_hold, backtest_long_flat
from src.backtest.walkforward import (
    directional_accuracy, expanding_window_folds, walk_forward_predict,
)
from src.data.loader import load_prices
from src.models import make_context
from src.models.arima import forecast_returns, predict_fold as arima_fold
from src.models.intervals import conformal_halfwidth
from src.models.persistence import predict_zero

TICKER, START, END = "AAPL", "2021-01-01", "2025-01-01"
COST_BPS = 5.0
BAR = "─" * 66


def act1_the_bug(prices: np.ndarray) -> None:
    print(f"\n{BAR}\nACT 1 — the bug this sprint started with (predictor.py:78)\n{BAR}")
    closes = prices.reshape(-1, 1)
    split = int(len(closes) * 0.8)

    leaky = MinMaxScaler().fit(closes)            # fit on EVERYTHING (the bug)
    honest = MinMaxScaler().fit(closes[:split])   # fit on train only (the fix)

    print(f"  {TICKER} {START}..{END}: {len(closes)} days, train/test split at day {split}")
    print(f"  leaky  scaler max: ${leaky.data_max_[0]:9.2f}  <- includes the TEST window's high")
    print(f"  honest scaler max: ${honest.data_max_[0]:9.2f}  <- knows only the train window")
    print("  The leaky scaler told the model where prices would top out a year")
    print("  early. Day 1 measured the damage: reported RMSE understated ~5-23%")
    print("  on 7/10 tickers (results/phase1_leakage_comparison.csv).")


def act2_walk_forward(prices: np.ndarray, dates) -> dict:
    print(f"\n{BAR}\nACT 2 — walk-forward: champion (ARIMA) vs the random walk\n{BAR}")
    folds = expanding_window_folds(len(prices), n_folds=5)
    ctx = make_context(prices, dates)

    t0 = time.perf_counter()
    arima_res = walk_forward_predict(prices, lambda p, f: arima_fold(p, f, ctx), folds)
    zero_res = walk_forward_predict(prices, predict_zero, folds)
    secs = time.perf_counter() - t0

    out = {}
    for name, res in (("arima", arima_res), ("random walk", zero_res)):
        pred = np.concatenate([r["pred_ret"] for r in res])
        act = np.concatenate([r["actual_ret"] for r in res])
        out[name] = (pred, act)
        rmse = float(np.sqrt(np.mean([r["rmse_ret"] ** 2 for r in res])))
        if np.all(pred == 0.0):
            dacc_s = "  —   (forecasts no edge, takes no position)"
        else:
            dacc = directional_accuracy(pred, act)
            dacc_s = f"{dacc:.3f}"
        print(f"  {name:<12} RMSE(ret) {rmse:.6f}   dir-acc {dacc_s}")

    _, act = out["arima"]
    print(f"  always-up baseline dir-acc: {np.mean(act > 0):.3f}"
          f"   ({len(act)} out-of-sample days, {len(folds)} folds, {secs:.1f}s)")
    return out


def act3_the_backtest(out: dict) -> None:
    print(f"\n{BAR}\nACT 3 — the only question that matters: P&L after {COST_BPS:.0f} bps costs\n{BAR}")
    pred, act = out["arima"]
    strat = backtest_long_flat(pred, act, cost_bps=COST_BPS)
    bh = backtest_buy_and_hold(act, cost_bps=COST_BPS)

    print(f"  {'':<16}{'total ret':>10}{'Sharpe':>9}{'max DD':>9}{'trades':>8}")
    print(f"  {'ARIMA long/flat':<16}{strat.total_return:>+9.1%}{strat.sharpe:>9.2f}"
          f"{strat.max_drawdown:>9.1%}{strat.n_trades:>8d}")
    print(f"  {'buy-and-hold':<16}{bh.total_return:>+9.1%}{bh.sharpe:>9.2f}"
          f"{bh.max_drawdown:>9.1%}{bh.n_trades:>8d}")
    verdict = "BEATS" if strat.sharpe > bh.sharpe else "LOSES TO"
    print(f"\n  Verdict: the best forecaster in the repo {verdict} buy-and-hold.")
    print("  Across all 10 tickers (Day 3): ARIMA Sharpe 1.34 vs B&H 1.83.")
    print("  The honest result IS the result. No cherry-picked window changes it.")


def act4_the_interval(prices: np.ndarray) -> None:
    print(f"\n{BAR}\nACT 4 — a forecast that admits what it doesn't know\n{BAR}")
    horizon, calib_days = 5, 120
    ret = prices[1:] / prices[:-1] - 1.0

    # Conformal band from the last `calib_days` realised returns (the champion
    # forecasts ~0, so its 1-step errors ≈ the returns themselves).
    q80 = conformal_halfwidth(ret[-calib_days:], alpha=0.20)
    pred_ret = forecast_returns(prices, horizon)
    last = prices[-1]
    path = last * np.cumprod(1.0 + pred_ret)

    print(f"  last close ${last:.2f} -> {horizon}-day point path with 80% conformal bands:")
    for h in range(horizon):
        w = q80 * np.sqrt(h + 1)
        print(f"    day +{h+1}:  ${path[h]:8.2f}   [{path[h]*(1-w):8.2f} .. {path[h]*(1+w):8.2f}]")
    print("  Bands come from the model's own calibrated errors and carry a")
    print("  checkable claim: ~80% of outcomes should land inside (Day 5")
    print("  measured 80.8% — the old volatility heuristic claimed nothing).")


def main() -> None:
    print(f"{BAR}\nStockAI — honest time-series evaluation, end to end\n{BAR}")
    t0 = time.perf_counter()
    prices, dates = load_prices(TICKER, START, END)   # cached CSV, offline

    act1_the_bug(prices)
    out = act2_walk_forward(prices, dates)
    act3_the_backtest(out)
    act4_the_interval(prices)

    print(f"\n{BAR}\nTotal demo runtime: {time.perf_counter() - t0:.1f}s — "
          f"90 pytest guards behind it (tests/)\n{BAR}")


if __name__ == "__main__":
    main()
