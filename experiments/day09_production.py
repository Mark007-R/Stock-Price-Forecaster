"""
Day 9 — Phase 7: production wrapper measurements.

Two measured questions, because ops claims need numbers like model claims do:

A. **What does each price-cache tier actually cost?** Cold yfinance fetch vs
   local disk CSV vs Redis GET, same series. The expected (and honest) result:
   Redis is NOT there to be faster than a local file — it's there so separate
   containers share one cache instead of each hitting yfinance.

B. **MLflow-tracked walk-forward runs** for the registry's fast models across
   the sprint's 10 tickers — the tracked-run baseline the ops dashboard and
   any future retrain job append to. Every run logs strategy AND buy-and-hold
   metrics; the store is born unable to show one without the other.

Usage:
    python experiments/day09_production.py --stage cache     # A
    python experiments/day09_production.py --stage mlflow    # B
    python experiments/day09_production.py                   # both
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
SAMPLES = RESULTS / "samples"

TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "KO"]
START, END = "2021-01-01", "2025-01-01"
# Fast registry members only: LSTM/PatchTST need TF + minutes per fold and
# their walk-forward story is already recorded in results/ from Days 2 and 7.
MODELS = ["persistence", "momentum", "arima", "xgboost", "xgboost_tuned"]
N_REPS = 20   # latency reps for warm tiers


def _reload_loader():
    """Fresh loader module so the Redis client/flags reset between scenarios."""
    import src.data.loader as loader
    return importlib.reload(loader)


def bench_cache() -> pd.DataFrame:
    rows = []

    # ── Tier 1: cold network fetch (no disk cache, no redis) ────────────────
    os.environ.pop("REDIS_URL", None)
    loader = _reload_loader()
    cold_ticker = "AAPL"
    cold_range = ("2019-01-01", "2020-01-01")   # a range no experiment cached
    for f in (loader.DEFAULT_CACHE and Path(loader.DEFAULT_CACHE).glob(
            f"*_{cold_ticker}_{cold_range[0]}_{cold_range[1]}.csv")):
        f.unlink()
    t0 = time.perf_counter()
    df_cold = loader.load_ohlcv(cold_ticker, *cold_range)
    cold_ms = (time.perf_counter() - t0) * 1000
    rows.append({"tier": "yfinance (cold fetch)", "reps": 1,
                 "p50_ms": round(cold_ms, 1), "p95_ms": round(cold_ms, 1),
                 "n_rows": len(df_cold),
                 "shared_across_containers": False})

    # ── Tier 2: local disk CSV cache ─────────────────────────────────────────
    lat = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        df = loader.load_ohlcv("AAPL", START, END)
        lat.append((time.perf_counter() - t0) * 1000)
    rows.append({"tier": "disk CSV cache", "reps": N_REPS,
                 "p50_ms": round(statistics.median(lat), 1),
                 "p95_ms": round(sorted(lat)[int(0.95 * N_REPS) - 1], 1),
                 "n_rows": len(df), "shared_across_containers": False})

    # ── Tier 3: Redis GET (real server if reachable, else fakeredis) ────────
    redis_kind = "redis"
    try:
        import redis as _redis
        _redis.Redis(host="localhost", port=6379,
                     socket_connect_timeout=1).ping()
        os.environ["REDIS_URL"] = "redis://localhost:6379/0"
        loader = _reload_loader()
    except Exception:  # noqa: BLE001 — no live server: measure via fakeredis
        redis_kind = "fakeredis (in-process, lower bound)"
        os.environ["REDIS_URL"] = "redis://fake"
        loader = _reload_loader()
        import fakeredis
        loader._redis_client = fakeredis.FakeRedis()

    loader.load_ohlcv("AAPL", START, END)          # populate the key
    lat = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        df = loader._redis_get(f"stockai:ohlcv:AAPL:{START}:{END}")
        lat.append((time.perf_counter() - t0) * 1000)
    assert df is not None and len(df) > 0, "Redis tier returned nothing"
    rows.append({"tier": f"Redis GET ({redis_kind})", "reps": N_REPS,
                 "p50_ms": round(statistics.median(lat), 1),
                 "p95_ms": round(sorted(lat)[int(0.95 * N_REPS) - 1], 1),
                 "n_rows": len(df), "shared_across_containers": True})

    os.environ.pop("REDIS_URL", None)
    _reload_loader()

    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "phase7_cache_bench.csv", index=False)
    print(out.to_string(index=False))
    return out


def run_mlflow_batch() -> pd.DataFrame:
    from src.tracking.mlflow_runs import track_walkforward_run

    rows = []
    for model in MODELS:
        for tk in TICKERS:
            t0 = time.perf_counter()
            s = track_walkforward_run(tk, model)
            s["wall_seconds"] = round(time.perf_counter() - t0, 2)
            rows.append(s)
            print(f"[{model:>14}] {tk:>5} dir_acc={s['dir_acc_mean']} "
                  f"sharpe={s['strat_sharpe_net']} vs B&H {s['bh_sharpe_net']} "
                  f"({s['wall_seconds']}s) run={s['run_id'][:8]}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase7_mlflow_runs.csv", index=False)

    # Summary: per model, how many tickers beat B&H + mean Sharpe gap.
    summary = (df.groupby("model")
                 .agg(runs=("run_id", "count"),
                      mean_dir_acc=("dir_acc_mean", "mean"),
                      mean_sharpe=("strat_sharpe_net", "mean"),
                      mean_bh_sharpe=("bh_sharpe_net", "mean"),
                      beats_bh=("beats_buy_and_hold_sharpe", "sum"),
                      mean_wall_s=("wall_seconds", "mean"))
                 .round(4).reset_index())
    summary.to_csv(RESULTS / "phase7_mlflow_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    # Plot: champion Sharpe vs B&H per ticker, from the tracked runs.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    champ = df[df["model"] == "arima"].set_index("ticker").loc[TICKERS]
    x = range(len(TICKERS))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - 0.2 for i in x], champ["strat_sharpe_net"], width=0.4,
           label="ARIMA long/flat (net of 5 bps/side)")
    ax.bar([i + 0.2 for i in x], champ["bh_sharpe_net"], width=0.4,
           label="buy & hold (net)")
    ax.set_xticks(list(x), TICKERS)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Sharpe (walk-forward OOS)")
    ax.set_title("Tracked runs, champion vs benchmark — every MLflow run "
                 "logs both sides")
    ax.legend()
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / "day09_mlflow_sharpe.png", dpi=110)
    plt.close(fig)

    # Samples: a handful of tracked-run summaries as the day's sample output.
    SAMPLES.mkdir(exist_ok=True)
    with open(SAMPLES / "day09_mlflow_run_samples.json", "w", encoding="utf-8") as f:
        json.dump(rows[:2] + rows[-2:], f, indent=2)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "mlflow", "all"], default="all")
    args = ap.parse_args()
    if args.stage in ("cache", "all"):
        print("── A. price-cache tier latency ─────────────────────────────")
        bench_cache()
    if args.stage in ("mlflow", "all"):
        print("\n── B. MLflow-tracked walk-forward runs ─────────────────────")
        run_mlflow_batch()


if __name__ == "__main__":
    main()
