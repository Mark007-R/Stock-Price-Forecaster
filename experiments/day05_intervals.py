"""
Day 5 — Does the new uncertainty machinery actually keep its promises?

Phase 3 swaps predictor.py's volatility-only "confidence" (vol<2 → 85, <4 →
70, else 55 — numbers with no coverage semantics) for split-conformal
prediction intervals built from the model's own held-out residuals. That swap
is only an upgrade if the new intervals are CALIBRATED: an 80% band should
contain the realised value ~80% of the time, a 95% band ~95%.

This experiment measures exactly that, on the Day-3 out-of-sample champion
(ARIMA) forecast streams — 10 tickers x ~500 OOS days, walk-forward, no
peeking — using a rolling 120-day calibration window and sequential
evaluation (every interval is built strictly from errors observed BEFORE the
day it covers).

Three uncertainty methods face the same residual streams:
  1. old volatility heuristic — operationalised as the Gaussian band at its
     own claimed level (vol<2% ⇒ an "85% confident" ±z85·vol band, etc.),
     because the shipped code never defined an interval at all;
  2. Gaussian ±z·σ on model residuals — the textbook parametric choice;
  3. split-conformal — distribution-free finite-sample quantile.

Hypothesis: daily-return residuals are fat-tailed, so Gaussian bands OVER-
cover at central levels (its 80% band is too wide) and UNDER-cover in the
tails (its 99% band misses too often) — the S-shaped calibration curve —
while conformal tracks the diagonal at every level. The 99% band is where
risk lives, and it is exactly where the Gaussian assumption fails.

Also validated end-to-end here (execute-everything rule):
  * predictor.py's rewired confidence path, by running run_prediction() for a
    real ticker and checking the interval fields + band monotonicity;
  * the FastAPI service, by booting uvicorn and exercising all endpoints
    (/health, /predict, /backtest, /indicators, /correlation) over HTTP.
"""
import os
import sys
import json
import time
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.models.intervals import conformal_halfwidth  # noqa: E402

TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "KO"]
CHAMPION_COL = "pred_arima"          # Day-3 champion among forecasting models
CALIB_WINDOW = 120                    # same window the serving layer uses
LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
PLOTS = os.path.join(RESULTS, "plots")
for d in (RESULTS, SAMPLES, PLOTS):
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Part A — rolling-calibration coverage study on Day-3 OOS streams
# ─────────────────────────────────────────────────────────────────────────────
def old_heuristic_claim(vol_pct: float) -> float:
    """The shipped heuristic's claimed confidence for a given asset vol."""
    if vol_pct < 2:
        return 0.85
    if vol_pct < 4:
        return 0.70
    return 0.55


def coverage_study():
    rows = []
    heur_rows = []
    for tk in TICKERS:
        path = os.path.join(SAMPLES, f"day03_{tk}_oos_predictions.csv")
        s = pd.read_csv(path, parse_dates=["date"])
        pred = s[CHAMPION_COL].to_numpy(dtype=float)
        actual = s["actual_ret"].to_numpy(dtype=float)
        resid = pred - actual
        n = len(resid)

        # Sequential evaluation: the interval for day t uses residuals
        # [t-W, t) only — errors fully observed before t. No peeking.
        eval_idx = np.arange(CALIB_WINDOW, n)

        for level in LEVELS:
            alpha = 1.0 - level
            z = norm.ppf(1.0 - alpha / 2.0)
            cov_g, cov_c, w_g, w_c = [], [], [], []
            for t in eval_idx:
                window = resid[t - CALIB_WINDOW:t]
                hw_g = z * np.std(window, ddof=1)
                hw_c = conformal_halfwidth(window, alpha=alpha)
                cov_g.append(abs(pred[t] - actual[t]) <= hw_g)
                cov_c.append(abs(pred[t] - actual[t]) <= hw_c)
                w_g.append(hw_g)
                w_c.append(hw_c)
            rows.append({"ticker": tk, "level": level, "method": "gaussian",
                         "coverage": float(np.mean(cov_g)),
                         "mean_halfwidth": float(np.mean(w_g)),
                         "n_eval": len(eval_idx)})
            rows.append({"ticker": tk, "level": level, "method": "conformal",
                         "coverage": float(np.mean(cov_c)),
                         "mean_halfwidth": float(np.mean(w_c)),
                         "n_eval": len(eval_idx)})

        # Old heuristic: one claimed level per ticker (from rolling asset vol),
        # evaluated as the Gaussian band at exactly that claimed level.
        cov_h, w_h, claims = [], [], []
        for t in eval_idx:
            vol_win = actual[t - CALIB_WINDOW:t]
            vol_pct = float(np.std(vol_win, ddof=1) * 100)
            claim = old_heuristic_claim(vol_pct)
            z = norm.ppf(1.0 - (1.0 - claim) / 2.0)
            hw = z * np.std(vol_win, ddof=1)     # asset vol, NOT model residuals
            cov_h.append(abs(pred[t] - actual[t]) <= hw)
            w_h.append(hw)
            claims.append(claim)
        heur_rows.append({
            "ticker": tk,
            "claimed_confidence_mean": float(np.mean(claims)),
            "empirical_coverage": float(np.mean(cov_h)),
            "gap": float(np.mean(cov_h) - np.mean(claims)),
            "mean_halfwidth": float(np.mean(w_h)),
            "n_eval": len(eval_idx),
        })
        print(f"  {tk}: heuristic claims {np.mean(claims):.0%} "
              f"→ covers {np.mean(cov_h):.1%}")

    df = pd.DataFrame(rows)
    hdf = pd.DataFrame(heur_rows)
    df.to_csv(os.path.join(RESULTS, "phase3_intervals.csv"), index=False)
    hdf.to_csv(os.path.join(RESULTS, "phase3_old_heuristic.csv"), index=False)
    return df, hdf


def plots(df, hdf):
    agg = df.groupby(["method", "level"]).agg(
        coverage=("coverage", "mean"), width=("mean_halfwidth", "mean")).reset_index()

    # 1) Calibration curve — the headline picture
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0.45, 1.0], [0.45, 1.0], "k--", lw=1, label="perfect calibration")
    for m, c in [("gaussian", "#c44e52"), ("conformal", "#4c72b0")]:
        sub = agg[agg.method == m].sort_values("level")
        ax.plot(sub.level, sub.coverage, "o-", color=c, label=m)
    hx, hy = hdf["claimed_confidence_mean"].mean(), hdf["empirical_coverage"].mean()
    ax.scatter([hx], [hy], marker="X", s=140, c="#55a868", zorder=5,
               label=f"old vol heuristic (claims {hx:.0%}, covers {hy:.0%})")
    ax.set_xlabel("Nominal (claimed) coverage")
    ax.set_ylabel("Empirical coverage (10 tickers, walk-forward OOS)")
    ax.set_title("Day 5 — Interval calibration: conformal vs Gaussian vs shipped heuristic")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day05_calibration_curve.png"), dpi=130)
    plt.close()

    # 2) Tail zoom — miss rates at the 99% level, per ticker
    fig, ax = plt.subplots(figsize=(9, 5))
    t99 = df[df.level == 0.99].pivot(index="ticker", columns="method", values="coverage")
    x = np.arange(len(t99))
    ax.bar(x - 0.2, (1 - t99["gaussian"]) * 100, 0.4, label="gaussian", color="#c44e52")
    ax.bar(x + 0.2, (1 - t99["conformal"]) * 100, 0.4, label="conformal", color="#4c72b0")
    ax.axhline(1.0, ls="--", c="k", label="nominal miss rate = 1%")
    ax.set_xticks(x); ax.set_xticklabels(t99.index, rotation=30, ha="right")
    ax.set_ylabel("Miss rate at 99% nominal (%)")
    ax.set_title("Day 5 — Where risk lives: 99%-band miss rates per ticker")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day05_tail_miss_rates.png"), dpi=130)
    plt.close()

    # 3) Width cost — what conformal's honesty costs in band width
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m, c in [("gaussian", "#c44e52"), ("conformal", "#4c72b0")]:
        sub = agg[agg.method == m].sort_values("level")
        ax.plot(sub.level, sub.width * 100, "o-", color=c, label=m)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Mean half-width (% return)")
    ax.set_title("Day 5 — Price of calibration: interval width by level")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day05_interval_width.png"), dpi=130)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Part B — predictor.py integration: run the real Flask job path
# ─────────────────────────────────────────────────────────────────────────────
def predictor_integration_check():
    print("\n[predictor.py] running run_prediction() end-to-end (AAPL)...")
    import predictor as pmod

    job_id = "day05-check"
    pmod.run_prediction(job_id, "AAPL", 10)
    job = pmod._jobs[job_id]
    if job["status"] != "done":
        raise RuntimeError(f"run_prediction failed: {job.get('error')}")
    r = job["result"]

    for key in ("interval_lower80", "interval_upper80", "interval_lower95",
                "interval_upper95", "interval_halfwidth80", "interval_rel_width80",
                "confidence", "confidence_score"):
        if key not in r:
            raise RuntimeError(f"missing field in result: {key}")

    lo80, hi80 = np.array(r["interval_lower80"]), np.array(r["interval_upper80"])
    lo95, hi95 = np.array(r["interval_lower95"]), np.array(r["interval_upper95"])
    preds = np.array(r["future_preds"])
    assert np.all(lo95 <= lo80) and np.all(hi80 <= hi95), "95% band must contain 80%"
    assert np.all(lo80 <= preds) and np.all(preds <= hi80), "point must sit inside band"
    w80 = hi80 - lo80
    assert np.all(np.diff(w80) > -1e-9), "bands must widen with horizon (sqrt(h))"

    sample = {k: v for k, v in r.items()
              if k not in ("actual_predicted_zip", "future_zip", "actual",
                           "predicted", "dates")}
    with open(os.path.join(SAMPLES, "day05_predictor_result_AAPL.json"), "w") as fh:
        json.dump(sample, fh, indent=2)
    print(f"  ok — confidence={r['confidence']} ({r['confidence_score']}), "
          f"80% halfwidth=${r['interval_halfwidth80']} "
          f"({r['interval_rel_width80']:.2%} of price), bands widen sqrt(h)")
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Part C — FastAPI live test: boot uvicorn, hit every endpoint over HTTP
# ─────────────────────────────────────────────────────────────────────────────
def api_live_test():
    import httpx

    print("\n[api] booting uvicorn on :8000...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.serving.api:app",
         "--port", "8000", "--log-level", "warning"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = "http://127.0.0.1:8000"
    out = {}
    try:
        for _ in range(60):
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        else:
            raise RuntimeError("API did not come up within 60s")

        checks = []

        r = httpx.get(f"{base}/health", timeout=30)
        out["health"] = r.json(); checks.append(("GET /health", r.status_code))

        r = httpx.post(f"{base}/predict",
                       json={"ticker": "AAPL", "horizon": 10}, timeout=300)
        out["predict"] = r.json(); checks.append(("POST /predict AAPL h=10", r.status_code))

        for model in ("persistence", "arima"):
            r = httpx.post(f"{base}/backtest",
                           json={"ticker": "SPY", "model": model,
                                 "n_folds": 5, "cost_bps": 5.0}, timeout=600)
            out[f"backtest_{model}"] = r.json()
            checks.append((f"POST /backtest SPY {model}", r.status_code))

        r = httpx.get(f"{base}/indicators/AAPL", timeout=120)
        out["indicators"] = r.json(); checks.append(("GET /indicators/AAPL", r.status_code))

        r = httpx.get(f"{base}/correlation",
                      params={"tickers": "AAPL,MSFT,SPY,NVDA"}, timeout=120)
        out["correlation"] = r.json(); checks.append(("GET /correlation x4", r.status_code))

        # Schema guards must actually refuse bad input
        r = httpx.post(f"{base}/backtest",
                       json={"ticker": "SPY", "model": "arima", "cost_bps": 0.0},
                       timeout=30)
        checks.append(("POST /backtest cost_bps=0 (must 422)", r.status_code))
        assert r.status_code == 422, "zero-cost backtest must be refused"
        r = httpx.post(f"{base}/backtest",
                       json={"ticker": "SPY", "model": "nonsense"}, timeout=30)
        checks.append(("POST /backtest bad model (must 422)", r.status_code))
        assert r.status_code == 422, "unknown model must be refused"

        for name, code in checks:
            print(f"  {code}  {name}")
        ok = [c for _, c in checks[:6]]
        if any(c != 200 for c in ok):
            raise RuntimeError(f"endpoint failure: {checks}")

        with open(os.path.join(SAMPLES, "day05_api_responses.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        out["_checks"] = checks
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─────────────────────────────────────────────────────────────────────────────
def write_metrics(df, hdf, pred_result, api_out):
    path = os.path.join(RESULTS, "metrics.json")
    blob = {}
    if os.path.exists(path):
        with open(path) as fh:
            blob = json.load(fh)

    agg = df.groupby(["method", "level"]).coverage.mean()
    cov = {m: {f"{lv:.2f}": round(float(agg[m][lv]), 4) for lv in LEVELS}
           for m in ("gaussian", "conformal")}
    bt = api_out.get("backtest_arima", {})

    blob["day05"] = {
        "day": 5,
        "phase": "3 — champion integration + production refactor + conformal intervals",
        "calibration_window_days": CALIB_WINDOW,
        "evaluation": "rolling-calibration, sequential (interval built only from "
                      "errors observed before the day it covers), on Day-3 "
                      "walk-forward OOS ARIMA streams, 10 tickers",
        "n_eval_points": int(df[df.level == 0.80].n_eval.sum() / 1),
        "empirical_coverage": cov,
        "old_heuristic": {
            "claimed_mean": round(float(hdf.claimed_confidence_mean.mean()), 4),
            "empirical_coverage_mean": round(float(hdf.empirical_coverage.mean()), 4),
            "per_ticker_gap_range": [round(float(hdf.gap.min()), 4),
                                     round(float(hdf.gap.max()), 4)],
            "verdict": "its number was never a probability of anything; "
                       "operationalised charitably it still misses its own claim",
        },
        "predictor_integration": {
            "ticker": "AAPL",
            "confidence": pred_result["confidence"],
            "confidence_score": pred_result["confidence_score"],
            "halfwidth80_usd": pred_result["interval_halfwidth80"],
            "rel_width80": pred_result["interval_rel_width80"],
            "band_monotonicity_checks": "passed (nesting, point-inside, sqrt-h widening)",
        },
        "api": {
            "endpoints_validated": ["/health", "/predict", "/backtest",
                                    "/indicators/{ticker}", "/correlation"],
            "schema_guards": "zero-cost backtest and unknown model both 422",
            "backtest_arima_SPY": {
                "strategy_sharpe": bt.get("strategy", {}).get("sharpe"),
                "buy_and_hold_sharpe": bt.get("buy_and_hold", {}).get("sharpe"),
                "beats_buy_and_hold": bt.get("beats_buy_and_hold_sharpe"),
            },
        },
        "requirements_txt": "now lists everything actually imported "
                            "(yfinance/streamlit/plotly/vaderSentiment/fastapi/...)",
    }
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)


def main():
    t0 = time.time()
    print("Part A — coverage study (10 tickers x 7 levels x ~380 rolling evals)")
    df, hdf = coverage_study()
    plots(df, hdf)

    agg = df.groupby(["method", "level"]).coverage.mean().unstack(0)
    print("\nEmpirical coverage (mean of 10 tickers):")
    print(agg.round(4).to_string())

    pred_result = predictor_integration_check()
    api_out = api_live_test()
    write_metrics(df, hdf, pred_result, api_out)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
