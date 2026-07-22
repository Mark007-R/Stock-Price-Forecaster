"""
Day 8 — Phase 6: frontier LLM comparison (designed negative result) + ablation.

The two questions this closes
-----------------------------
1. **"Why not just ask an LLM?"** Give a frontier chat model exactly what the
   specialized stack sees — a month of recent returns plus the indicator panel
   — and ask for tomorrow's direction, 200 times. The finance literature says
   this lands at a coin flip; we measure it on OUR out-of-sample days instead
   of citing someone else's. The prompts are ANONYMIZED (no ticker, no dates):
   the eval window is 2023-24, which many frontier models have memorised, so a
   dated prompt would test *recall*, not *forecasting*. Hiding identifiers is
   the contamination guard that keeps the comparison honest.
2. **"What did each sprint upgrade actually buy?"** A five-rung ablation —
   persistence → +returns target (LSTM) → +features (XGB default) → +champion
   (ARIMA) → +tuning (XGB tuned+decay) — scored on the IDENTICAL frozen
   walk-forward predictions from Day 7, so the ladder is an apples-to-apples
   decomposition, not a collage of numbers from different splits.

Protocol notes
--------------
* Stage ``prep`` writes 200 anonymized prompt lines (id + features, NO label)
  and the ground truth to SEPARATE files. The LLM answers from the prompt file
  alone; ground truth is only joined at scoring time.
* Stage ``score`` reads the LLM's predictions, scores them against the same-
  sample specialized models + the always-up baseline (binomial CIs), rebuilds
  the ablation ladder, assembles ``results/frontier_comparison.csv`` +
  ``results/ablation.csv``, and renders the Day-8 plots.
* The naive-notebook row of the frontier table (leaky scaler, price-level
  target, single holdout) is loaded from Day 1's ``baseline_metrics.json`` —
  those numbers are reported ONLY as the "what the repo used to claim"
  reference, never as a score.
* Sampling is seeded (SEED=42): 10 tickers x 5 folds x 4 OOS days = 200.
"""
import argparse
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.data.loader import load_prices                     # noqa: E402
from src.backtest.walkforward import (                      # noqa: E402
    expanding_window_folds, assert_no_peeking, rmse, directional_accuracy,
)
from src.backtest.trading import (                          # noqa: E402
    backtest_long_flat, backtest_buy_and_hold,
)
from src.models import make_context                         # noqa: E402
from src.models import arima as m_arima                     # noqa: E402
from src.models import xgb as m_xgb                         # noqa: E402

SEED = 42
TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "KO"]
START, END = "2021-01-01", "2025-01-01"          # identical span to Days 2-7
N_FOLDS = 5
COST_BPS = 5.0
PER_CELL = 4                                     # samples per (ticker, fold) cell

RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
PLOTS = os.path.join(RESULTS, "plots")
for d in (RESULTS, SAMPLES, PLOTS):
    os.makedirs(d, exist_ok=True)

OOS_FROZEN = os.path.join(SAMPLES, "day07_oos_predictions.csv")
PROMPTS_PATH = os.path.join(SAMPLES, "day08_llm_prompts.jsonl")
TRUTH_PATH = os.path.join(SAMPLES, "day08_llm_ground_truth.csv")
PREDS_PATH = os.path.join(SAMPLES, "day08_llm_predictions.json")

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers (close-only, trailing — same definitions as the app ships)
# ─────────────────────────────────────────────────────────────────────────────
def _rsi(closes: np.ndarray, window: int = 14) -> float:
    d = np.diff(closes[-(window + 1):])
    gain, loss = d[d > 0].sum() / window, -d[d < 0].sum() / window
    if loss < 1e-12:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + gain / loss))


def _pct_b(closes: np.ndarray, window: int = 20, k: float = 2.0) -> float:
    w = closes[-window:]
    mid, sd = w.mean(), w.std(ddof=0)
    if sd < 1e-12:
        return 0.5
    return float((closes[-1] - (mid - k * sd)) / (2 * k * sd))


def _macd_hist(closes: np.ndarray) -> float:
    s = pd.Series(closes)
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    return float(hist.iloc[-1] / closes[-1] * 100.0)     # % of price, scale-free


def features_at(prices: np.ndarray, i: int, pred_date: pd.Timestamp) -> dict:
    """Anonymized feature line for predicting the return realised on day i.

    Uses prices[:i] only — everything known at the close of day i-1. All
    values are scale-free (%, ratios) so nothing identifies the ticker.
    """
    c = prices[:i]
    r = c[1:] / c[:-1] - 1.0
    high252 = c[-252:].max()
    return {
        "r1": round(r[-1] * 100, 2), "r2": round(r[-2] * 100, 2),
        "r3": round(r[-3] * 100, 2), "r4": round(r[-4] * 100, 2),
        "r5": round(r[-5] * 100, 2),
        "r21": round((c[-1] / c[-22] - 1) * 100, 1),
        "r63": round((c[-1] / c[-64] - 1) * 100, 1),
        "rsi14": round(_rsi(c), 1),
        "pctb": round(_pct_b(c), 2),
        "macdh": round(_macd_hist(c), 3),
        "c_sma20": round(c[-1] / c[-20:].mean(), 3),
        "c_sma50": round(c[-1] / c[-50:].mean(), 3),
        "vol21_ann": round(r[-21:].std(ddof=1) * np.sqrt(252) * 100, 1),
        "dd252": round((c[-1] / high252 - 1) * 100, 1),
        "dow": DOW[pred_date.dayofweek],
    }


def prompt_line(sid: int, f: dict) -> str:
    return (
        f"id={sid:03d} | last5 daily ret%: [{f['r1']}, {f['r2']}, {f['r3']}, "
        f"{f['r4']}, {f['r5']}] (most recent first) | 21d ret {f['r21']}% | "
        f"63d ret {f['r63']}% | RSI14 {f['rsi14']} | BB%B {f['pctb']} | "
        f"MACDhist {f['macdh']}%px | close/SMA20 {f['c_sma20']} | "
        f"close/SMA50 {f['c_sma50']} | vol21(ann) {f['vol21_ann']}% | "
        f"off 52w-high {f['dd252']}% | next day is a {f['dow']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage: prep — build the 200-sample LLM eval set
# ─────────────────────────────────────────────────────────────────────────────
def stage_prep() -> None:
    rng = np.random.default_rng(SEED)
    frozen = pd.read_csv(OOS_FROZEN, parse_dates=["date"])
    oos_days = frozen[["ticker", "fold", "date"]].drop_duplicates()

    prompts, truth_rows = [], []
    sid = 0
    for tk in TICKERS:
        prices, dates = load_prices(tk, START, END)
        idx_of = {d: i for i, d in enumerate(pd.DatetimeIndex(dates))}
        for fold in range(N_FOLDS):
            cell = oos_days[(oos_days.ticker == tk) & (oos_days.fold == fold)]
            pick = cell.sort_values("date").iloc[
                rng.choice(len(cell), size=PER_CELL, replace=False)
            ]
            for _, row in pick.iterrows():
                i = idx_of[row.date]
                feats = features_at(prices, i, row.date)
                actual = prices[i] / prices[i - 1] - 1.0
                prompts.append({"id": sid, "prompt": prompt_line(sid, feats)})
                truth_rows.append({
                    "id": sid, "ticker": tk, "fold": fold,
                    "date": row.date.date().isoformat(),
                    "actual_ret": actual,
                    "direction": "up" if actual > 0 else ("down" if actual < 0 else "flat"),
                    **feats,
                })
                sid += 1

    with open(PROMPTS_PATH, "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")
    pd.DataFrame(truth_rows).to_csv(TRUTH_PATH, index=False)
    ups = sum(1 for t in truth_rows if t["direction"] == "up")
    print(f"prep: wrote {len(prompts)} anonymized prompts -> {PROMPTS_PATH}")
    print(f"prep: ground truth (kept separate) -> {TRUTH_PATH}  "
          f"[base rate up = {ups}/{len(truth_rows)} = {ups/len(truth_rows):.3f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation ladder — frozen Day-7 predictions + a fresh default-XGB run
# ─────────────────────────────────────────────────────────────────────────────
RUNGS = [
    # (rung label, what it adds, model key in the frozen frame or special)
    ("1_persistence",   "random walk r̂=0 (RMSE floor, takes no position)", "zero"),
    ("2_returns_lstm",  "+returns target (LSTM on returns, Day 2)",        "lstm_returns"),
    ("3_features_xgb",  "+18 no-look-ahead features (XGB default, Day 3)", "xgb_default"),
    ("4_champion_arima", "+champion forecaster (ARIMA by AIC, Day 3)",     "arima"),
    ("5_tuned_xgb",     "+Optuna tuning + time-decay weights (Day 6)",     "xgb_tuned_decay"),
]


def run_xgb_default() -> tuple[pd.DataFrame, float]:
    """Default-XGB predictions on the identical folds (not in the frozen file)."""
    rows, secs = [], 0.0
    for tk in TICKERS:
        prices, dates = load_prices(tk, START, END)
        folds = expanding_window_folds(len(prices), n_folds=N_FOLDS)
        assert_no_peeking(folds, n_samples=len(prices))
        ctx = make_context(prices, dates, with_features=True)
        ret = np.zeros_like(prices)
        ret[1:] = prices[1:] / prices[:-1] - 1.0
        for f in folds:
            t0 = time.time()
            pred = np.asarray(m_xgb.predict_fold(prices, f, ctx), dtype=float)
            secs += time.time() - t0
            for k in range(f.test_len):
                rows.append({"model": "xgb_default", "ticker": tk, "fold": f.fold,
                             "date": dates[f.test_start + k],
                             "pred": pred[k], "actual": ret[f.test_start + k]})
        print(f"  xgb_default [{tk}] done")
    return pd.DataFrame(rows), secs


def score_rung(df: pd.DataFrame) -> dict:
    """Pooled + per-ticker metrics for one model's OOS frame."""
    pred, act = df["pred"].to_numpy(), df["actual"].to_numpy()
    zero_rmse = rmse(act, np.zeros_like(act))
    sharpes, bh_sharpes = [], []
    for tk, g in df.sort_values("date").groupby("ticker"):
        sharpes.append(backtest_long_flat(g["pred"].to_numpy(), g["actual"].to_numpy(),
                                          cost_bps=COST_BPS).sharpe)
        bh_sharpes.append(backtest_buy_and_hold(g["actual"].to_numpy(),
                                                cost_bps=COST_BPS).sharpe)
    return {
        "rmse_ret": rmse(act, pred),
        "rmse_vs_zero": rmse(act, pred) / zero_rmse,
        "dir_acc": directional_accuracy(pred, act),
        "sharpe_net": float(np.mean(sharpes)),
        "bh_sharpe_net": float(np.mean(bh_sharpes)),
        "n_days": len(df),
    }


def build_ablation(frozen: pd.DataFrame, xgb_default: pd.DataFrame) -> pd.DataFrame:
    base = frozen[frozen.model == "arima"][["ticker", "fold", "date", "actual"]].copy()
    act = base["actual"].to_numpy()
    always_up = directional_accuracy(np.ones_like(act), act)

    rows = []
    for label, adds, key in RUNGS:
        if key == "zero":
            df = base.assign(model="persistence", pred=0.0)
        elif key == "xgb_default":
            df = xgb_default
        else:
            df = frozen[frozen.model == key]
        rows.append({"rung": label, "adds": adds, **score_rung(df)})
    ab = pd.DataFrame(rows)
    ab["always_up_dir_acc"] = always_up
    return ab


# ─────────────────────────────────────────────────────────────────────────────
# Stage: score — LLM scoring + frontier table + plots
# ─────────────────────────────────────────────────────────────────────────────
def binom_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def binom_p_two_sided(k: int, n: int) -> float:
    from scipy.stats import binomtest
    return float(binomtest(k, n, 0.5).pvalue)


def stage_score() -> None:
    truth = pd.read_csv(TRUTH_PATH, parse_dates=["date"])
    with open(PREDS_PATH, encoding="utf-8") as fh:
        preds = {int(k): v for k, v in json.load(fh).items()}
    assert len(preds) == len(truth), f"{len(preds)} preds for {len(truth)} samples"

    truth["llm_dir"] = truth["id"].map(lambda i: preds[i]["direction"])
    truth["llm_conf"] = truth["id"].map(lambda i: float(preds[i]["confidence"]))
    scored = truth[truth.direction != "flat"].copy()
    scored["llm_hit"] = (scored.llm_dir == scored.direction)

    n, k = len(scored), int(scored.llm_hit.sum())
    acc = k / n
    lo, hi = binom_ci(k, n)
    pval = binom_p_two_sided(k, n)
    up_rate = float((scored.direction == "up").mean())
    llm_up_rate = float((scored.llm_dir == "up").mean())
    # return actually captured by trading each LLM call vs owning every day
    sgn = scored.llm_dir.map({"up": 1.0, "down": -1.0}).to_numpy()
    mean_captured = float(np.mean(sgn * scored.actual_ret.to_numpy()))
    mean_bh = float(scored.actual_ret.mean())

    print(f"LLM zero-shot: {k}/{n} = {acc:.3f}  CI95 [{lo:.3f}, {hi:.3f}]  "
          f"p(vs coin)={pval:.3f}")
    print(f"  base rate up {up_rate:.3f} | LLM said up {llm_up_rate:.3f} | "
          f"mean ret captured {mean_captured*1e4:.1f} bps/day vs B&H {mean_bh*1e4:.1f}")

    # specialized models on the IDENTICAL 200 sampled days
    frozen = pd.read_csv(OOS_FROZEN, parse_dates=["date"])
    xgb_default, xgb_secs = run_xgb_default()
    xgb_days = int(len(xgb_default))
    key = scored[["ticker", "date", "actual_ret", "llm_hit"]].copy()
    same_sample = {"llm_zero_shot": {"acc": acc, "n": n, "lo": lo, "hi": hi}}
    for name, frame in [("arima", frozen[frozen.model == "arima"]),
                        ("xgb_tuned_decay", frozen[frozen.model == "xgb_tuned_decay"]),
                        ("lstm_returns", frozen[frozen.model == "lstm_returns"]),
                        ("patchtst", frozen[frozen.model == "patchtst"]),
                        ("xgb_default", xgb_default)]:
        j = key.merge(frame[["ticker", "date", "pred"]], on=["ticker", "date"])
        hits = int((np.sign(j["pred"]) == np.sign(j["actual_ret"])).sum())
        l2, h2 = binom_ci(hits, len(j))
        same_sample[name] = {"acc": hits / len(j), "n": len(j), "lo": l2, "hi": h2}
    au_hits = int((scored.direction == "up").sum())
    l3, h3 = binom_ci(au_hits, n)
    same_sample["always_up"] = {"acc": au_hits / n, "n": n, "lo": l3, "hi": h3}
    ss = (pd.DataFrame(same_sample).T
          .rename_axis("method").reset_index()
          .sort_values("acc", ascending=False))
    ss.to_csv(os.path.join(RESULTS, "phase6_same_sample.csv"), index=False)
    print("\nsame-200-sample directional accuracy:")
    print(ss.to_string(index=False))

    # confidence calibration buckets
    bins = [0.0, 0.55, 0.65, 1.01]
    scored["conf_bucket"] = pd.cut(scored.llm_conf, bins,
                                   labels=["<=0.55", "0.55-0.65", ">0.65"])
    calib = (scored.groupby("conf_bucket", observed=True)
             .agg(n=("llm_hit", "size"), acc=("llm_hit", "mean"))
             .reset_index())
    calib.to_csv(os.path.join(RESULTS, "phase6_llm_calibration.csv"), index=False)
    print("\nLLM accuracy by stated confidence:")
    print(calib.to_string(index=False))

    # ablation ladder on the full 5,000-day frozen OOS set
    ablation = build_ablation(frozen, xgb_default)
    ablation.to_csv(os.path.join(RESULTS, "ablation.csv"), index=False)
    print("\nablation ladder (5,000 frozen OOS days):")
    print(ablation.drop(columns=["adds"]).to_string(index=False))

    # honest latency for the frontier table: one ARIMA fold, amortised per day
    tk = "AAPL"
    prices, dates = load_prices(tk, START, END)
    folds = expanding_window_folds(len(prices), n_folds=N_FOLDS)
    ctx = make_context(prices, dates, with_features=True)
    t0 = time.time()
    m_arima.predict_fold(prices, folds[-1], ctx)
    arima_ms_day = (time.time() - t0) / folds[-1].test_len * 1000
    xgb_ms_day = xgb_secs / max(xgb_days, 1) * 1000

    day1 = json.load(open(os.path.join(RESULTS, "baseline_metrics.json")))
    ab = ablation.set_index("rung")
    frontier = pd.DataFrame([
        {"system": "naive_notebook (pre-sprint)",
         "setup": "LSTM on price level, scaler fit on FULL series (leaky), single 80/20 holdout",
         "dir_acc": day1["mean_dir_acc_fixed"],
         "rmse_vs_zero": np.nan,
         "sharpe_net_5bps": np.nan,
         "latency_ms_per_pred": np.nan,
         "cost_per_pred_usd": 0.0,
         "notes": (f"claimed price-RMSE {day1['mean_rmse_leaky']:.2f} was leakage-inflated; "
                   f"honest {day1['mean_rmse_fixed_HONEST']:.2f}; persistence "
                   f"{day1['mean_rmse_persistence_baseline']:.2f} beats it on 10/10 tickers")},
        {"system": "full_pipeline (this sprint)",
         "setup": "returns target, leakage-fixed scaling, 18 causal features, ARIMA champion, "
                  "conformal intervals, walk-forward, costs",
         "dir_acc": ab.loc["4_champion_arima", "dir_acc"],
         "rmse_vs_zero": ab.loc["4_champion_arima", "rmse_vs_zero"],
         "sharpe_net_5bps": ab.loc["4_champion_arima", "sharpe_net"],
         "latency_ms_per_pred": round(arima_ms_day, 1),
         "cost_per_pred_usd": 0.0,
         "notes": f"honest verdict: still loses to B&H Sharpe {ab.loc['4_champion_arima','bh_sharpe_net']:.2f}"},
        {"system": "Claude (this session), zero-shot",
         "setup": "anonymized 30d returns + indicator panel, 200 OOS samples, no catalog of history",
         "dir_acc": acc,
         "rmse_vs_zero": np.nan,
         "sharpe_net_5bps": np.nan,
         "latency_ms_per_pred": 2000.0,
         "cost_per_pred_usd": 0.02,
         "notes": (f"CI95 [{lo:.3f},{hi:.3f}], p={pval:.2f} vs coin; latency/cost are "
                   "ESTIMATES (in-session run); mean captured "
                   f"{mean_captured*1e4:.1f} bps/day vs B&H {mean_bh*1e4:.1f}")},
        {"system": "GPT-5.4, zero-shot",
         "setup": "not run — no API key in the autonomous environment",
         "dir_acc": np.nan, "rmse_vs_zero": np.nan, "sharpe_net_5bps": np.nan,
         "latency_ms_per_pred": np.nan, "cost_per_pred_usd": np.nan,
         "notes": "reported as not-run rather than estimated"},
    ])
    frontier.to_csv(os.path.join(RESULTS, "frontier_comparison.csv"), index=False)
    print("\nfrontier comparison written -> results/frontier_comparison.csv")
    print(f"  (xgb_default amortised latency: {xgb_ms_day:.2f} ms/day)")

    make_plots(ss, calib, ablation, scored)

    # samples + metrics.json
    with open(PROMPTS_PATH, encoding="utf-8") as fh:
        prompt_map = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in fh}
    ex = scored.sample(10, random_state=SEED)
    examples = [{"id": int(r.id), "prompt": prompt_map[int(r.id)],
                 "llm_direction": r.llm_dir, "llm_confidence": r.llm_conf,
                 "actual_direction": r.direction,
                 "actual_ret_pct": round(r.actual_ret * 100, 2),
                 "hit": bool(r.llm_hit)} for r in ex.itertuples()]
    with open(os.path.join(SAMPLES, "day08_llm_examples.json"), "w",
              encoding="utf-8") as fh:
        json.dump(examples, fh, indent=1)

    mpath = os.path.join(RESULTS, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    metrics["day08"] = {
        "llm": {"n_scored": n, "hits": k, "dir_acc": acc, "ci95": [lo, hi],
                "p_vs_coin": pval, "up_call_rate": llm_up_rate,
                "base_up_rate": up_rate,
                "mean_captured_bps": mean_captured * 1e4,
                "mean_bh_bps": mean_bh * 1e4,
                "latency_cost": "estimates — in-session run",
                "contamination_guard": "prompts anonymized: no ticker, no dates"},
        "same_sample_dir_acc": {r.method: r.acc for r in ss.itertuples()},
        "ablation": ablation.drop(columns=["adds"]).to_dict("records"),
        "gpt_5_4": "not run (no API key)",
    }
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)
    print("metrics.json updated (day08)")


def make_plots(ss: pd.DataFrame, calib: pd.DataFrame,
               ablation: pd.DataFrame, scored: pd.DataFrame) -> None:
    # 1 — same-sample directional accuracy with CIs
    fig, ax = plt.subplots(figsize=(9, 5))
    order = ss.sort_values("acc")
    y = np.arange(len(order))
    err = np.vstack([order.acc - order.lo, order.hi - order.acc])
    colors = ["#d62728" if m == "llm_zero_shot" else
              ("#2ca02c" if m == "always_up" else "#1f77b4") for m in order.method]
    ax.barh(y, order.acc, xerr=err, color=colors, alpha=0.85, capsize=3)
    ax.axvline(0.5, color="k", ls="--", lw=1, label="coin flip")
    ax.set_yticks(y, order.method)
    ax.set_xlabel("directional accuracy (same 200 OOS samples, Wilson 95% CI)")
    ax.set_title("Day 8 — every forecaster vs a coin and vs 'always up'")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "day08_same_sample_diracc.png"), dpi=150)
    plt.close(fig)

    # 2 — LLM accuracy by stated confidence
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(calib.conf_bucket.astype(str), calib.acc, color="#d62728", alpha=0.85)
    for i, r in calib.iterrows():
        ax.text(i, r.acc + 0.01, f"n={r.n}", ha="center", fontsize=9)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_ylim(0, max(0.7, calib.acc.max() + 0.08))
    ax.set_xlabel("LLM stated confidence")
    ax.set_ylabel("hit rate")
    ax.set_title("Does the LLM know when it knows? (calibration by confidence)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "day08_llm_calibration.png"), dpi=150)
    plt.close(fig)

    # 3 — ablation ladder
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = ["1\npersistence", "2\n+returns\n(LSTM)", "3\n+features\n(XGB)",
              "4\n+champion\n(ARIMA)", "5\n+tuning\n(XGB)"]
    axes[0].bar(labels, ablation.rmse_vs_zero, color="#1f77b4", alpha=0.85)
    axes[0].axhline(1.0, color="k", ls="--", lw=1, label="predict-zero floor")
    axes[0].set_title("RMSE vs zero (lower=better)")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, ablation.dir_acc, color="#1f77b4", alpha=0.85)
    axes[1].axhline(ablation.always_up_dir_acc.iloc[0], color="#2ca02c", ls="--",
                    lw=1.2, label=f"always-up {ablation.always_up_dir_acc.iloc[0]:.3f}")
    axes[1].set_title("directional accuracy")
    axes[1].legend(fontsize=8)
    axes[2].bar(labels, ablation.sharpe_net, color="#1f77b4", alpha=0.85)
    axes[2].axhline(ablation.bh_sharpe_net.iloc[0], color="#2ca02c", ls="--",
                    lw=1.2, label=f"buy-and-hold {ablation.bh_sharpe_net.iloc[0]:.2f}")
    axes[2].set_title(f"Sharpe net {COST_BPS:.0f} bps")
    axes[2].legend(fontsize=8)
    for a in axes:
        a.tick_params(axis="x", labelsize=8)
    fig.suptitle("Day 8 — what each sprint upgrade bought (5,000 frozen OOS days)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "day08_ablation_ladder.png"), dpi=150)
    plt.close(fig)

    # 4 — LLM call mix vs reality
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ct = pd.crosstab(scored.llm_dir, scored.direction)
    im = ax.imshow(ct.values, cmap="Blues")
    ax.set_xticks(range(ct.shape[1]), ct.columns)
    ax.set_yticks(range(ct.shape[0]), ct.index)
    for r in range(ct.shape[0]):
        for c in range(ct.shape[1]):
            ax.text(c, r, ct.values[r, c], ha="center", va="center",
                    color="black", fontsize=12)
    ax.set_xlabel("actual next-day direction")
    ax.set_ylabel("LLM call")
    ax.set_title("LLM zero-shot: calls vs outcomes (200 OOS samples)")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "day08_llm_confusion.png"), dpi=150)
    plt.close(fig)
    print("plots saved -> results/plots/day08_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "score"], required=True)
    args = ap.parse_args()
    np.random.seed(SEED)
    if args.stage == "prep":
        stage_prep()
    else:
        stage_score()
