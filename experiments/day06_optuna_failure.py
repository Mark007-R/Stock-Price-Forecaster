"""
Day 6 — Phase 4: Optuna sweep on the champion learner + failure-mode analysis.

The question
------------
Days 3-4 left one objection standing against the ML side of the bake-off:
*"you ran XGBoost on default hyperparameters — of course it lost."* Day 6
answers it with a proper 40-trial Optuna sweep, tuned the only honest way a
time series allows: every trial is scored on an inner-validation window carved
off the END of each walk-forward fold's TRAIN slice. The outer test blocks are
never touched during tuning, so the final re-evaluation is as out-of-sample as
Day 3's was.

Then the part tuning cannot answer: WHERE does the model fail? The Day-3
out-of-sample predictions (saved per day, per method) are sliced three ways —
by market regime (bull / bear / sideways from the trailing 63-day return), by
volatility bucket (trailing 21-day vol terciles), and by ticker sector — to
find where directional accuracy actually collapses. The failure map then picks
the targeted fix, from the two candidates Day 4 did not already rule out
(regime *features* were tried and made things worse): vol-scaled targets and
time-decay sample weighting, each re-run through the identical walk-forward.

Tuning protocol (no peeking)
----------------------------
* Outer folds: the exact expanding-window folds of Days 2-5.
* Inner validation: the last 120 feature rows of each outer fold's train
  slice. The model fits on train-minus-inner, is scored on inner. Test rows
  are untouched.
* Objective: mean inner RMSE(next-day returns) across 5 tuning tickers x 5
  folds (25 evaluations per trial). The other 5 tickers never inform tuning
  and act as a semi-held-out check in the final table.
* Reference on the SAME inner windows: the random walk (predict 0). A trial
  only "learned" something if it beats that number.

Public yfinance data only (cached by src.data.loader). Costs never hidden.
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

import optuna                                     # noqa: E402
from optuna.samplers import TPESampler            # noqa: E402
from xgboost import XGBRegressor                  # noqa: E402

from src.data.loader import load_prices           # noqa: E402
from src.backtest.walkforward import (            # noqa: E402
    expanding_window_folds, assert_no_peeking, rmse, directional_accuracy,
)
from src.backtest.trading import (                # noqa: E402
    backtest_long_flat, backtest_buy_and_hold,
)
from src.features.engineer import (               # noqa: E402
    build_feature_frame, assert_no_lookahead, FEATURE_COLS,
)

SEED = 42
TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "KO"]
# Tuning sees only these 5 (index, mega-cap tech, high-vol semi, bank, staple);
# the other 5 are a semi-held-out sanity check on the tuned config.
TUNE_TICKERS = ["SPY", "AAPL", "NVDA", "JPM", "KO"]
START, END = "2021-01-01", "2025-01-01"           # identical span to Days 2-5
N_FOLDS = 5
COST_BPS = 5.0
INNER = 120                                       # inner-validation tail (rows)
N_TRIALS = 40
DECAY_HALF_LIFE = 126                             # trading days (~6 months)

SECTOR = {
    "SPY": "index", "AAPL": "tech", "MSFT": "tech", "NVDA": "tech",
    "GOOGL": "comm-svcs", "META": "comm-svcs", "AMZN": "consumer-disc",
    "JPM": "financials", "XOM": "energy", "KO": "staples",
}

DEFAULT_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, n_jobs=4, verbosity=0,
)

RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
PLOTS = os.path.join(RESULTS, "plots")
for d in (RESULTS, SAMPLES, PLOTS):
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared data prep — one feature frame + fold set per ticker, reused everywhere
# ─────────────────────────────────────────────────────────────────────────────
def prepare(tickers):
    data = {}
    for tk in tickers:
        prices, dates = load_prices(tk, START, END)
        folds = expanding_window_folds(len(prices), n_folds=N_FOLDS)
        assert_no_peeking(folds, n_samples=len(prices))
        assert_no_lookahead(prices, probe_at=folds[0].train_end)
        data[tk] = {
            "prices": prices, "dates": dates, "folds": folds,
            "feats": build_feature_frame(prices, dates),
        }
        print(f"  [{tk}] n={len(prices)} folds={len(folds)} (look-ahead check passed)")
    return data


def xgb_rows(feats, fold):
    """Train/test feature-row indices under the Day-3 leakage protocol:
    row t's target is day t+1's return, so training stops at train_end-2."""
    tr = np.asarray(feats.index[(feats["valid"]) & (feats.index <= fold.train_end - 2)])
    te = np.arange(fold.test_start - 1, fold.test_end - 1)
    return tr, te


def fit_predict(feats, fit_rows, pred_rows, params,
                vol_scale=False, decay_half_life=None):
    """Fit XGB on ``fit_rows``, predict next-day returns at ``pred_rows``.

    vol_scale: train on target / trailing vol_21 (both known at row t), then
    rescale predictions by the prediction row's own trailing vol — a
    homoskedastic target, undone with information available at forecast time.
    decay_half_life: exponential time-decay sample weights (recent rows count
    more); the newest training row has weight 1.
    """
    X = feats.loc[fit_rows, FEATURE_COLS].to_numpy(dtype=float)
    y = feats.loc[fit_rows, "target"].to_numpy(dtype=float)

    w = None
    if decay_half_life is not None:
        age = fit_rows.max() - fit_rows.astype(float)
        w = 0.5 ** (age / float(decay_half_life))
    if vol_scale:
        v = np.maximum(feats.loc[fit_rows, "vol_21"].to_numpy(dtype=float), 1e-6)
        y = y / v

    model = XGBRegressor(**params)
    model.fit(X, y, sample_weight=w)

    Xp = feats.loc[pred_rows, FEATURE_COLS].to_numpy(dtype=float)
    pred = model.predict(Xp).astype(float)
    if vol_scale:
        vp = np.maximum(feats.loc[pred_rows, "vol_21"].to_numpy(dtype=float), 1e-6)
        pred = pred * vp
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Optuna sweep (inner-validation on train slices only)
# ─────────────────────────────────────────────────────────────────────────────
def inner_splits(data):
    """(ticker, fit_rows, val_rows) triples — val is the train slice's tail."""
    out = []
    for tk in TUNE_TICKERS:
        feats, folds = data[tk]["feats"], data[tk]["folds"]
        for f in folds:
            tr, _ = xgb_rows(feats, f)
            if len(tr) <= INNER + 50:
                raise RuntimeError(f"{tk} fold {f.fold}: train too small for inner split")
            out.append((tk, tr[:-INNER], tr[-INNER:]))
    return out


def run_optuna(data):
    splits = inner_splits(data)

    # Random-walk floor on the SAME inner windows — the number a trial must
    # beat before "tuned" means anything.
    zero_rmses = []
    for tk, _, val in splits:
        yv = data[tk]["feats"].loc[val, "target"].to_numpy(dtype=float)
        zero_rmses.append(float(np.sqrt(np.mean(yv ** 2))))
    zero_floor = float(np.mean(zero_rmses))
    print(f"\nInner-val random-walk floor (predict 0): RMSE {zero_floor:.6f}")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 600),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 64.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            random_state=SEED, n_jobs=4, verbosity=0,
        )
        rmses, abs_preds = [], []
        for tk, fit_rows, val_rows in splits:
            feats = data[tk]["feats"]
            pred = fit_predict(feats, fit_rows, val_rows, params)
            yv = feats.loc[val_rows, "target"].to_numpy(dtype=float)
            rmses.append(float(np.sqrt(np.mean((pred - yv) ** 2))))
            abs_preds.append(float(np.mean(np.abs(pred))))
        # How much forecast the config actually emits — shrinkage telemetry.
        trial.set_user_attr("mean_abs_pred", float(np.mean(abs_preds)))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"Optuna: {N_TRIALS} trials in {(time.time()-t0)/60:.1f} min")

    tdf = study.trials_dataframe()
    tdf.to_csv(os.path.join(RESULTS, "phase4_optuna_trials.csv"), index=False)

    # Default config scored on the identical inner windows, for the delta.
    default_inner = np.mean([
        np.sqrt(np.mean((fit_predict(data[tk]["feats"], fr, vr, DEFAULT_PARAMS)
                         - data[tk]["feats"].loc[vr, "target"].to_numpy(float)) ** 2))
        for tk, fr, vr in splits])
    default_abs = np.mean([
        np.mean(np.abs(fit_predict(data[tk]["feats"], fr, vr, DEFAULT_PARAMS)))
        for tk, fr, vr in splits])

    best = study.best_trial
    print(f"Best trial #{best.number}: inner RMSE {best.value:.6f} "
          f"(default {default_inner:.6f}, zero floor {zero_floor:.6f})")
    print(f"  mean |prediction|: best {best.user_attrs['mean_abs_pred']:.6f} "
          f"vs default {default_abs:.6f}")
    print("  best params:", {k: (round(v, 5) if isinstance(v, float) else v)
                             for k, v in best.params.items()})

    # fANOVA-style importances (which knobs actually moved the objective).
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        imp = {}

    tuned_params = dict(best.params, random_state=SEED, n_jobs=4, verbosity=0)
    return study, tuned_params, zero_floor, float(default_inner), float(default_abs), imp


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Failure-mode analysis of the Day-3 OOS predictions
# ─────────────────────────────────────────────────────────────────────────────
FM_METHODS = ["always_up", "momentum", "arima", "prophet", "xgboost",
              "lstm_returns"]

def regime_table(data):
    """Per-ticker, per-OOS-day regime/vol labels, all trailing (shift(1) so a
    day's label uses information through the PREVIOUS close only)."""
    frames = []
    for tk in TICKERS:
        p = pd.Series(data[tk]["prices"], index=pd.DatetimeIndex(data[tk]["dates"]))
        ret = p.pct_change()
        mom63 = p.pct_change(63).shift(1)
        vol21 = ret.rolling(21).std().shift(1)
        lab = pd.DataFrame({"date": p.index, "mom63": mom63.values,
                            "vol21": vol21.values})
        lab["ticker"] = tk
        frames.append(lab)
    out = pd.concat(frames, ignore_index=True)
    out["regime"] = np.select(
        [out["mom63"] > 0.05, out["mom63"] < -0.05],
        ["bull", "bear"], default="sideways")
    return out


def failure_modes(data, tuned_oos):
    labels = regime_table(data)
    rows = []
    for tk in TICKERS:
        path = os.path.join(SAMPLES, f"day03_{tk}_oos_predictions.csv")
        df = pd.read_csv(path, parse_dates=["date"])
        df["ticker"] = tk
        # append the freshly-tuned model's OOS stream on the identical days
        t = tuned_oos[tk]
        assert len(t["pred"]) == len(df), f"{tk}: tuned OOS length mismatch"
        df["pred_xgboost_tuned"] = t["pred"]
        rows.append(df)
    oos = pd.concat(rows, ignore_index=True)
    oos = oos.merge(labels[["ticker", "date", "regime", "vol21"]],
                    on=["ticker", "date"], how="left")
    # Vol terciles PER TICKER over its own OOS window. Diagnostic slicing only
    # (bucket edges use the full window) — never fed back into any model.
    oos["vol_bucket"] = (oos.groupby("ticker")["vol21"]
                            .transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"])))
    oos["sector"] = oos["ticker"].map(SECTOR)
    oos["up_day"] = (oos["actual_ret"] > 0).astype(float)

    methods = FM_METHODS + ["xgboost_tuned"]

    def bucket_stats(g):
        out = {"n_days": len(g), "up_rate": float(g["up_day"].mean())}
        for m in methods:
            out[f"acc_{m}"] = directional_accuracy(
                g[f"pred_{m}"].to_numpy(float), g["actual_ret"].to_numpy(float))
        return pd.Series(out)

    by_regime = oos.groupby("regime").apply(bucket_stats).reset_index()
    by_vol = oos.groupby("vol_bucket", observed=True).apply(bucket_stats).reset_index()
    by_sector = oos.groupby("sector").apply(bucket_stats).reset_index()
    for frame, key in ((by_regime, "regime"), (by_vol, "vol_bucket"),
                       (by_sector, "sector")):
        frame.insert(0, "slice", key)
        frame.rename(columns={key: "bucket"}, inplace=True)
    fm = pd.concat([by_regime, by_vol, by_sector], ignore_index=True)
    # Edge vs the no-skill directional baseline INSIDE the bucket: always-up's
    # accuracy in a bucket IS the bucket's up-rate, so edge = acc - up_rate.
    for m in methods:
        fm[f"edge_{m}"] = fm[f"acc_{m}"] - fm["up_rate"]
    fm.to_csv(os.path.join(RESULTS, "phase4_failure_modes.csv"), index=False)
    return oos, fm


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — final walk-forward re-evaluation: default vs tuned vs targeted fixes
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_configs(data, tuned_params):
    configs = {
        "xgb_default":      dict(params=DEFAULT_PARAMS),
        "xgb_tuned":        dict(params=tuned_params),
        "xgb_tuned_volscale": dict(params=tuned_params, vol_scale=True),
        "xgb_tuned_decay":  dict(params=tuned_params, decay_half_life=DECAY_HALF_LIFE),
    }
    fold_rows, trade_rows = [], []
    tuned_oos = {}
    for tk in TICKERS:
        feats, folds, prices = data[tk]["feats"], data[tk]["folds"], data[tk]["prices"]
        dates = data[tk]["dates"]
        streams = {c: [] for c in configs}
        actual_stream, date_stream = [], []
        for f in folds:
            tr, te = xgb_rows(feats, f)
            actual = prices[f.test_start:f.test_end] / prices[f.test_start - 1:f.test_end - 1] - 1.0
            zero_rmse = rmse(actual, np.zeros_like(actual))
            actual_stream.append(actual)
            date_stream.append(np.asarray(dates[f.test_start:f.test_end]))
            for name, cfg in configs.items():
                t0 = time.time()
                pred = fit_predict(feats, tr, te, cfg["params"],
                                   vol_scale=cfg.get("vol_scale", False),
                                   decay_half_life=cfg.get("decay_half_life"))
                fold_rows.append({
                    "ticker": tk, "fold": f.fold, "config": name,
                    "rmse_ret": rmse(actual, pred),
                    "rmse_vs_zero": rmse(actual, pred) / zero_rmse,
                    "dir_acc": directional_accuracy(pred, actual),
                    "mean_abs_pred": float(np.mean(np.abs(pred))),
                    "fit_secs": round(time.time() - t0, 2),
                })
                streams[name].append(pred)
        actual_all = np.concatenate(actual_stream)
        bh = backtest_buy_and_hold(actual_all, cost_bps=COST_BPS)
        trade_rows.append({"ticker": tk, "config": "buy_and_hold", **bh.as_dict()})
        for name in configs:
            pred_all = np.concatenate(streams[name])
            bt = backtest_long_flat(pred_all, actual_all, cost_bps=COST_BPS)
            trade_rows.append({"ticker": tk, "config": name, **bt.as_dict()})
        tuned_oos[tk] = {"pred": np.concatenate(streams["xgb_tuned"]),
                         "actual": actual_all,
                         "date": np.concatenate(date_stream)}
        print(f"  [{tk}] configs done  b&h sharpe={bh.sharpe:.2f}")

    fdf = pd.DataFrame(fold_rows)
    tdf = pd.DataFrame(trade_rows)
    fdf.to_csv(os.path.join(RESULTS, "phase4_configs_folds.csv"), index=False)
    tdf.to_csv(os.path.join(RESULTS, "phase4_configs_trading.csv"), index=False)

    lb = (fdf.groupby("config")
             .agg(mean_rmse_ret=("rmse_ret", "mean"),
                  mean_rmse_vs_zero=("rmse_vs_zero", "mean"),
                  mean_dir_acc=("dir_acc", "mean"),
                  mean_abs_pred=("mean_abs_pred", "mean"),
                  mean_fit_secs=("fit_secs", "mean"))
             .join(tdf[tdf.config != "buy_and_hold"].groupby("config")
                      .agg(mean_sharpe=("sharpe", "mean"),
                           mean_total_return=("total_return", "mean"),
                           mean_exposure=("exposure", "mean"),
                           mean_trades=("n_trades", "mean")))
             .reset_index())
    bh_sharpe = float(tdf[tdf.config == "buy_and_hold"]["sharpe"].mean())
    lb["sharpe_vs_bh"] = lb["mean_sharpe"] - bh_sharpe
    lb.to_csv(os.path.join(RESULTS, "phase4_leaderboard.csv"), index=False)
    return fdf, tdf, lb, bh_sharpe, tuned_oos


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plots(study, zero_floor, default_inner, imp, lb, fm, fdf):
    # 1) Optuna optimisation history vs the random-walk floor
    vals = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.minimum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(range(len(vals)), vals, s=18, alpha=0.6, label="trial")
    ax.plot(best_so_far, c="#c44e52", lw=1.6, label="best so far")
    ax.axhline(zero_floor, ls="--", c="k",
               label=f"random walk (predict 0) = {zero_floor:.5f}")
    ax.axhline(default_inner, ls=":", c="#4c72b0",
               label=f"default params = {default_inner:.5f}")
    ax.set_xlabel("trial"); ax.set_ylabel("inner-validation RMSE (returns)")
    ax.set_title("Day 6 — Optuna on XGBoost: 40 trials vs the random-walk floor")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day06_optuna_history.png"), dpi=130)
    plt.close()

    # 2) parameter importances
    if imp:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        keys = list(imp.keys()); v = [imp[k] for k in keys]
        ax.barh(keys[::-1], v[::-1], color="#55a868")
        ax.set_xlabel("fANOVA importance (share of objective variance)")
        ax.set_title("Day 6 — Which hyperparameters mattered")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, "day06_param_importance.png"), dpi=130)
        plt.close()

    # 3) config leaderboard: RMSE-vs-zero + dir-acc
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    order = ["xgb_default", "xgb_tuned", "xgb_tuned_volscale", "xgb_tuned_decay"]
    sub = lb.set_index("config").reindex(order)
    axes[0].bar(order, sub["mean_rmse_vs_zero"], color="#8172b2")
    axes[0].axhline(1.0, ls="--", c="k", label="random-walk parity")
    axes[0].set_ylabel("RMSE ratio vs predict-zero (lower is better)")
    axes[0].legend(); axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(order, sub["mean_dir_acc"], color="#ccb974")
    axes[1].axhline(0.549, ls="--", c="k", label="always-up baseline (0.549)")
    axes[1].set_ylabel("directional accuracy"); axes[1].legend()
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim(0.45, 0.58)
    fig.suptitle("Day 6 — Tuning + targeted fixes vs the two no-skill baselines")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day06_tuned_vs_default.png"), dpi=130)
    plt.close()

    # 4) failure modes: dir-acc by regime
    fm_r = fm[fm["slice"] == "regime"].set_index("bucket").reindex(
        ["bull", "sideways", "bear"])
    methods = ["always_up", "arima", "xgboost", "xgboost_tuned", "lstm_returns"]
    x = np.arange(len(fm_r)); width = 0.16
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, m in enumerate(methods):
        ax.bar(x + (i - 2) * width, fm_r[f"acc_{m}"], width, label=m)
    for xi, (b, row) in enumerate(fm_r.iterrows()):
        ax.hlines(row["up_rate"], xi - 0.42, xi + 0.42, colors="k",
                  linestyles="--", lw=1.2)
    ax.hlines([], [], [], colors="k", linestyles="--", label="bucket up-rate (no-skill)")
    ax.axhline(0.5, c="grey", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={int(fm_r.loc[b,'n_days'])})"
                                          for b in fm_r.index])
    ax.set_ylabel("directional accuracy")
    ax.set_title("Day 6 — Where accuracy lives: by market regime (trailing 63-day return)")
    ax.legend(fontsize=8, ncol=3); ax.set_ylim(0.30, 0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day06_failure_regime.png"), dpi=130)
    plt.close()

    # 5) failure modes: by volatility bucket
    fm_v = fm[fm["slice"] == "vol_bucket"].set_index("bucket").reindex(
        ["low", "mid", "high"])
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, m in enumerate(methods):
        ax.bar(np.arange(len(fm_v)) + (i - 2) * width, fm_v[f"acc_{m}"], width, label=m)
    for xi, (b, row) in enumerate(fm_v.iterrows()):
        ax.hlines(row["up_rate"], xi - 0.42, xi + 0.42, colors="k",
                  linestyles="--", lw=1.2)
    ax.axhline(0.5, c="grey", lw=0.7)
    ax.set_xticks(np.arange(len(fm_v)))
    ax.set_xticklabels([f"{b} vol\n(n={int(fm_v.loc[b,'n_days'])})" for b in fm_v.index])
    ax.set_ylabel("directional accuracy")
    ax.set_title("Day 6 — By trailing 21-day volatility tercile (dashed = bucket up-rate)")
    ax.legend(fontsize=8, ncol=3); ax.set_ylim(0.30, 0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day06_failure_vol.png"), dpi=130)
    plt.close()

    # 6) failure modes: edge by sector (model skill after removing drift)
    fm_s = fm[fm["slice"] == "sector"].set_index("bucket")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(["arima", "xgboost", "xgboost_tuned", "lstm_returns"]):
        ax.bar(np.arange(len(fm_s)) + (i - 1.5) * 0.2, fm_s[f"edge_{m}"] * 100,
               0.2, label=m)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(np.arange(len(fm_s))); ax.set_xticklabels(fm_s.index, rotation=20)
    ax.set_ylabel("directional edge vs bucket up-rate (pp)")
    ax.set_title("Day 6 — Edge over no-skill by sector (positive = genuine signal)")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day06_failure_sector.png"), dpi=130)
    plt.close()

    # 7) shrinkage: how much forecast each config emits
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub2 = fdf.groupby("config")["mean_abs_pred"].mean().reindex(order)
    ax.bar(order, sub2.values * 1e4, color="#c44e52")
    ax.set_ylabel("mean |predicted return| (bps)")
    ax.set_title("Day 6 — Forecast magnitude: what tuning did to the model's conviction")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "day06_pred_shrinkage.png"), dpi=130)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
def write_metrics(study, zero_floor, default_inner, default_abs, imp,
                  lb, bh_sharpe, fm, tuned_params):
    path = os.path.join(RESULTS, "metrics.json")
    blob = {}
    if os.path.exists(path):
        with open(path) as fh:
            blob = json.load(fh)

    best = study.best_trial
    fm_r = fm[fm["slice"] == "regime"].set_index("bucket")
    fm_v = fm[fm["slice"] == "vol_bucket"].set_index("bucket")

    def cfg_row(name):
        r = lb[lb.config == name].iloc[0]
        return {"rmse_vs_zero": round(float(r["mean_rmse_vs_zero"]), 4),
                "dir_acc": round(float(r["mean_dir_acc"]), 4),
                "mean_abs_pred": round(float(r["mean_abs_pred"]), 6),
                "sharpe": round(float(r["mean_sharpe"]), 3),
                "sharpe_vs_bh": round(float(r["sharpe_vs_bh"]), 3),
                "exposure": round(float(r["mean_exposure"]), 3),
                "n_trades": round(float(r["mean_trades"]), 1)}

    blob["day06"] = {
        "day": 6,
        "phase": "4 — Optuna sweep + failure-mode analysis + targeted fixes",
        "tickers": TICKERS,
        "tune_tickers": TUNE_TICKERS,
        "protocol": ("40 TPE trials; objective = mean inner-val RMSE over the "
                     "last 120 rows of each outer fold's train slice "
                     "(5 tickers x 5 folds); outer test blocks untouched"),
        "optuna": {
            "n_trials": N_TRIALS,
            "best_trial": best.number,
            "best_inner_rmse": round(best.value, 6),
            "default_inner_rmse": round(default_inner, 6),
            "zero_floor_inner_rmse": round(zero_floor, 6),
            "best_beats_zero_floor": bool(best.value < zero_floor),
            "mean_abs_pred_best": round(best.user_attrs["mean_abs_pred"], 6),
            "mean_abs_pred_default": round(float(default_abs), 6),
            "best_params": {k: (round(v, 6) if isinstance(v, float) else v)
                            for k, v in best.params.items()},
            "param_importances": {k: round(float(v), 4) for k, v in imp.items()},
        },
        "final_walkforward": {name: cfg_row(name) for name in
                              ["xgb_default", "xgb_tuned",
                               "xgb_tuned_volscale", "xgb_tuned_decay"]},
        "buy_and_hold_sharpe": round(bh_sharpe, 3),
        "failure_modes": {
            "by_regime": {b: {"n_days": int(fm_r.loc[b, "n_days"]),
                              "up_rate": round(float(fm_r.loc[b, "up_rate"]), 4),
                              "acc_always_up": round(float(fm_r.loc[b, "acc_always_up"]), 4),
                              "acc_arima": round(float(fm_r.loc[b, "acc_arima"]), 4),
                              "acc_xgboost_tuned": round(float(fm_r.loc[b, "acc_xgboost_tuned"]), 4)}
                          for b in fm_r.index},
            "by_vol_bucket": {b: {"n_days": int(fm_v.loc[b, "n_days"]),
                                  "up_rate": round(float(fm_v.loc[b, "up_rate"]), 4),
                                  "acc_always_up": round(float(fm_v.loc[b, "acc_always_up"]), 4),
                                  "acc_arima": round(float(fm_v.loc[b, "acc_arima"]), 4),
                                  "acc_xgboost_tuned": round(float(fm_v.loc[b, "acc_xgboost_tuned"]), 4)}
                              for b in fm_v.index},
        },
        "tuned_params_shipped": {k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in tuned_params.items()
                                 if k not in ("n_jobs", "verbosity")},
    }
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)


def main():
    t0 = time.time()
    print("Preparing data (cached yfinance closes, shared feature frames)...")
    data = prepare(TICKERS)

    print("\n== Part 1: Optuna sweep (inner-validation, no peeking) ==")
    study, tuned_params, zero_floor, default_inner, default_abs, imp = run_optuna(data)

    print("\n== Part 3a: final walk-forward — default vs tuned vs fixes ==")
    fdf, tdf, lb, bh_sharpe, tuned_oos = evaluate_configs(data, tuned_params)

    print("\n== Part 2: failure-mode analysis (regime / vol / sector) ==")
    oos, fm = failure_modes(data, tuned_oos)

    # samples: tuned OOS prediction streams for 3 representative tickers
    for tk in ("SPY", "NVDA", "KO"):
        t = tuned_oos[tk]
        pd.DataFrame({"date": t["date"], "actual_ret": t["actual"],
                      "pred_xgb_tuned": t["pred"]}).to_csv(
            os.path.join(SAMPLES, f"day06_{tk}_tuned_predictions.csv"), index=False)
    fm_disp = fm[["slice", "bucket", "n_days", "up_rate"] +
                 [c for c in fm.columns if c.startswith(("acc_", "edge_"))]]
    fm_disp.to_csv(os.path.join(SAMPLES, "day06_failure_mode_table.csv"), index=False)

    plots(study, zero_floor, default_inner, imp, lb, fm, fdf)
    write_metrics(study, zero_floor, default_inner, default_abs, imp,
                  lb, bh_sharpe, fm, tuned_params)

    print("\n" + "=" * 78)
    print("DAY 6 SUMMARY")
    print("=" * 78)
    print(lb[["config", "mean_rmse_vs_zero", "mean_dir_acc", "mean_abs_pred",
              "mean_sharpe", "sharpe_vs_bh"]].to_string(index=False))
    print(f"\nBuy-and-hold Sharpe: {bh_sharpe:.3f}")
    print("\nBy regime (up_rate = no-skill baseline in that bucket):")
    fr = fm[fm["slice"] == "regime"][["bucket", "n_days", "up_rate",
                                      "acc_always_up", "acc_arima",
                                      "acc_xgboost", "acc_xgboost_tuned"]]
    print(fr.to_string(index=False))
    print(f"\nTotal wall-clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
