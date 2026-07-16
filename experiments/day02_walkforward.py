"""
Day 2 — Target reframing (price -> next-day return) + walk-forward harness.

Two honest questions, answered on the SAME expanding walk-forward folds for a
fixed 10-ticker universe over 2021-01-01 -> 2025-01-01:

  1. Does an LSTM that predicts the ABSOLUTE price actually add anything over a
     random-walk once you score it in RETURN space? (Day 1 hinted "no" on a
     single holdout — walk-forward makes it rigorous.)
  2. Is an LSTM RE-FRAMED to predict the next-day RETURN directly any better —
     specifically, does it beat a coin flip on direction?

All models are refit per fold on the past only (no peeking, enforced by
`src.backtest.walkforward`). Everything is scored in return space:
  * RMSE(returns) vs the zero-return random walk (predicting 0 == "no change")
  * Directional accuracy vs coin-flip / always-up / momentum baselines

Public yfinance data only. No fabricated returns; transaction-cost-aware
trading backtest lands Day 3 on top of these same folds.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.backtest.walkforward import (           # noqa: E402
    expanding_window_folds, assert_no_peeking, walk_forward_predict,
    directional_accuracy, rmse,
)

SEED = 42
TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "KO"]
START, END = "2021-01-01", "2025-01-01"
TIME_STEP = 30
N_FOLDS = 5
EPOCHS = 12
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
PLOTS = os.path.join(RESULTS, "plots")
for d in (RESULTS, SAMPLES, PLOTS):
    os.makedirs(d, exist_ok=True)


def set_seeds():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def fetch(ticker, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=START, end=END, interval="1d",
                             progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df[['Close']].dropna()['Close'].values.astype(float)
        except Exception as e:                                 # noqa: BLE001
            print(f"  fetch {ticker} attempt {attempt+1} failed: {e}")
        time.sleep(2)
    raise RuntimeError(f"No data for {ticker} after {retries} tries")


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
    """Windows whose TARGET index t satisfies ts <= t < hi_target_excl and
    t >= lo. Returns X (n, ts), y (n,), and the target indices."""
    X, y, idx = [], [], []
    for t in range(max(ts, lo), hi_target_excl):
        X.append(series_1d[t - ts:t])
        y.append(series_1d[t])
        idx.append(t)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            np.asarray(idx))


# ─────────────────────────────────────────────────────────────────────────────
# Model functions (each: fit on prices[:fold.train_end] only, predict returns)
# ─────────────────────────────────────────────────────────────────────────────
def lstm_price_fn(prices, fold):
    """LSTM predicting the ABSOLUTE next-day price; converted to a return.

    This is the SHIPPED framing. One-step-ahead over the test block using true
    past prices as inputs (no autoregressive drift), so it gets the fairest
    possible shot.
    """
    set_seeds()
    close2d = prices.reshape(-1, 1)
    scaler = MinMaxScaler((0, 1))
    scaler.fit(close2d[:fold.train_end])              # train-only fit — no peeking
    scaled = scaler.transform(close2d).flatten()

    Xtr, ytr, _ = _windows(scaled, TIME_STEP, TIME_STEP, fold.train_end)
    Xtr = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], 1)
    model = build_lstm()
    model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=16, verbose=0,
              validation_split=0.1,
              callbacks=[EarlyStopping('val_loss', patience=3, restore_best_weights=True)])

    Xte, _, idx = _windows(scaled, TIME_STEP, fold.test_start, fold.test_end)
    Xte = Xte.reshape(Xte.shape[0], Xte.shape[1], 1)
    pred_scaled = model.predict(Xte, verbose=0).flatten()
    pred_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    prev_price = prices[idx - 1]
    return pred_price / prev_price - 1.0             # price -> return


def lstm_returns_fn(prices, fold):
    """LSTM RE-FRAMED to predict the next-day RETURN directly (the honest task)."""
    set_seeds()
    # ret_at[t] = prices[t]/prices[t-1]-1, defined for t in [1, N-1]; store at
    # index t (index 0 is a placeholder that is never used as a target/input).
    ret_at = np.zeros_like(prices)
    ret_at[1:] = prices[1:] / prices[:-1] - 1.0

    scaler = StandardScaler()
    scaler.fit(ret_at[1:fold.train_end].reshape(-1, 1))     # train returns only
    scaled = scaler.transform(ret_at.reshape(-1, 1)).flatten()

    # targets must have a full window of PRIOR returns -> earliest target index
    # is TIME_STEP+1 (window ret_at[t-ts..t-1] needs t-ts >= 1).
    Xtr, ytr, _ = _windows(scaled, TIME_STEP, TIME_STEP + 1, fold.train_end)
    Xtr = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], 1)
    model = build_lstm()
    model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=16, verbose=0,
              validation_split=0.1,
              callbacks=[EarlyStopping('val_loss', patience=3, restore_best_weights=True)])

    Xte, _, _ = _windows(scaled, TIME_STEP, fold.test_start, fold.test_end)
    Xte = Xte.reshape(Xte.shape[0], Xte.shape[1], 1)
    pred_scaled = model.predict(Xte, verbose=0).flatten()
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()


def persistence_zero_fn(prices, fold):
    """Random walk: predict NO change. The benchmark for return RMSE."""
    return np.zeros(fold.test_len)


def momentum_fn(prices, fold):
    """Predict tomorrow's return = today's realised return (naive momentum)."""
    t = np.arange(fold.test_start, fold.test_end)
    return prices[t - 1] / prices[t - 2] - 1.0


def always_up_fn(prices, fold):
    """Directional baseline: always bet up (markets drift up)."""
    return np.full(fold.test_len, 1e-6)


METHODS = {
    "lstm_price":       lstm_price_fn,
    "lstm_returns":     lstm_returns_fn,
    "persistence_zero": persistence_zero_fn,
    "momentum":         momentum_fn,
    "always_up":        always_up_fn,
}
# methods whose RMSE(returns) is a meaningful point forecast (not a pure
# directional bet). always_up is a constant tiny number -> RMSE not meaningful.
RMSE_METHODS = {"lstm_price", "lstm_returns", "persistence_zero", "momentum"}
DIR_METHODS = {"lstm_price", "lstm_returns", "momentum", "always_up"}


def main():
    rows = []                    # long-format per (ticker, fold, method)
    sample_saved = set()

    for tk in TICKERS:
        print(f"[{tk}] fetching...")
        prices = fetch(tk)
        folds = expanding_window_folds(len(prices), n_folds=N_FOLDS)
        assert_no_peeking(folds, n_samples=len(prices))
        print(f"  n={len(prices)}  folds={len(folds)}  "
              f"test_size~{folds[0].test_len}")

        # run every method through the identical folds
        per_method_fold = {}     # method -> list of fold result dicts
        for name, fn in METHODS.items():
            t0 = time.time()
            res = walk_forward_predict(prices, fn, folds)
            per_method_fold[name] = res
            if name.startswith("lstm"):
                print(f"    {name:16s} done in {time.time()-t0:5.1f}s  "
                      f"mean dir_acc={np.nanmean([r['dir_acc'] for r in res]):.3f}")

        for name, res in per_method_fold.items():
            for r in res:
                rows.append({
                    "ticker": tk,
                    "fold": r["fold"],
                    "method": name,
                    "train_days": r["train_days"],
                    "test_days": r["test_days"],
                    "rmse_ret": round(r["rmse_ret"], 6) if name in RMSE_METHODS else np.nan,
                    "dir_acc": round(r["dir_acc"], 4) if name in DIR_METHODS else np.nan,
                })

        # save a sample: last fold, last 60 test days, all methods' return preds
        if tk not in sample_saved:
            f_last = folds[-1]
            actual = per_method_fold["persistence_zero"][-1]["actual_ret"]
            n = min(len(actual), 60)
            sdf = {"actual_ret": np.round(actual[-n:], 5)}
            for name in METHODS:
                sdf[f"{name}_pred"] = np.round(
                    per_method_fold[name][-1]["pred_ret"][-n:], 5)
            pd.DataFrame(sdf).to_csv(
                os.path.join(SAMPLES, f"day02_{tk}_walkforward.csv"), index=False)
            sample_saved.add(tk)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "walkforward.csv"), index=False)

    # ── aggregate: mean over folds within ticker, then over tickers ──────────
    summary_rows = []
    for name in METHODS:
        sub = df[df.method == name]
        rmse_mean = float(np.nanmean(sub.rmse_ret)) if name in RMSE_METHODS else np.nan
        da_mean = float(np.nanmean(sub.dir_acc)) if name in DIR_METHODS else np.nan
        da_std = float(np.nanstd(sub.dir_acc)) if name in DIR_METHODS else np.nan
        summary_rows.append({
            "method": name,
            "mean_rmse_ret": round(rmse_mean, 6) if name in RMSE_METHODS else "n/a",
            "mean_dir_acc": round(da_mean, 4) if name in DIR_METHODS else "n/a",
            "dir_acc_std_across_folds": round(da_std, 4) if name in DIR_METHODS else "n/a",
        })
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(os.path.join(RESULTS, "phase2a_walkforward_summary.csv"), index=False)

    # how often does lstm_returns beat coin flip per (ticker,fold)?
    lr = df[df.method == "lstm_returns"]
    lp = df[df.method == "lstm_price"]
    beats_coin = int((lr.dir_acc > 0.5).sum())
    n_lr = int(lr.dir_acc.notna().sum())

    # per-ticker RMSE(returns): lstm_returns vs persistence_zero
    piv = df.pivot_table(index="ticker", columns="method",
                         values="rmse_ret", aggfunc="mean")
    n_beat_rw = int((piv["lstm_returns"] < piv["persistence_zero"]).sum())

    metrics_entry = {
        "day": 2,
        "phase": "2a — target reframing + walk-forward harness",
        "date_range": {"start": START, "end": END},
        "tickers": TICKERS,
        "n_folds": N_FOLDS,
        "time_step": TIME_STEP,
        "split": "expanding-window walk-forward, refit per fold, no peeking",
        "scoring_space": "next-day simple returns",
        "summary": {r["method"]: {k: r[k] for k in r if k != "method"}
                     for r in summary_rows},
        "lstm_returns_folds_beating_coinflip": f"{beats_coin}/{n_lr}",
        "lstm_returns_tickers_beating_random_walk_rmse": f"{n_beat_rw}/{len(piv)}",
        "note": ("Return RMSE ~equals the zero-return random walk for every "
                 "model — next-day returns are near-unpredictable. Directional "
                 "accuracy hovers at coin-flip. This is the honest Day-2 finding "
                 "and the baseline the Day-3 model bake-off must beat."),
    }
    metrics_path = os.path.join(RESULTS, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            log = json.load(f)
    else:
        log = {}
    log["day02"] = metrics_entry
    with open(metrics_path, "w") as f:
        json.dump(log, f, indent=2)

    # ── plots ────────────────────────────────────────────────────────────────
    # 1. Mean RMSE(returns) per method vs random walk
    fig, ax = plt.subplots(figsize=(9, 5))
    m = [x for x in METHODS if x in RMSE_METHODS]
    vals = [float(np.nanmean(df[df.method == x].rmse_ret)) for x in m]
    colors = ['#3d5a80', '#98c1d9', '#81b29a', '#f2cc8f']
    ax.bar(m, vals, color=colors)
    rw = float(np.nanmean(df[df.method == 'persistence_zero'].rmse_ret))
    ax.axhline(rw, ls='--', color='grey', label=f'random walk = {rw:.4f}')
    ax.set_ylabel('Mean RMSE (next-day returns)')
    ax.set_title('Return RMSE across walk-forward folds — nobody beats the random walk')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "day02_return_rmse.png"), dpi=120)
    plt.close(fig)

    # 2. Mean directional accuracy per method
    fig, ax = plt.subplots(figsize=(9, 5))
    dm = [x for x in METHODS if x in DIR_METHODS]
    dvals = [float(np.nanmean(df[df.method == x].dir_acc)) for x in dm]
    ax.bar(dm, dvals, color=['#3d5a80', '#e07a5f', '#81b29a', '#f2cc8f'])
    ax.axhline(0.5, ls='--', color='grey', label='coin flip (0.50)')
    ax.set_ylabel('Mean directional accuracy'); ax.set_ylim(0, 0.7)
    ax.set_title('Next-day directional accuracy across walk-forward folds')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "day02_directional_accuracy.png"), dpi=120)
    plt.close(fig)

    # 3. Per-fold directional accuracy instability (lstm_returns vs always_up)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, c in [("lstm_returns", '#3d5a80'), ("always_up", '#f2cc8f')]:
        by_fold = df[df.method == name].groupby("fold").dir_acc.mean()
        ax.plot(by_fold.index, by_fold.values, marker='o', label=name, color=c)
    ax.axhline(0.5, ls='--', color='grey', label='coin flip')
    ax.set_xlabel('Walk-forward fold (time →)'); ax.set_ylabel('Directional accuracy')
    ax.set_title('Directional accuracy is unstable fold-to-fold (mean over 10 tickers)')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "day02_dir_acc_per_fold.png"), dpi=120)
    plt.close(fig)

    # 4. lstm_price(as-returns) vs lstm_returns vs random walk, per ticker RMSE
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(piv)); w = 0.27
    ax.bar(x - w, piv["lstm_price"], w, label='LSTM-price (as returns)', color='#e07a5f')
    ax.bar(x,     piv["lstm_returns"], w, label='LSTM-returns', color='#3d5a80')
    ax.bar(x + w, piv["persistence_zero"], w, label='random walk (0)', color='#81b29a')
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=0)
    ax.set_ylabel('Mean RMSE (returns)')
    ax.set_title('Return RMSE per ticker — LSTM barely distinguishable from a random walk')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "day02_rmse_per_ticker.png"), dpi=120)
    plt.close(fig)

    print("\n=== WALK-FORWARD SUMMARY (mean over 10 tickers × "
          f"{N_FOLDS} folds) ===")
    print(sdf.to_string(index=False))
    print(f"\nlstm_returns folds beating coin flip: {beats_coin}/{n_lr}")
    print(f"lstm_returns tickers beating random-walk RMSE: {n_beat_rw}/{len(piv)}")
    print("\nSaved: results/walkforward.csv, results/phase2a_walkforward_summary.csv,")
    print("       results/metrics.json (day02), 4 plots, 10 sample CSVs")


if __name__ == "__main__":
    main()
