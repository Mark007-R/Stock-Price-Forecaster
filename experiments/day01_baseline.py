"""
Day 1 — Honest baseline + scaler-leakage before/after.

For a fixed universe of 10 tickers over a fixed date range this script:
  1. Reproduces the ORIGINAL leaky pipeline (scaler fit on the FULL series
     before the split) and the LEAKAGE-FIXED pipeline (scaler fit on the train
     slice only) using the same LSTM(32) architecture as predictor.py.
  2. Computes trivial baselines on the SAME test window:
       - persistence  (y_hat_t = y_{t-1})            -> RMSE/MAE/MAPE
       - buy-and-hold (return over the test window)  -> total return
       - always-up    (directional baseline)          -> directional accuracy
  3. Reports directional accuracy for the fixed LSTM.

Everything is measured in PRICE space. The point of Day 1 is the honest number,
so we report the leakage-fixed LSTM as "our" score and quantify how much the
leaky scaler inflated the old numbers.

Public market data only (yfinance). No fabricated returns.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "KO"]
START, END = "2021-01-01", "2025-01-01"     # fixed, reproducible range
TIME_STEP = 30                               # matches predictor.py
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
PLOTS = os.path.join(RESULTS, "plots")
for d in (RESULTS, SAMPLES, PLOTS):
    os.makedirs(d, exist_ok=True)


def set_seeds():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def fetch(ticker):
    df = yf.download(ticker, start=START, end=END, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].dropna()
    return df['Close'].values.astype(float)


def make_windows(scaled, ts):
    X, y = [], []
    for i in range(len(scaled) - ts - 1):
        X.append(scaled[i:i + ts, 0])
        y.append(scaled[i + ts, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_model():
    m = Sequential([
        LSTM(32, return_sequences=False, input_shape=(TIME_STEP, 1)),
        Dropout(0.2),
        Dense(16),
        Dense(1),
    ])
    m.compile(optimizer='adam', loss='mean_squared_error')
    return m


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float).flatten()
    pred = np.asarray(pred, dtype=float).flatten()
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = float(np.mean(np.abs(pred - actual)))
    mape = float(np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100)
    return rmse, mae, mape


def directional_accuracy(prev_actual, actual, pred):
    """Fraction of days the predicted move sign matches the realised move sign."""
    actual_dir = np.sign(actual - prev_actual)
    pred_dir = np.sign(pred - prev_actual)
    mask = actual_dir != 0
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(pred_dir[mask] == actual_dir[mask]))


def run_lstm(close, leaky):
    """Return (test_pred_prices, y_test_prices, prev_prices) for one config."""
    set_seeds()
    close2d = close.reshape(-1, 1)
    raw_split = int(len(close2d) * 0.8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    if leaky:
        scaled = scaler.fit_transform(close2d)              # BUG: sees test min/max
    else:
        scaler.fit(close2d[:raw_split])                     # train-only fit
        scaled = scaler.transform(close2d)

    X, y = make_windows(scaled, TIME_STEP)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = build_model()
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0,
              validation_split=0.1, callbacks=[es])

    pred = scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1)).flatten()
    y_test_price = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # previous actual price for each test target (for directional accuracy)
    # target for test window j (global index split+j) sits at original index
    # (split+j)+TIME_STEP ; its previous price is original index -1.
    test_target_idx = np.array([(split + j) + TIME_STEP for j in range(len(y_test))])
    prev_prices = close[test_target_idx - 1]
    return pred, y_test_price, prev_prices


def main():
    all_rows = []
    per_ticker = {}
    for tk in TICKERS:
        print(f"[{tk}] fetching + training...")
        close = fetch(tk)

        leaky_pred, y_test, prev = run_lstm(close, leaky=True)
        fixed_pred, y_test2, prev2 = run_lstm(close, leaky=False)
        assert np.allclose(y_test, y_test2) and np.allclose(prev, prev2)

        r_leak = metrics(y_test, leaky_pred)
        r_fix = metrics(y_test, fixed_pred)

        # persistence baseline: predict previous day's price
        persist_pred = prev
        r_persist = metrics(y_test, persist_pred)

        # directional accuracy
        da_fixed = directional_accuracy(prev, y_test, fixed_pred)
        da_leak = directional_accuracy(prev, y_test, leaky_pred)
        up_rate = float(np.mean(np.sign(y_test - prev) > 0))  # always-up baseline acc

        # buy-and-hold over the test window
        bh_return = float(y_test[-1] / prev[0] - 1.0)

        row = {
            "ticker": tk,
            "test_days": int(len(y_test)),
            "rmse_leaky": round(r_leak[0], 4),
            "rmse_fixed": round(r_fix[0], 4),
            "rmse_inflation_pct": round((r_fix[0] - r_leak[0]) / r_leak[0] * 100, 1),
            "rmse_persistence": round(r_persist[0], 4),
            "mae_fixed": round(r_fix[1], 4),
            "mape_fixed": round(r_fix[2], 4),
            "mape_persistence": round(r_persist[2], 4),
            "dir_acc_fixed": round(da_fixed, 4),
            "dir_acc_leaky": round(da_leak, 4),
            "dir_acc_alwaysup": round(up_rate, 4),
            "buy_hold_return_pct": round(bh_return * 100, 2),
            "lstm_beats_persistence_rmse": bool(r_fix[0] < r_persist[0]),
        }
        all_rows.append(row)
        per_ticker[tk] = row
        print(f"    RMSE leaky={r_leak[0]:.3f}  fixed={r_fix[0]:.3f}  "
              f"persistence={r_persist[0]:.3f}  dir_acc={da_fixed:.3f}")

        # save a sample of predictions
        n = min(len(y_test), 60)
        pd.DataFrame({
            "prev_actual": np.round(prev[-n:], 2),
            "actual": np.round(y_test[-n:], 2),
            "lstm_fixed_pred": np.round(fixed_pred[-n:], 2),
            "lstm_leaky_pred": np.round(leaky_pred[-n:], 2),
            "persistence_pred": np.round(persist_pred[-n:], 2),
        }).to_csv(os.path.join(SAMPLES, f"{tk}_test_predictions.csv"), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS, "phase1_leakage_comparison.csv"), index=False)

    # aggregate
    agg = {
        "date_range": {"start": START, "end": END},
        "tickers": TICKERS,
        "time_step": TIME_STEP,
        "lstm_arch": "LSTM(32)->Dropout(0.2)->Dense(16)->Dense(1)  [matches predictor.py]",
        "note": ("Leakage-fixed metrics are the HONEST baseline. Leaky metrics are "
                 "shown only to quantify the inflation and are NEVER reported as 'our score'."),
        "mean_rmse_leaky": round(float(df.rmse_leaky.mean()), 4),
        "mean_rmse_fixed_HONEST": round(float(df.rmse_fixed.mean()), 4),
        "mean_rmse_persistence_baseline": round(float(df.rmse_persistence.mean()), 4),
        "mean_rmse_inflation_pct": round(float(df.rmse_inflation_pct.mean()), 1),
        "mean_mape_fixed_HONEST_pct": round(float(df.mape_fixed.mean()), 4),
        "mean_mape_persistence_pct": round(float(df.mape_persistence.mean()), 4),
        "mean_dir_acc_fixed": round(float(df.dir_acc_fixed.mean()), 4),
        "mean_dir_acc_alwaysup": round(float(df.dir_acc_alwaysup.mean()), 4),
        "mean_buy_hold_return_pct": round(float(df.buy_hold_return_pct.mean()), 2),
        "n_tickers_lstm_beats_persistence_rmse": int(df.lstm_beats_persistence_rmse.sum()),
        "per_ticker": per_ticker,
    }
    with open(os.path.join(RESULTS, "baseline_metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # ── plots ──────────────────────────────────────────────────────────────
    # 1. leakage before/after RMSE
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.4
    ax.bar(x - w/2, df.rmse_leaky, w, label='Leaky scaler (inflated / optimistic)', color='#e07a5f')
    ax.bar(x + w/2, df.rmse_fixed, w, label='Fixed scaler (honest)', color='#3d5a80')
    ax.set_xticks(x); ax.set_xticklabels(df.ticker)
    ax.set_ylabel('Test RMSE (price $)')
    ax.set_title('Scaler leakage: RMSE before vs after the fix (fit on train slice only)')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "leakage_rmse_before_after.png"), dpi=120)
    plt.close(fig)

    # 2. LSTM (fixed) vs persistence RMSE
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, df.rmse_fixed, w, label='LSTM (leakage-fixed)', color='#3d5a80')
    ax.bar(x + w/2, df.rmse_persistence, w, label='Persistence (y_t = y_{t-1})', color='#81b29a')
    ax.set_xticks(x); ax.set_xticklabels(df.ticker)
    ax.set_ylabel('Test RMSE (price $)')
    ax.set_title('Honest LSTM vs trivial persistence baseline')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "lstm_vs_persistence_rmse.png"), dpi=120)
    plt.close(fig)

    # 3. directional accuracy
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, df.dir_acc_fixed, w, label='LSTM directional acc', color='#3d5a80')
    ax.bar(x + w/2, df.dir_acc_alwaysup, w, label='Always-up baseline', color='#f2cc8f')
    ax.axhline(0.5, ls='--', color='grey', label='coin flip (0.50)')
    ax.set_xticks(x); ax.set_xticklabels(df.ticker)
    ax.set_ylabel('Directional accuracy'); ax.set_ylim(0, 1)
    ax.set_title('Next-day directional accuracy (leakage-fixed LSTM)')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "directional_accuracy.png"), dpi=120)
    plt.close(fig)

    print("\n=== AGGREGATE ===")
    print(json.dumps({k: v for k, v in agg.items() if k not in ('per_ticker',)}, indent=2))
    print("\nSaved: results/baseline_metrics.json, results/phase1_leakage_comparison.csv, 3 plots, samples/")


if __name__ == "__main__":
    main()
