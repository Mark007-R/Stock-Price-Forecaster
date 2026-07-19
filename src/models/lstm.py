"""
LSTM on standardised returns — predictor.py's architecture, walk-forward form.

Same LSTM(32)→Dropout(0.2)→Dense(16)→Dense(1) as the shipped Flask app, so the
bake-off compares the *actual* production model, not a strawman. Day 2's
verdict stands: on returns it converges to predicting ≈0 (RMSE ties the random
walk at 0.0164) because zero is the honest answer for next-day returns.

The scaler is fit on TRAIN returns only — the Day-1 leakage fix carried into
every walk-forward context. TensorFlow imports live inside the functions so
the registry (and the FastAPI service, which defaults to ARIMA) can load
without pulling in TF.
"""
from __future__ import annotations

import os

import numpy as np

TIME_STEP = 30
EPOCHS = 12
SEED = 42


def _build(time_step: int = TIME_STEP):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    m = Sequential([
        LSTM(32, return_sequences=False, input_shape=(time_step, 1)),
        Dropout(0.2),
        Dense(16),
        Dense(1),
    ])
    m.compile(optimizer='adam', loss='mean_squared_error')
    return m


def _windows(series_1d, ts, lo, hi_target_excl):
    X, y = [], []
    for t in range(max(ts, lo), hi_target_excl):
        X.append(series_1d[t - ts:t])
        y.append(series_1d[t])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def predict_fold(prices: np.ndarray, fold, ctx: dict) -> np.ndarray:
    """Train on pre-fold returns (train-only scaler), predict the test block."""
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.callbacks import EarlyStopping

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    prices = np.asarray(prices, dtype=float).flatten()
    ret_at = np.zeros_like(prices)
    ret_at[1:] = prices[1:] / prices[:-1] - 1.0

    scaler = StandardScaler()
    scaler.fit(ret_at[1:fold.train_end].reshape(-1, 1))  # train returns only
    scaled = scaler.transform(ret_at.reshape(-1, 1)).flatten()

    Xtr, ytr = _windows(scaled, TIME_STEP, TIME_STEP + 1, fold.train_end)
    Xtr = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], 1)
    model = _build()
    model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=16, verbose=0,
              validation_split=0.1,
              callbacks=[EarlyStopping('val_loss', patience=3,
                                       restore_best_weights=True)])

    Xte, _ = _windows(scaled, TIME_STEP, fold.test_start, fold.test_end)
    Xte = Xte.reshape(Xte.shape[0], Xte.shape[1], 1)
    pred_scaled = model.predict(Xte, verbose=0).flatten()
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
