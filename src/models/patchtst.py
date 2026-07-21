"""
PatchTST-style transformer on standardised returns — walk-forward form.

Why this exists
---------------
The repo's name promises modern time-series ML; through Day 6 the deepest
sequence model was the shipped LSTM(32), which converged to predicting ~0.
PatchTST (Nie et al., 2023) is the architecture that made transformers
competitive on long-horizon TS benchmarks, via two ideas: (1) PATCH the
series into short sub-windows and embed each patch as one token, so attention
runs over ~L/P tokens instead of L points; (2) channel independence — each
series is modelled separately, which for this univariate next-day-return task
is free. This module is a faithful small-scale Keras implementation, not a
port: patch embedding + learned positional embedding + pre-norm transformer
encoder blocks + linear head.

Honesty protocol (identical to src.models.lstm):
* StandardScaler fit on TRAIN returns only — the Day-1 leakage fix.
* Fits only on ``prices[:fold.train_end]``; test windows may use past
  actuals (correct online behaviour), never future ones.
* Seeded; TF imports live inside functions so the registry loads without TF.

Sized for the sprint's CPU budget: L=64, patch 8, d_model=32, 2 blocks,
~15k parameters — deliberately in the same capacity class as the LSTM(32)
(~5k) so a win or loss is about architecture, not parameter count.
"""
from __future__ import annotations

import os

import numpy as np

CONTEXT_LEN = 64          # lookback window L
PATCH_LEN = 8             # points per patch (non-overlapping: stride = patch)
D_MODEL = 32
N_HEADS = 4
N_BLOCKS = 2
FF_DIM = 64
DROPOUT = 0.1
EPOCHS = 15
BATCH = 32
SEED = 42


def _build():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    n_patches = CONTEXT_LEN // PATCH_LEN

    inp = layers.Input(shape=(CONTEXT_LEN,))
    # Patchify: (batch, L) -> (batch, n_patches, patch_len); each patch
    # becomes one attention token via a shared linear embedding.
    x = layers.Reshape((n_patches, PATCH_LEN))(inp)
    x = layers.Dense(D_MODEL)(x)

    # Learned positional embedding over patch positions.
    pos = tf.range(n_patches)
    pos_emb = layers.Embedding(n_patches, D_MODEL)(pos)
    x = x + pos_emb

    for _ in range(N_BLOCKS):
        # Pre-norm block, as in the paper.
        h = layers.LayerNormalization()(x)
        h = layers.MultiHeadAttention(num_heads=N_HEADS,
                                      key_dim=D_MODEL // N_HEADS,
                                      dropout=DROPOUT)(h, h)
        x = x + h
        h = layers.LayerNormalization()(x)
        h = layers.Dense(FF_DIM, activation="gelu")(h)
        h = layers.Dropout(DROPOUT)(h)
        h = layers.Dense(D_MODEL)(h)
        x = x + h

    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)

    m = Model(inp, out)
    m.compile(optimizer="adam", loss="mean_squared_error")
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

    Xtr, ytr = _windows(scaled, CONTEXT_LEN, CONTEXT_LEN + 1, fold.train_end)
    model = _build()
    model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH, verbose=0,
              validation_split=0.1,
              callbacks=[EarlyStopping('val_loss', patience=3,
                                       restore_best_weights=True)])

    Xte, _ = _windows(scaled, CONTEXT_LEN, fold.test_start, fold.test_end)
    pred_scaled = model.predict(Xte, verbose=0).flatten()
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
