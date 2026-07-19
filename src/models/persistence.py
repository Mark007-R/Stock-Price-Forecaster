"""
Persistence baselines — the models every other model must beat to exist.

``predict_zero`` is the random walk: tomorrow's expected return is zero. On
daily equity returns this is the RMSE floor no bake-off model has beaten
(Day 2–4). It predicts no edge, so in the trading backtest it takes no
position — which is the honest translation of "I don't know".

``predict_momentum`` bets that today's return repeats tomorrow. Day 3 measured
it 41% WORSE than the random walk on RMSE — daily returns are not positively
autocorrelated — and it is kept as the cautionary baseline, not a candidate.
"""
from __future__ import annotations

import numpy as np


def predict_zero(prices: np.ndarray, fold, ctx: dict | None = None) -> np.ndarray:
    """Random walk: forecast no change for every test day."""
    return np.zeros(fold.test_len)


def predict_momentum(prices: np.ndarray, fold, ctx: dict | None = None) -> np.ndarray:
    """Tomorrow's return = today's realised return (uses only past prices)."""
    prices = np.asarray(prices, dtype=float).flatten()
    t = np.arange(fold.test_start, fold.test_end)
    return prices[t - 1] / prices[t - 2] - 1.0
