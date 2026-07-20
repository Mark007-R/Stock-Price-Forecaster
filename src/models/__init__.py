"""
Model registry — the Day-3 bake-off families, packaged for production.

Every model exposes ``predict_fold(prices, fold, ctx) -> np.ndarray`` of
``fold.test_len`` predicted next-day RETURNS, fitting strictly on
``prices[:fold.train_end]`` (the walk-forward contract from
``src.backtest.walkforward``). ``ctx`` comes from :func:`make_context` and
carries shared precomputations (returns, dates, features) so ten models on the
same series don't rebuild them ten times.

The registry order reflects the Day-3 leaderboard: ``arima`` is the champion
among forecasting models (Sharpe 1.34 net of costs), but the honest headline
stays what it was — none of them beats buy-and-hold (1.83), and persistence
(r̂ = 0) is the RMSE floor every candidate must justify itself against.
"""
from __future__ import annotations

import numpy as np

from . import persistence, arima, xgb, lstm  # noqa: E402

# name -> (module, human description)
MODELS = {
    "persistence": (persistence.predict_zero,
                    "random walk r̂=0 — the RMSE floor; takes no position"),
    "momentum":    (persistence.predict_momentum,
                    "r̂_t = r_{t-1} — naive autocorrelation bet"),
    "arima":       (arima.predict_fold,
                    "ARIMA, order by AIC on train; Day-3 champion forecaster"),
    "xgboost":     (xgb.predict_fold,
                    "XGBoost on 18 no-look-ahead features"),
    "xgboost_tuned": (xgb.predict_fold_tuned,
                      "Day-6 Optuna params + time-decay weights — best XGB "
                      "dir-acc (0.531), still below always-up (0.549)"),
    "lstm":        (lstm.predict_fold,
                    "LSTM(32) on standardised returns — predictor.py's arch"),
}

CHAMPION = "arima"  # best Sharpe among actual forecasters, Day 3


def get_model(name: str):
    """Return the model's predict_fold callable or raise with the valid names."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name][0]


def make_context(prices: np.ndarray, dates=None, with_features: bool = False) -> dict:
    """Shared per-series precomputation handed to every ``predict_fold``.

    ``ret[t]`` is the return realised ON day t (nan at t=0). Features are
    optional because only XGBoost needs them and building them costs a pass
    over the full indicator stack.
    """
    prices = np.asarray(prices, dtype=float).flatten()
    ret = np.full(len(prices), np.nan)
    ret[1:] = prices[1:] / prices[:-1] - 1.0
    ctx = {"ret": ret, "dates": dates}
    if with_features:
        from src.features.engineer import build_feature_frame
        ctx["feats"] = build_feature_frame(prices, dates)
    return ctx
