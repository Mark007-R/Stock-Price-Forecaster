"""
ARIMA on next-day returns — the Day-3 champion among forecasting models.

Champion with an asterisk: it "won" the bake-off (Sharpe 1.34 net of costs,
RMSE 0.016377) yet still loses to buy-and-hold (1.83) and edges the random
walk by a rounding error. It is the best *forecaster* we have, which is the
point the whole sprint keeps making: the best forecaster is barely a forecast.

Fitting protocol (identical to the bake-off, so numbers stay comparable):
* Order selected by AIC on the TRAIN returns only, from a small grid.
* Through the test block the fitted model is *conditioned* on realised returns
  as they arrive (``append(..., refit=False)``) but never re-estimated. That is
  correct online behaviour, not peeking — at each step it has only the past.
"""
from __future__ import annotations

import numpy as np

ORDER_GRID = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 2)]


def select_order(train_ret: np.ndarray) -> tuple[int, int, int]:
    """AIC-best order on the training returns; falls back to AR(1)."""
    from statsmodels.tsa.arima.model import ARIMA

    best_order, best_aic = (1, 0, 0), np.inf
    for order in ORDER_GRID:
        try:
            aic = ARIMA(train_ret, order=order).fit().aic
            if aic < best_aic:
                best_aic, best_order = aic, order
        except Exception:  # noqa: BLE001 — non-invertible candidates just lose
            continue
    return best_order


def predict_fold(prices: np.ndarray, fold, ctx: dict) -> np.ndarray:
    """Rolling one-step-ahead return forecasts for the fold's test block."""
    from statsmodels.tsa.arima.model import ARIMA

    r = ctx["ret"]
    train_ret = r[1:fold.train_end]

    order = select_order(train_ret)
    res = ARIMA(train_ret, order=order).fit()
    ctx.setdefault("arima_orders", []).append(order)

    preds = []
    for k in range(fold.test_len):
        preds.append(float(res.forecast(steps=1)[0]))
        res = res.append([r[fold.test_start + k]], refit=False)
    return np.asarray(preds)


def forecast_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Fit on ALL given history and forecast ``horizon`` future daily returns.

    This is the serving-time entry point (no fold structure): the API's
    /predict endpoint uses it for the point forecast, with uncertainty coming
    from `src.models.intervals` rather than ARIMA's own Gaussian assumptions.
    """
    from statsmodels.tsa.arima.model import ARIMA

    prices = np.asarray(prices, dtype=float).flatten()
    ret = prices[1:] / prices[:-1] - 1.0
    order = select_order(ret)
    res = ARIMA(ret, order=order).fit()
    return np.asarray(res.forecast(steps=horizon), dtype=float)
