"""Walk-forward backtesting harness (no look-ahead) + cost-aware trading sim."""
from .walkforward import (
    Fold,
    expanding_window_folds,
    rolling_window_folds,
    assert_no_peeking,
    returns_from_prices,
    rmse,
    directional_accuracy,
    walk_forward_predict,
)
from .trading import (
    BacktestResult,
    backtest_long_flat,
    backtest_buy_and_hold,
    TRADING_DAYS,
)

__all__ = [
    "Fold",
    "expanding_window_folds",
    "rolling_window_folds",
    "assert_no_peeking",
    "returns_from_prices",
    "rmse",
    "directional_accuracy",
    "walk_forward_predict",
    "BacktestResult",
    "backtest_long_flat",
    "backtest_buy_and_hold",
    "TRADING_DAYS",
]
