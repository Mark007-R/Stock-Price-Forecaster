"""
Walk-forward backtesting harness — the honest evaluation core for StockAI.

Why this exists
---------------
Day 1 measured every model on a SINGLE 80/20 holdout. That is fragile: the
"test window" is one arbitrary slice of history, and a model can look good or
bad purely by which regime it landed in. A quant demands **walk-forward**
evaluation: repeatedly refit the model on the past and score it on the
immediately-following out-of-sample block, sliding forward through time. The
model is NEVER shown a day it will later be scored on — no peeking.

This module provides the split logic + metric helpers ONLY. It is deliberately
model-agnostic so Day 3's bake-off (persistence / ARIMA / Prophet / XGBoost /
LSTM) reuses the exact same folds, and Day 10's regression test
(`test_walkforward_no_peeking.py`) can assert the boundaries structurally.

No-peeking guarantee (structural)
---------------------------------
* Folds are contiguous and non-overlapping: every fold's ``test_start`` equals
  its ``train_end`` (fit strictly on the past, score strictly on the future).
* ``walk_forward_predict`` hands each model-fn ``prices[:train_end]`` for
  FITTING and, separately, the fold's test span for PREDICTION. Inputs formed
  during the test block may use past actuals (that is correct online
  behaviour), but the fitting slice can never contain a test day.
* ``assert_no_peeking`` re-checks the invariants at runtime.

Everything downstream is scored in **return space** (next-day simple returns),
because Day 1 showed absolute-price targets are dominated by the previous
price and make any model look deceptively accurate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fold definition + generators
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Fold:
    """One walk-forward fold. All indices are into the price array.

    ``train`` is ``[train_start, train_end)`` and ``test`` is
    ``[test_start, test_end)`` with ``test_start == train_end`` always.
    """
    fold: int
    train_start: int
    train_end: int        # exclusive — first index the model may NOT fit on
    test_start: int       # inclusive — always equals train_end
    test_end: int         # exclusive

    @property
    def train_len(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_len(self) -> int:
        return self.test_end - self.test_start


def expanding_window_folds(
    n_samples: int,
    n_folds: int = 5,
    test_size: int | None = None,
    min_train: int | None = None,
) -> List[Fold]:
    """Expanding-window walk-forward folds.

    Train window GROWS each fold (start pinned at 0); the test block slides
    forward by ``test_size``. This is the standard choice for financial
    series — you always train on *all* history available up to that point.

    ``test_size`` defaults to an even split of the tail across ``n_folds``.
    ``min_train`` is the size of the first training window (defaults to the
    remainder before the first test block).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    if test_size is None:
        # reserve ~half of history for testing, split evenly across folds
        test_size = max(1, (n_samples // 2) // n_folds)
    total_test = test_size * n_folds
    if min_train is None:
        min_train = n_samples - total_test
    if min_train < 1:
        raise ValueError(
            f"Not enough samples: n={n_samples}, need >{total_test} for "
            f"{n_folds} folds of test_size={test_size}"
        )

    folds: List[Fold] = []
    for k in range(n_folds):
        train_end = min_train + k * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        if test_start >= n_samples or test_end <= test_start:
            break
        folds.append(Fold(k, 0, train_end, test_start, test_end))
    return folds


def rolling_window_folds(
    n_samples: int,
    n_folds: int = 5,
    test_size: int | None = None,
    train_size: int | None = None,
) -> List[Fold]:
    """Rolling (fixed-width) walk-forward folds.

    Train window is a FIXED width that slides forward — useful for probing
    whether the model only works in recent regimes. Test blocks are again
    contiguous and non-overlapping.
    """
    if test_size is None:
        test_size = max(1, (n_samples // 2) // n_folds)
    if train_size is None:
        train_size = n_samples - test_size * n_folds
    if train_size < 1:
        raise ValueError("train_size too small for the requested folds")

    folds: List[Fold] = []
    for k in range(n_folds):
        train_start = k * test_size
        train_end = train_start + train_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        if test_end <= test_start:
            break
        folds.append(Fold(k, train_start, train_end, test_start, test_end))
    return folds


def assert_no_peeking(folds: List[Fold], n_samples: int | None = None) -> None:
    """Raise if any fold violates the walk-forward invariants."""
    for f in folds:
        if f.test_start != f.train_end:
            raise AssertionError(
                f"Fold {f.fold}: test_start ({f.test_start}) != train_end "
                f"({f.train_end}) — the model would be scored on a day it "
                f"trained on."
            )
        if f.train_len < 1 or f.test_len < 1:
            raise AssertionError(f"Fold {f.fold}: empty train or test span.")
        if n_samples is not None and f.test_end > n_samples:
            raise AssertionError(
                f"Fold {f.fold}: test_end {f.test_end} exceeds n={n_samples}."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Return-space metrics
# ─────────────────────────────────────────────────────────────────────────────
def returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Simple next-step returns r_t = p_t / p_{t-1} - 1 (length n-1)."""
    p = np.asarray(prices, dtype=float).flatten()
    return p[1:] / p[:-1] - 1.0


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float).flatten()
    p = np.asarray(pred, dtype=float).flatten()
    return float(np.sqrt(np.mean((p - a) ** 2)))


def directional_accuracy(pred_ret: np.ndarray, actual_ret: np.ndarray) -> float:
    """Fraction of days the predicted return SIGN matches the realised sign.

    Days with a flat realised move (sign 0) are excluded — you cannot be
    right or wrong about a move that did not happen. Predictions that are
    themselves exactly 0 count as a miss on any non-flat day (a flat forecast
    takes no directional position, so it earns no directional credit).
    """
    pr = np.sign(np.asarray(pred_ret, dtype=float).flatten())
    ar = np.sign(np.asarray(actual_ret, dtype=float).flatten())
    mask = ar != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(pr[mask] == ar[mask]))


# ─────────────────────────────────────────────────────────────────────────────
# Generic orchestrator
# ─────────────────────────────────────────────────────────────────────────────
# A model-fn takes the FULL price array + the fold, fits ONLY on
# prices[:fold.train_end], and returns a length-``fold.test_len`` array of
# predicted next-day RETURNS for the test block.
ModelFn = Callable[[np.ndarray, Fold], np.ndarray]


def walk_forward_predict(
    prices: np.ndarray,
    model_fn: ModelFn,
    folds: List[Fold],
) -> List[dict]:
    """Run ``model_fn`` across ``folds`` and score each in return space.

    Returns one dict per fold: fold id, train/test sizes, RMSE(returns),
    directional accuracy, plus the raw predicted/actual return arrays so the
    caller can aggregate or build a trading backtest on top (Day 3).
    """
    prices = np.asarray(prices, dtype=float).flatten()
    assert_no_peeking(folds, n_samples=len(prices))

    out: List[dict] = []
    for f in folds:
        pred_ret = np.asarray(model_fn(prices, f), dtype=float).flatten()
        # realised next-day returns over the test span: for test index t the
        # target return is p_t / p_{t-1} - 1 (p_{t-1} may be the last train day)
        test_prices = prices[f.test_start:f.test_end]
        prev_prices = prices[f.test_start - 1:f.test_end - 1]
        actual_ret = test_prices / prev_prices - 1.0
        if len(pred_ret) != len(actual_ret):
            raise ValueError(
                f"Fold {f.fold}: model returned {len(pred_ret)} preds for "
                f"{len(actual_ret)} test days."
            )
        out.append({
            "fold": f.fold,
            "train_days": f.train_len,
            "test_days": f.test_len,
            "rmse_ret": rmse(actual_ret, pred_ret),
            "dir_acc": directional_accuracy(pred_ret, actual_ret),
            "pred_ret": pred_ret,
            "actual_ret": actual_ret,
        })
    return out
