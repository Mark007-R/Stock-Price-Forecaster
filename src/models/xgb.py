"""
XGBoost on engineered features — kept as the honest negative result.

Day 3 measured it 9.3% WORSE than predicting zero, with feature gain spread
almost uniformly across all 18 inputs (the fingerprint of trees splitting on
noise), and Day 4 showed that adding calendar/regime features made it worse
still. It stays in the registry because the ablation story needs it, and
because a production layout that silently drops its failures is curating.

Leakage protocol: row t pairs features known at the close of day t with the
return realised on day t+1, so training stops at row ``train_end - 2`` — no
training row's target falls inside the test block. ``assert_no_lookahead``
in ``src.features.engineer`` proves the features causal empirically.

Day 6 added a 40-trial Optuna sweep (inner-validation windows carved off each
fold's train slice — outer tests untouched). The optimum it found is a
regularisation corner: learning-rate 31x smaller than default, depth 3,
min_child_weight 15 — a config whose mean |prediction| is 5.6x smaller than
default. That shrinkage bought back random-walk RMSE parity (ratio 1.107 ->
1.001) but no directional edge; ``TUNED_PARAMS`` is kept because honest parity
beats confidently-wrong, and because the ablation needs the datapoint. The
tuned variant below also applies the one fix from the Day-6 failure-mode
analysis that helped at all: exponential time-decay sample weights (half-life
126 trading days), which lifted directional accuracy to 0.531 — the best of
any XGB config in the sprint, and still below the 0.549 always-up baseline.
"""
from __future__ import annotations

import numpy as np

from src.features.engineer import FEATURE_COLS

SEED = 42
PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, n_jobs=4, verbosity=0,
)

# Optuna best trial (#25 of 40), Day 6 — see results/phase4_optuna_trials.csv.
TUNED_PARAMS = dict(
    n_estimators=555, max_depth=3, learning_rate=0.00161,
    subsample=0.615082, colsample_bytree=0.502346,
    min_child_weight=15.185263, reg_lambda=7.825181, reg_alpha=0.051148,
    random_state=SEED, n_jobs=4, verbosity=0,
)

# Time-decay half-life for the tuned variant (trading days). Chosen on the
# Day-6 walk-forward, where it was the only targeted fix that moved
# directional accuracy at all (+1.5 pp over default).
DECAY_HALF_LIFE = 126


def _rows(feats, fold):
    tr = np.asarray(feats.index[(feats["valid"]) & (feats.index <= fold.train_end - 2)])
    te = np.arange(fold.test_start - 1, fold.test_end - 1)
    return tr, te


def _fit_predict(feats, fold, params, decay_half_life=None, gain_sink=None):
    from xgboost import XGBRegressor

    tr_rows, te_rows = _rows(feats, fold)
    Xtr = feats.loc[tr_rows, FEATURE_COLS].to_numpy(dtype=float)
    ytr = feats.loc[tr_rows, "target"].to_numpy(dtype=float)
    Xte = feats.loc[te_rows, FEATURE_COLS].to_numpy(dtype=float)

    w = None
    if decay_half_life is not None:
        age = tr_rows.max() - tr_rows.astype(float)
        w = 0.5 ** (age / float(decay_half_life))

    model = XGBRegressor(**params)
    model.fit(Xtr, ytr, sample_weight=w)
    if gain_sink is not None:
        gain_sink.append(
            dict(zip(FEATURE_COLS, model.feature_importances_.astype(float))))
    return model.predict(Xte).astype(float)


def predict_fold(prices: np.ndarray, fold, ctx: dict) -> np.ndarray:
    """Fit on pre-fold feature rows, predict the fold's next-day returns."""
    if "feats" not in ctx:
        raise ValueError(
            "xgboost needs ctx built with make_context(..., with_features=True)")
    return _fit_predict(ctx["feats"], fold, PARAMS,
                        gain_sink=ctx.setdefault("xgb_gain", []))


def predict_fold_tuned(prices: np.ndarray, fold, ctx: dict) -> np.ndarray:
    """Day-6 tuned variant: Optuna params + time-decay sample weights.

    The best XGB directional accuracy of the sprint (0.531 walk-forward mean)
    — reported next to the fact that always-up scores 0.549 on the same days.
    """
    if "feats" not in ctx:
        raise ValueError(
            "xgboost needs ctx built with make_context(..., with_features=True)")
    return _fit_predict(ctx["feats"], fold, TUNED_PARAMS,
                        decay_half_life=DECAY_HALF_LIFE)
