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


def predict_fold(prices: np.ndarray, fold, ctx: dict) -> np.ndarray:
    """Fit on pre-fold feature rows, predict the fold's next-day returns."""
    from xgboost import XGBRegressor

    if "feats" not in ctx:
        raise ValueError(
            "xgboost needs ctx built with make_context(..., with_features=True)")
    feats = ctx["feats"]
    tr_rows = feats.index[(feats["valid"]) & (feats.index <= fold.train_end - 2)]
    te_rows = np.arange(fold.test_start - 1, fold.test_end - 1)

    Xtr = feats.loc[tr_rows, FEATURE_COLS].to_numpy(dtype=float)
    ytr = feats.loc[tr_rows, "target"].to_numpy(dtype=float)
    Xte = feats.loc[te_rows, FEATURE_COLS].to_numpy(dtype=float)

    model = XGBRegressor(**PARAMS)
    model.fit(Xtr, ytr)
    ctx.setdefault("xgb_gain", []).append(
        dict(zip(FEATURE_COLS, model.feature_importances_.astype(float))))
    return model.predict(Xte).astype(float)
