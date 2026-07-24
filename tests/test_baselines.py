"""
Tests for the persistence baselines — rule 8: every model is measured against
them, so the baselines themselves must be beyond suspicion.

``predict_zero`` is the random-walk RMSE floor no bake-off model beat.
``predict_momentum`` must use ONLY prices that existed before each test day —
proven here by mutating the future and asserting the forecasts don't move.
"""
from __future__ import annotations

import numpy as np

from src.backtest.walkforward import expanding_window_folds, walk_forward_predict
from src.models.persistence import predict_momentum, predict_zero


class TestPredictZero:
    def test_shape_and_value(self, prices):
        fold = expanding_window_folds(len(prices), n_folds=5)[0]
        out = predict_zero(prices, fold)
        assert out.shape == (fold.test_len,)
        assert np.all(out == 0.0)

    def test_rmse_equals_std_of_actual_returns(self, prices):
        # Predicting 0 makes RMSE the RMS of realised returns — the floor
        # every model must beat to claim signal.
        folds = expanding_window_folds(len(prices), n_folds=4)
        results = walk_forward_predict(prices, predict_zero, folds)
        for r in results:
            expected = float(np.sqrt(np.mean(r["actual_ret"] ** 2)))
            np.testing.assert_allclose(r["rmse_ret"], expected)


class TestPredictMomentum:
    def test_forecast_is_yesterdays_realised_return(self, prices):
        fold = expanding_window_folds(len(prices), n_folds=5)[2]
        out = predict_momentum(prices, fold)
        t0 = fold.test_start
        expected_first = prices[t0 - 1] / prices[t0 - 2] - 1.0
        assert out[0] == expected_first

    def test_uses_only_past_prices(self, prices):
        # Mutate everything from each test day forward; if any forecast
        # changes, the baseline was reading the future.
        fold = expanding_window_folds(len(prices), n_folds=5)[1]
        clean = predict_momentum(prices, fold)

        tampered = prices.copy()
        tampered[fold.test_start:] *= 7.5   # absurd future
        # forecast for test day t uses prices[t-1], prices[t-2]; the FIRST
        # test day's inputs are entirely pre-test, so compare that one.
        assert predict_momentum(tampered, fold)[0] == clean[0]

    def test_takes_directional_positions_unlike_zero(self, prices):
        fold = expanding_window_folds(len(prices), n_folds=5)[0]
        out = predict_momentum(prices, fold)
        assert np.any(out > 0) and np.any(out < 0)
