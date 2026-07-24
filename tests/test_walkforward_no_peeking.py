"""
No-peeking regression tests: the harness, the metrics, and the features.

Three layers of the same guarantee:
1. ``assert_no_peeking`` refuses folds whose test span touches training data.
2. ``walk_forward_predict`` scores against the correct realised returns and
   refuses models that return the wrong number of predictions.
3. ``assert_no_lookahead`` (features) PROVES every engineered column at row t
   is identical whether or not the future exists — rebuilt on a truncated
   series, the features must not change. This is the empirical check that
   catches centred windows, bfill, and full-series scalers in one sweep.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.backtest.walkforward import (
    Fold,
    assert_no_peeking,
    directional_accuracy,
    expanding_window_folds,
    returns_from_prices,
    rmse,
    walk_forward_predict,
)
from src.features.engineer import (
    FEATURE_COLS,
    assert_no_lookahead,
    build_feature_frame,
)


class TestAssertNoPeeking:
    def test_accepts_valid_folds(self, prices):
        folds = expanding_window_folds(len(prices), n_folds=5)
        assert_no_peeking(folds, n_samples=len(prices))  # must not raise

    def test_rejects_gap_or_overlap_at_boundary(self):
        bad = Fold(fold=0, train_start=0, train_end=60, test_start=50, test_end=80)
        with pytest.raises(AssertionError, match="test_start"):
            assert_no_peeking([bad])

    def test_rejects_empty_spans(self):
        empty_test = Fold(fold=0, train_start=0, train_end=60, test_start=60, test_end=60)
        with pytest.raises(AssertionError, match="empty"):
            assert_no_peeking([empty_test])

    def test_rejects_test_beyond_series(self):
        runaway = Fold(fold=0, train_start=0, train_end=60, test_start=60, test_end=999)
        with pytest.raises(AssertionError, match="exceeds"):
            assert_no_peeking([runaway], n_samples=100)


class TestWalkForwardPredict:
    def test_actual_returns_are_the_true_test_returns(self, prices):
        folds = expanding_window_folds(len(prices), n_folds=4)
        results = walk_forward_predict(prices, lambda p, f: np.zeros(f.test_len), folds)
        for f, r in zip(folds, results):
            expected = prices[f.test_start:f.test_end] / prices[f.test_start - 1:f.test_end - 1] - 1.0
            np.testing.assert_allclose(r["actual_ret"], expected)

    def test_rejects_wrong_prediction_length(self, prices):
        folds = expanding_window_folds(len(prices), n_folds=3)
        with pytest.raises(ValueError, match="preds"):
            walk_forward_predict(prices, lambda p, f: np.zeros(f.test_len + 1), folds)

    def test_perfect_model_scores_zero_rmse(self, prices):
        # A model handed the answers scores 0 — sanity that the scorer aligns
        # predictions to the right days (an off-by-one would not score 0).
        folds = expanding_window_folds(len(prices), n_folds=3)

        def oracle(p, f):
            return p[f.test_start:f.test_end] / p[f.test_start - 1:f.test_end - 1] - 1.0

        results = walk_forward_predict(prices, oracle, folds)
        assert all(r["rmse_ret"] < 1e-15 for r in results)
        assert all(r["dir_acc"] == 1.0 for r in results)


class TestReturnSpaceMetrics:
    def test_returns_from_prices(self):
        np.testing.assert_allclose(
            returns_from_prices([100.0, 110.0, 99.0]), [0.10, -0.10])

    def test_rmse_hand_case(self):
        assert rmse([0.0, 0.0], [0.03, -0.04]) == pytest.approx(0.035355339)

    def test_directional_accuracy_hand_case(self):
        acc = directional_accuracy([0.1, -0.2, 0.3], [0.2, -0.1, -0.3])
        assert acc == pytest.approx(2.0 / 3.0)

    def test_flat_actual_days_are_excluded(self):
        acc = directional_accuracy([0.1, 0.1], [0.0, 0.05])
        assert acc == 1.0

    def test_zero_prediction_earns_no_directional_credit(self):
        # r̂=0 takes no position: it cannot be "right" about an up day.
        assert directional_accuracy([0.0], [0.02]) == 0.0

    def test_all_flat_returns_nan(self):
        assert np.isnan(directional_accuracy([0.1], [0.0]))


class TestFeatureNoLookahead:
    def test_base_features_pass_the_truncation_proof(self, long_prices):
        assert_no_lookahead(long_prices, probe_at=300)

    def test_extended_features_pass_the_truncation_proof(self, long_prices, long_dates):
        assert_no_lookahead(long_prices, probe_at=380,
                            dates=long_dates, extended=True)

    def test_target_is_strictly_tomorrows_return(self, long_prices):
        df = build_feature_frame(long_prices)
        t = 100
        expected = long_prices[t + 1] / long_prices[t] - 1.0
        assert df.loc[t, "target"] == pytest.approx(expected)

    def test_final_row_target_is_nan(self, long_prices):
        df = build_feature_frame(long_prices)
        assert np.isnan(df["target"].iloc[-1])

    def test_warmup_rows_are_invalid(self, long_prices):
        df = build_feature_frame(long_prices)
        assert not df["valid"].iloc[:60].any()
        assert df["valid"].iloc[100:].all()

    def test_rejects_series_too_short_to_warm_up(self):
        with pytest.raises(ValueError, match="warm up"):
            build_feature_frame(np.linspace(100, 110, 40))

    def test_feature_columns_all_present(self, long_prices):
        df = build_feature_frame(long_prices)
        assert set(FEATURE_COLS).issubset(df.columns)
