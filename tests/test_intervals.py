"""
Tests for the conformal prediction intervals that replaced the volatility-only
"confidence" heuristic (Day 5).

The heuristic was unfalsifiable; the intervals make a checkable claim — the
80% band should contain ~80% of outcomes. These tests pin the finite-sample
quantile arithmetic by hand, the conservative small-n behaviour, the √h
widening, and the coverage claim itself on synthetic residuals.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.intervals import (
    build_interval_forecast,
    confidence_from_width,
    conformal_halfwidth,
    empirical_coverage,
    scale_for_horizon,
)


class TestConformalQuantile:
    def test_finite_sample_rank_hand_case(self):
        # n=10, alpha=0.2 -> rank ceil(11*0.8) = 9 -> 9th smallest |residual|.
        residuals = np.arange(1.0, 11.0)
        assert conformal_halfwidth(residuals, alpha=0.2) == 9.0

    def test_small_n_falls_back_to_max_conservative(self):
        # n=10, alpha=0.05 -> corrected rank 11 > n -> use the max score.
        # Coverage may exceed nominal; it can never be optimistic.
        residuals = np.arange(1.0, 11.0)
        assert conformal_halfwidth(residuals, alpha=0.05) == 10.0

    def test_uses_absolute_residuals(self):
        assert conformal_halfwidth([-5.0, 1.0, 2.0], alpha=0.2) == 5.0

    def test_ignores_nans(self):
        assert conformal_halfwidth([1.0, np.nan, 3.0], alpha=0.2) == 3.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            conformal_halfwidth([], alpha=0.2)


class TestHorizonScaling:
    def test_sqrt_h(self):
        assert scale_for_horizon(2.0, 4) == pytest.approx(4.0)

    def test_rejects_nonpositive_horizon(self):
        with pytest.raises(ValueError):
            scale_for_horizon(1.0, 0)


class TestCoverage:
    def test_empirical_coverage_hand_case(self):
        cov = empirical_coverage(
            actual=[1.0, 5.0, 10.0, np.nan],
            lower=[0.0, 6.0, 9.0, 0.0],
            upper=[2.0, 7.0, 11.0, 1.0],
        )
        assert cov == pytest.approx(2.0 / 3.0)   # NaN excluded, middle missed

    def test_conformal_band_hits_nominal_on_exchangeable_residuals(self):
        # The actual guarantee: calibrate on 500 draws, test on 4000 fresh
        # draws from the same fat-tailed distribution -> coverage ≈ 80%.
        rng = np.random.default_rng(42)
        calib = rng.standard_t(df=3, size=500) * 0.01
        test = rng.standard_t(df=3, size=4000) * 0.01
        q = conformal_halfwidth(calib, alpha=0.2)
        cov = empirical_coverage(test, np.full_like(test, -q), np.full_like(test, q))
        assert 0.77 <= cov <= 0.83


class TestConfidenceLabel:
    def test_thresholds(self):
        assert confidence_from_width(0.010)[0] == "High"
        assert confidence_from_width(0.020)[0] == "Medium"
        assert confidence_from_width(0.050)[0] == "Low"

    def test_score_is_nominal_coverage_until_band_is_useless(self):
        assert confidence_from_width(0.02)[1] == 80
        # ±10% of price at 80% confidence -> degraded score, floored at 50.
        assert confidence_from_width(0.10)[1] == 72
        assert confidence_from_width(0.50)[1] == 50

    def test_label_is_monotone_in_width(self):
        order = {"High": 0, "Medium": 1, "Low": 2}
        labels = [order[confidence_from_width(w)[0]]
                  for w in (0.001, 0.01, 0.02, 0.05, 0.2)]
        assert labels == sorted(labels)


class TestIntervalForecastBundle:
    def test_bands_widen_with_sqrt_horizon(self):
        iv = build_interval_forecast(
            residuals=np.arange(1.0, 11.0),
            point_forecasts=[100.0, 100.0, 100.0, 100.0],
            reference_price=100.0,
        )
        w = iv.upper80 - iv.lower80
        np.testing.assert_allclose(w / w[0], np.sqrt([1, 2, 3, 4]))

    def test_95_band_contains_80_band(self):
        iv = build_interval_forecast(
            residuals=np.random.default_rng(0).normal(0, 1, 200),
            point_forecasts=[50.0, 51.0],
            reference_price=50.0,
        )
        assert np.all(iv.lower95 <= iv.lower80)
        assert np.all(iv.upper95 >= iv.upper80)

    def test_horizon_mismatch_raises(self):
        with pytest.raises(ValueError, match="align"):
            build_interval_forecast(
                residuals=[1.0, 2.0],
                point_forecasts=[100.0, 101.0],
                reference_price=100.0,
                horizons=[1, 2, 3],
            )
