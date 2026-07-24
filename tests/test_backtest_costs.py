"""
Tests for the cost-aware trading backtest — arithmetic pinned by hand.

Rule 8 of the sprint: a backtest without transaction costs is theater. These
tests verify the cost model with hand-computed equity curves, confirm costs
are charged on TURNOVER (not per day), that higher costs strictly hurt an
active strategy, and that buy-and-hold pays exactly one entry commission.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.backtest.trading import (
    backtest_buy_and_hold,
    backtest_long_flat,
)


class TestLongFlatHandComputed:
    """pred=[+,-,+], act=[2%,3%,-1%], 10 bps/side — fully worked example."""

    @pytest.fixture
    def result(self):
        return backtest_long_flat(
            pred_ret=[0.01, -0.01, 0.02],
            actual_ret=[0.02, 0.03, -0.01],
            cost_bps=10.0,
        )

    def test_positions_and_turnover(self, result):
        # positions [1,0,1]: enter, exit, re-enter = 3 turnover events.
        assert result.n_trades == 3
        assert result.exposure == pytest.approx(2.0 / 3.0)

    def test_net_equity_curve(self, result):
        # net daily = [0.02-0.001, 0-0.001, -0.01-0.001]
        expected = (1.019 * 0.999 * 0.989) - 1.0
        assert result.total_return == pytest.approx(expected)

    def test_gross_and_cost_drag_reconcile(self, result):
        gross = (1.02 * 1.00 * 0.99) - 1.0
        assert result.gross_return == pytest.approx(gross)
        assert result.cost_drag == pytest.approx(gross - result.total_return)

    def test_max_drawdown_is_negative(self, result):
        assert result.max_drawdown < 0.0


class TestCostModel:
    def test_zero_forecast_takes_no_position_and_pays_nothing(self):
        r = backtest_long_flat(np.zeros(50), np.full(50, 0.01), cost_bps=5.0)
        assert r.total_return == 0.0
        assert r.n_trades == 0
        assert r.exposure == 0.0
        assert r.sharpe == 0.0, "flat strategy must not divide by zero vol"

    def test_higher_costs_strictly_hurt_an_active_strategy(self):
        rng = np.random.default_rng(3)
        pred = rng.normal(0, 0.01, 200)
        act = rng.normal(0.0005, 0.01, 200)
        totals = [backtest_long_flat(pred, act, cost_bps=b).total_return
                  for b in (1.0, 5.0, 25.0)]
        assert totals[0] > totals[1] > totals[2]

    def test_costs_charged_on_turnover_not_holding(self):
        # Always-long: one entry, then held — a single cost event regardless
        # of span length.
        pred = np.full(100, 0.01)
        act = np.full(100, 0.001)
        r = backtest_long_flat(pred, act, cost_bps=10.0)
        assert r.n_trades == 1
        assert r.cost_drag == pytest.approx(0.001, rel=0.15)

    def test_threshold_filters_weak_signals(self):
        r = backtest_long_flat([0.001, 0.02], [0.05, 0.05],
                               cost_bps=0.1, threshold=0.01)
        assert r.exposure == 0.5, "only the forecast above the bar trades"

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            backtest_long_flat([0.01], [0.01, 0.02])


class TestBuyAndHold:
    def test_single_entry_cost_then_hold(self):
        r = backtest_buy_and_hold([0.01, 0.02], cost_bps=5.0)
        expected = (1.0 + 0.01 - 0.0005) * 1.02 - 1.0
        assert r.total_return == pytest.approx(expected)
        assert r.n_trades == 1
        assert r.exposure == 1.0

    def test_gross_ignores_the_commission(self):
        r = backtest_buy_and_hold([0.01, 0.02], cost_bps=5.0)
        assert r.gross_return == pytest.approx(1.01 * 1.02 - 1.0)

    def test_benchmark_beats_a_flat_strategy_in_a_rising_market(self):
        act = np.full(252, 0.001)
        bh = backtest_buy_and_hold(act, cost_bps=5.0)
        flat = backtest_long_flat(np.zeros(252), act, cost_bps=5.0)
        assert bh.total_return > flat.total_return
        assert bh.sharpe > flat.sharpe
