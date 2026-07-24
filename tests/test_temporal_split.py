"""
Tests for the walk-forward fold generators — the temporal-split contract.

Every fold must fit strictly on the past and score strictly on the future:
``test_start == train_end``, test blocks contiguous and non-overlapping,
nothing out of range. These are the invariants ``assert_no_peeking`` enforces
at runtime; here they are pinned for both generators across shapes and edge
cases so a refactor cannot quietly break the geometry.
"""
from __future__ import annotations

import pytest

from src.backtest.walkforward import (
    Fold,
    expanding_window_folds,
    rolling_window_folds,
)


class TestExpandingWindow:
    def test_default_geometry_n100(self):
        folds = expanding_window_folds(100, n_folds=5)
        # default test_size = (100//2)//5 = 10, min_train = 50
        assert len(folds) == 5
        assert [f.train_end for f in folds] == [50, 60, 70, 80, 90]
        assert [f.test_end for f in folds] == [60, 70, 80, 90, 100]
        assert all(f.train_start == 0 for f in folds), "expanding = pinned start"

    def test_boundary_no_overlap(self):
        for f in expanding_window_folds(500, n_folds=7):
            assert f.test_start == f.train_end

    def test_test_blocks_are_contiguous(self):
        folds = expanding_window_folds(500, n_folds=7)
        for a, b in zip(folds, folds[1:]):
            assert b.test_start == a.test_end

    def test_train_grows_monotonically(self):
        folds = expanding_window_folds(400, n_folds=5)
        lens = [f.train_len for f in folds]
        assert lens == sorted(lens) and len(set(lens)) == len(lens)

    def test_explicit_sizes_respected(self):
        folds = expanding_window_folds(300, n_folds=4, test_size=25, min_train=200)
        assert folds[0].train_len == 200
        assert all(f.test_len == 25 for f in folds)
        assert folds[-1].test_end == 300

    def test_rejects_empty_series(self):
        with pytest.raises(ValueError):
            expanding_window_folds(0, n_folds=3)

    def test_rejects_zero_folds(self):
        with pytest.raises(ValueError):
            expanding_window_folds(100, n_folds=0)

    def test_rejects_when_no_room_to_train(self):
        # 5 folds x 2 test days = all 10 samples -> nothing left to train on.
        with pytest.raises(ValueError):
            expanding_window_folds(10, n_folds=5, test_size=2)


class TestRollingWindow:
    def test_fixed_width_slides(self):
        folds = rolling_window_folds(100, n_folds=5)
        assert len(folds) == 5
        assert all(f.train_len == 50 for f in folds), "rolling = fixed width"
        assert [f.train_start for f in folds] == [0, 10, 20, 30, 40]
        assert folds[-1].test_end == 100

    def test_boundary_no_overlap(self):
        for f in rolling_window_folds(500, n_folds=6):
            assert f.test_start == f.train_end

    def test_rejects_train_window_too_small(self):
        with pytest.raises(ValueError):
            rolling_window_folds(10, n_folds=5, test_size=2)


class TestFoldProperties:
    def test_lengths(self):
        f = Fold(fold=0, train_start=10, train_end=60, test_start=60, test_end=75)
        assert f.train_len == 50
        assert f.test_len == 15
