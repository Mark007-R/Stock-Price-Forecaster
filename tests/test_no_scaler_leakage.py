"""
Regression tests for the Day-1 scaler-leakage fix — the sprint's first commit.

The original bug: ``MinMaxScaler().fit_transform(close_prices)`` ran on the
FULL price series BEFORE the train/test split in both ``predictor.py`` (lines
78-79) and ``stock_predictor.py`` (line 230), leaking the test window's
min/max into the training normalisation and understating the reported errors.

These tests pin the fix structurally (the scaler must be fit on a SLICE of the
series, and ``fit_transform`` must not reappear) and behaviourally (a
train-only fit provably cannot know the test-period maximum). If a refactor
reintroduces a full-series fit, this file is what fails.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_SRC = (ROOT / "predictor.py").read_text(encoding="utf-8")
STREAMLIT_SRC = (ROOT / "stock_predictor.py").read_text(encoding="utf-8")


def _fit_calls(source: str) -> list[ast.Call]:
    """All ``scaler.fit(...)`` / ``scaler.fit_transform(...)`` calls in a file."""
    calls = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("fit", "fit_transform")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "scaler"):
            calls.append(node)
    return calls


def _fit_transform_calls(source: str) -> list[ast.Call]:
    """Every ``<anything>.fit_transform(...)`` CALL in the code (AST-level, so
    the fix's own explanatory comments don't false-positive)."""
    return [node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit_transform"]


class TestPredictorScaler:
    def test_no_fit_transform_anywhere(self):
        # The buggy one-liner fit AND transformed on the full series. It must
        # not come back in any form (checked on the AST, so the fix's own
        # comments describing the old bug don't trip it).
        assert not _fit_transform_calls(PREDICTOR_SRC), (
            "predictor.py calls fit_transform again — the Day-1 leakage fix "
            "requires fit(train_slice) then transform(full)."
        )

    def test_scaler_fit_argument_is_a_slice(self):
        fits = [c for c in _fit_calls(PREDICTOR_SRC) if c.func.attr == "fit"]
        assert fits, "predictor.py no longer fits a scaler at all?"
        for call in fits:
            arg = call.args[0]
            assert isinstance(arg, ast.Subscript), (
                "scaler.fit() is called on something other than a slice of "
                "the price array — full-series fits leak the test min/max."
            )

    def test_split_defined_before_fit(self):
        # The train boundary must exist before the scaler is fit on it.
        split_pos = PREDICTOR_SRC.index("raw_split = int(")
        fit_pos = PREDICTOR_SRC.index("scaler.fit(")
        assert split_pos < fit_pos

    def test_fit_uses_the_train_boundary(self):
        assert "scaler.fit(close_prices[:raw_split])" in PREDICTOR_SRC


class TestStreamlitScaler:
    def test_no_fit_transform_anywhere(self):
        assert not _fit_transform_calls(STREAMLIT_SRC), (
            "stock_predictor.py calls fit_transform again — the same "
            "leakage the Day-1 fix removed at its line 230."
        )

    def test_fit_uses_the_train_boundary(self):
        assert "scaler.fit(data[['Close']].iloc[:raw_split])" in STREAMLIT_SRC


class TestLeakageMechanism:
    """Demonstrate the bug the fix prevents, so the test explains itself."""

    def test_train_only_fit_cannot_see_test_maximum(self):
        # Rising series: the global max lives in the test window.
        prices = np.linspace(100.0, 200.0, 250).reshape(-1, 1)
        split = int(len(prices) * 0.8)

        honest = MinMaxScaler().fit(prices[:split])
        leaky = MinMaxScaler().fit(prices)          # the old behaviour

        assert honest.data_max_[0] == prices[split - 1, 0]
        assert leaky.data_max_[0] == prices[-1, 0]
        # The leaky scaler maps the last TRAIN price below 1.0 — it "knows"
        # prices will keep rising. The honest scaler cannot.
        assert leaky.transform(prices[split - 1:split])[0, 0] < 0.999
        assert honest.transform(prices[split - 1:split])[0, 0] == pytest.approx(1.0)

    def test_transform_may_exceed_unit_range_out_of_sample(self):
        # Correct post-fix behaviour: unseen test prices above the train max
        # scale to >1. If every scaled test value is capped inside [0, 1],
        # someone re-fit on the full series.
        prices = np.linspace(100.0, 200.0, 250).reshape(-1, 1)
        split = int(len(prices) * 0.8)
        scaler = MinMaxScaler().fit(prices[:split])
        assert scaler.transform(prices[split:]).max() > 1.0
