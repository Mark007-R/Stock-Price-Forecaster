"""
Cost-aware trading backtest — turning a forecast into a P&L, honestly.

Why this exists
---------------
RMSE and directional accuracy are *statistical* scores. They do not answer the
only question that matters for a trading model: **after costs, would this have
made money — and more money than simply buying and holding?** Day 2 left the
LSTM tied with a random walk on RMSE and below an always-up baseline on
direction. Day 3 asks whether any model in the bake-off survives contact with
a realistic execution model.

The rules of the simulation
---------------------------
* **Signal.** Long-only: hold the stock for day ``t`` when the model predicted
  a positive return for day ``t``, otherwise hold cash. This is the mapping the
  Day-3 spec calls for ("long when predicted up"). A model that predicts
  exactly 0 (the random walk) therefore never takes a position and earns 0 —
  the correct outcome, not a bug: it forecasts no edge, so it bets nothing.

* **Costs.** Charged on TURNOVER, i.e. whenever the position changes. Entering
  costs ``cost_bps``; exiting costs ``cost_bps`` again. A strategy that flips
  in and out daily pays this ~250×/year, which is exactly the effect a
  cost-free backtest hides. The default 5 bps/side is a realistic retail
  all-in estimate (commission + spread + slippage) for liquid large-caps; Day 7
  sweeps this parameter to find the break-even.

* **No shorting, no leverage, no compounding assumptions beyond the equity
  curve.** Keeping the execution model simple keeps the comparison honest —
  every method faces the identical rules.

Why buy-and-hold is the benchmark that matters
----------------------------------------------
A long-only strategy on a rising market inherits the market's drift. Beating
"zero" is meaningless; the alternative to a trading model is not cash, it is
*owning the asset and doing nothing* — which costs one commission, ever, and
has beaten most active strategies. So every number here is reported next to
buy-and-hold on the identical days. A backtest without that column is theater.

Nothing in this module fabricates returns: it consumes realised out-of-sample
returns from the walk-forward folds and applies arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Outcome of one strategy over one span of realised returns."""
    total_return: float        # cumulative, net of costs (0.10 = +10%)
    ann_return: float          # geometric, annualised
    ann_vol: float             # annualised stdev of daily net returns
    sharpe: float              # ann_return / ann_vol, rf = 0
    max_drawdown: float        # worst peak-to-trough, negative
    n_trades: int              # position changes (each side of a round trip)
    exposure: float            # fraction of days holding the asset
    cost_drag: float           # total return given up to costs
    gross_return: float        # cumulative BEFORE costs

    def as_dict(self) -> dict:
        return asdict(self)


def _equity_metrics(net_daily: np.ndarray) -> tuple[float, float, float, float, float]:
    """Cumulative/annualised/vol/sharpe/max-dd from a net daily return stream."""
    equity = np.cumprod(1.0 + net_daily)
    total = float(equity[-1] - 1.0)

    n = len(net_daily)
    # Geometric annualisation — compounding the average daily result.
    ann_ret = float(equity[-1] ** (TRADING_DAYS / n) - 1.0) if n > 0 else 0.0
    ann_vol = float(np.std(net_daily, ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    # rf = 0. A flat strategy has zero vol -> Sharpe is undefined, report 0.0
    # rather than an inf that would top the leaderboard on a divide-by-zero.
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0

    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0))
    return total, ann_ret, ann_vol, sharpe, max_dd


def backtest_long_flat(
    pred_ret: np.ndarray,
    actual_ret: np.ndarray,
    cost_bps: float = 5.0,
    threshold: float = 0.0,
) -> BacktestResult:
    """Long-when-predicted-up / flat-otherwise, charged on turnover.

    ``pred_ret[i]`` is the model's forecast for the return realised on day
    ``i`` (``actual_ret[i]``). The position for day ``i`` is decided from that
    forecast, which was made at the close of day ``i-1`` — so the position is
    knowable before the return happens. No same-day peeking.

    ``threshold`` demands the forecast clear a bar before taking the position
    (0.0 = any positive forecast). Used by the cost-sensitivity work later.
    """
    pred = np.asarray(pred_ret, dtype=float).flatten()
    act = np.asarray(actual_ret, dtype=float).flatten()
    if len(pred) != len(act):
        raise ValueError(f"pred/actual length mismatch: {len(pred)} vs {len(act)}")

    cost = cost_bps / 10_000.0
    position = (pred > threshold).astype(float)

    # Turnover: start flat, so the first entry is charged. Each change of
    # position (0->1 or 1->0) pays one side of the spread.
    prev = np.concatenate([[0.0], position[:-1]])
    turnover = np.abs(position - prev)

    gross_daily = position * act
    net_daily = gross_daily - turnover * cost

    total, ann_ret, ann_vol, sharpe, max_dd = _equity_metrics(net_daily)
    gross_total = float(np.cumprod(1.0 + gross_daily)[-1] - 1.0)

    return BacktestResult(
        total_return=total,
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        n_trades=int(turnover.sum()),
        exposure=float(position.mean()),
        cost_drag=float(gross_total - total),
        gross_return=gross_total,
    )


def backtest_buy_and_hold(
    actual_ret: np.ndarray,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Own the asset for every day in the span. One entry cost, then nothing.

    This is the benchmark every model must clear to justify its existence.
    """
    act = np.asarray(actual_ret, dtype=float).flatten()
    cost = cost_bps / 10_000.0

    net_daily = act.copy()
    net_daily[0] -= cost                     # single entry commission, then hold

    total, ann_ret, ann_vol, sharpe, max_dd = _equity_metrics(net_daily)
    gross_total = float(np.cumprod(1.0 + act)[-1] - 1.0)

    return BacktestResult(
        total_return=total,
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        n_trades=1,
        exposure=1.0,
        cost_drag=float(gross_total - total),
        gross_return=gross_total,
    )
