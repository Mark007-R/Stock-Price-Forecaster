"""
Multi-ticker portfolio backtest — position sizing, stop-loss, cost sweeps.

Why this exists
---------------
Days 3-6 scored every model per ticker, one equity curve at a time. That
answers "does the signal work on AAPL?" but not the question an allocator
asks: **across a book of names, does sizing and risk management rescue a weak
per-name signal — or just add turnover?** Diversification mechanically lifts
Sharpe (uncorrelated noise cancels), so a portfolio backtest can flatter a
model that loses per-ticker. The only honest read is the same portfolio
machinery applied to the benchmark: equal-weight buy-and-hold of the SAME
tickers over the SAME days. Every result here is reported next to it.

The rules of the simulation
---------------------------
* **Sleeves.** Each ticker owns a sleeve of the book. The sleeve is IN when
  the model predicted a positive return for that ticker/day (decided at the
  prior close — no same-day peeking; the inputs are the out-of-sample
  walk-forward predictions, never refit here) and CASH otherwise.
* **Sizing schemes.**
  - ``equal_sleeve``  — every sleeve is 1/N of the book. Cash earns 0.
  - ``equal_active``  — the book splits evenly across today's active signals
    (fully invested when anything is on; concentration risk when few are).
  - ``inv_vol``       — sleeve budgets proportional to 1/vol21 (trailing,
    shifted one day so the weight for day t uses only data through t-1),
    normalised to sum to 1. The classic risk-parity-lite sizing.
  - ``signal_prop``   — sleeve budgets proportional to the predicted return
    magnitude (positive side only), capped so the book never exceeds 1.0
    gross. Tests whether forecast MAGNITUDE carries information beyond sign.
* **Stop-loss overlay (optional).** Per sleeve: track the equity of the open
  position since entry; if it falls ``stop_loss`` below the entry, force the
  sleeve to cash. The sleeve may re-enter only on a FRESH up-signal after at
  least one flat day (re-arm), so a stopped position cannot instantly re-buy
  the same falling knife.
* **Costs.** Charged on target-weight turnover: sum_i |w_i,t - w_i,t-1| x
  cost. Same 5 bps/side default as Days 3-6. (Drift between rebalances is
  not charged — a small understatement that flatters the ACTIVE strategies,
  not the benchmark, so it cannot manufacture a win for buy-and-hold.)
* **Benchmark.** Equal-weight buy-and-hold with drift: 1/N in each name on
  day one, one entry cost per sleeve, never touched again. No rebalancing —
  rebalancing is a strategy, and the benchmark should not have one.

Nothing here fabricates returns: inputs are realised out-of-sample returns
and frozen out-of-sample predictions; this module is arithmetic on top.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

SCHEMES = ("equal_sleeve", "equal_active", "inv_vol", "signal_prop")


@dataclass
class PortfolioResult:
    """Outcome of one portfolio configuration over the aligned OOS span."""
    scheme: str
    stop_loss: float | None
    cost_bps: float
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    exposure: float            # mean gross weight in the market
    daily_turnover: float      # mean sum |dw| per day
    cost_drag: float           # total return given up to costs
    n_stop_hits: int           # sleeve-days forced flat by the stop
    net_daily: np.ndarray = field(repr=False, default=None)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("net_daily")
        return d


def align_panel(per_ticker: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Inner-join per-ticker OOS frames on date into (dates x tickers) panels.

    Each input frame needs columns ``date``, ``pred``, ``actual`` and
    optionally ``vol21``. Returns dict of pivoted panels keyed by column.
    All ten sprint tickers trade the NYSE/NASDAQ calendar, so the join drops
    at most a handful of days; the count is the caller's job to log.
    """
    frames = []
    for tk, df in per_ticker.items():
        f = df.copy()
        f["ticker"] = tk
        frames.append(f)
    long = pd.concat(frames, ignore_index=True)
    panels = {}
    for col in ("pred", "actual", "vol21"):
        if col in long.columns:
            panels[col] = long.pivot(index="date", columns="ticker",
                                     values=col).dropna(how="any")
    # Restrict every panel to the common dates of all panels.
    common = panels["pred"].index
    for col in panels:
        common = common.intersection(panels[col].index)
    return {col: p.loc[common] for col, p in panels.items()}


def _weights(scheme: str, signal_on: np.ndarray, pred: np.ndarray,
             vol: np.ndarray | None) -> np.ndarray:
    """Target weights for one day given the active-signal mask. Sum <= 1."""
    n = len(signal_on)
    on = signal_on.astype(float)
    if scheme == "equal_sleeve":
        return on / n
    if scheme == "equal_active":
        k = on.sum()
        return on / k if k > 0 else np.zeros(n)
    if scheme == "inv_vol":
        if vol is None:
            raise ValueError("inv_vol sizing needs a vol21 panel")
        iv = np.where(vol > 1e-12, 1.0 / vol, 0.0)
        denom = iv.sum()
        base = iv / denom if denom > 0 else np.zeros(n)
        return base * on
    if scheme == "signal_prop":
        mag = np.maximum(pred, 0.0) * on
        s = mag.sum()
        # Normalise to fully invested when anything is on, like equal_active;
        # magnitude only decides HOW the invested book is split.
        return mag / s if s > 1e-15 else np.zeros(n)
    raise ValueError(f"unknown scheme: {scheme}")


def portfolio_backtest(
    panels: dict[str, pd.DataFrame],
    scheme: str = "equal_sleeve",
    stop_loss: float | None = None,
    cost_bps: float = 5.0,
) -> PortfolioResult:
    """Run one portfolio configuration over the aligned panels."""
    pred = panels["pred"].to_numpy(dtype=float)
    act = panels["actual"].to_numpy(dtype=float)
    vol = panels.get("vol21")
    vol = vol.to_numpy(dtype=float) if vol is not None else None
    n_days, n_tk = act.shape
    cost = cost_bps / 10_000.0

    prev_w = np.zeros(n_tk)
    in_pos = np.zeros(n_tk, dtype=bool)      # sleeve currently holding
    entry_eq = np.ones(n_tk)                 # sleeve equity at entry
    sleeve_eq = np.ones(n_tk)                # sleeve equity since entry
    stopped = np.zeros(n_tk, dtype=bool)     # stop hit, waiting to re-arm
    n_stop_hits = 0

    net_daily = np.zeros(n_days)
    gross_daily = np.zeros(n_days)
    turnover_daily = np.zeros(n_days)
    exposure_daily = np.zeros(n_days)

    for t in range(n_days):
        signal_on = pred[t] > 0.0
        # Re-arm: a stopped sleeve may trade again only after a flat signal.
        stopped &= signal_on                 # signal went flat -> re-armed
        active = signal_on & ~stopped

        w = _weights(scheme, active, pred[t], vol[t] if vol is not None else None)

        turnover = np.abs(w - prev_w).sum()
        day_gross = float((w * act[t]).sum())
        net_daily[t] = day_gross - turnover * cost
        gross_daily[t] = day_gross
        turnover_daily[t] = turnover
        exposure_daily[t] = w.sum()

        # Track per-sleeve equity for the stop-loss, based on holding state.
        newly_in = (w > 0) & ~in_pos
        entry_eq[newly_in] = sleeve_eq[newly_in] = 1.0
        in_pos = w > 0
        sleeve_eq[in_pos] *= (1.0 + act[t][in_pos])
        if stop_loss is not None:
            hit = in_pos & (sleeve_eq / entry_eq - 1.0 < -abs(stop_loss))
            if hit.any():
                n_stop_hits += int(hit.sum())
                stopped |= hit               # forced flat from tomorrow
        prev_w = w

    return _finish(scheme, stop_loss, cost_bps, net_daily, gross_daily,
                   turnover_daily, exposure_daily, n_stop_hits)


def buy_and_hold_portfolio(panels: dict[str, pd.DataFrame],
                           cost_bps: float = 5.0) -> PortfolioResult:
    """1/N in every name on day one, entry cost once, weights drift, no touch."""
    act = panels["actual"].to_numpy(dtype=float)
    n_days, n_tk = act.shape
    cost = cost_bps / 10_000.0

    sleeve_eq = np.full(n_tk, (1.0 - cost) / n_tk)   # entry cost per sleeve
    equity = np.zeros(n_days)
    prev_total = sleeve_eq.sum()
    net_daily = np.zeros(n_days)
    for t in range(n_days):
        sleeve_eq *= (1.0 + act[t])
        total = sleeve_eq.sum()
        net_daily[t] = total / prev_total - 1.0
        prev_total = total
        equity[t] = total

    gross_daily = act.mean(axis=1)                    # pre-cost reference
    return _finish("buy_and_hold", None, cost_bps, net_daily, gross_daily,
                   np.zeros(n_days), np.ones(n_days), 0)


def _finish(scheme, stop_loss, cost_bps, net_daily, gross_daily,
            turnover_daily, exposure_daily, n_stop_hits) -> PortfolioResult:
    equity = np.cumprod(1.0 + net_daily)
    n = len(net_daily)
    total = float(equity[-1] - 1.0)
    ann_ret = float(equity[-1] ** (TRADING_DAYS / n) - 1.0) if n else 0.0
    ann_vol = float(np.std(net_daily, ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0

    downside = net_daily[net_daily < 0]
    # Sortino with rf=0: annualised return over annualised DOWNSIDE deviation
    # (computed over all days, squaring only the negative ones).
    dd_dev = float(np.sqrt(np.mean(np.minimum(net_daily, 0.0) ** 2)) * np.sqrt(TRADING_DAYS))
    sortino = float(ann_ret / dd_dev) if dd_dev > 1e-12 else 0.0

    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0))

    gross_total = float(np.cumprod(1.0 + gross_daily)[-1] - 1.0)
    return PortfolioResult(
        scheme=scheme, stop_loss=stop_loss, cost_bps=cost_bps,
        total_return=total, ann_return=ann_ret, ann_vol=ann_vol,
        sharpe=sharpe, sortino=sortino, max_drawdown=max_dd,
        exposure=float(np.mean(exposure_daily)),
        daily_turnover=float(np.mean(turnover_daily)),
        cost_drag=float(gross_total - total),
        n_stop_hits=n_stop_hits, net_daily=net_daily,
    )


def regime_slice(net_daily: np.ndarray, dates: pd.Index,
                 regimes: pd.Series) -> pd.DataFrame:
    """Annualised return/vol/Sharpe of a daily net-return stream per regime.

    ``regimes`` is a date-indexed Series of labels (bull/bear/sideways),
    already shifted so the label for day t uses only data through t-1.
    """
    df = pd.DataFrame({"net": net_daily}, index=dates)
    df["regime"] = regimes.reindex(df.index)
    rows = []
    for label, g in df.dropna().groupby("regime"):
        n = len(g)
        mu = g["net"].mean() * TRADING_DAYS
        sd = g["net"].std(ddof=1) * np.sqrt(TRADING_DAYS)
        rows.append({
            "regime": label, "n_days": n,
            "ann_return": float(mu),
            "ann_vol": float(sd),
            "sharpe": float(mu / sd) if sd > 1e-12 else 0.0,
        })
    return pd.DataFrame(rows)
