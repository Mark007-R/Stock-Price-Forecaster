"""
Residual-based prediction intervals (split-conformal) — replacing vibes.

What was wrong
--------------
``predictor.py`` shipped a "confidence" that never touched the model:
``vol < 2% → High/85, < 4% → Medium/70, else Low/55``. It measured the ASSET's
volatility, not the MODEL's error, so it was unfalsifiable — no observable
event could prove an "85" wrong, because 85 was never a probability of
anything. A confidence number is only honest if it has coverage semantics:
"the 80% interval should contain the realised value 80% of the time" is a
claim the future can check.

What this module does
---------------------
**Split-conformal intervals** on held-out residuals. Take the model's absolute
errors |y − ŷ| on a calibration set the model did not train on, and use their
finite-sample-corrected (1−α) quantile as the interval half-width:

    q̂_α = the ⌈(n+1)(1−α)⌉-th smallest of n calibration scores
    interval = ŷ ± q̂_α

Guarantee (Vovk et al.): if calibration and test errors are exchangeable,
coverage ≥ 1−α — with NO distributional assumption. That matters here because
daily returns are fat-tailed: Day 5's experiment measures Gaussian ±zσ bands
under-covering at 95% while conformal holds the nominal rate. Distribution-free
beats distribution-assumed exactly when the assumed distribution is wrong.

Multi-step horizons: calibration residuals are 1-step-ahead errors, so h-step
intervals scale the half-width by √h — an *assumption* (independent errors),
stated here because hiding assumptions is how the old confidence happened.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core: conformal + Gaussian half-widths from calibration residuals
# ─────────────────────────────────────────────────────────────────────────────
def conformal_halfwidth(residuals: np.ndarray, alpha: float = 0.2) -> float:
    """Finite-sample-corrected (1−α) quantile of |residuals|.

    With n calibration points the corrected rank is ⌈(n+1)(1−α)⌉; if that
    exceeds n (too few points for the requested confidence) the max score is
    used and true coverage may exceed nominal — conservative, never optimistic.
    """
    scores = np.abs(np.asarray(residuals, dtype=float).flatten())
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    if n == 0:
        raise ValueError("Need at least one calibration residual")
    rank = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    return float(np.sort(scores)[rank - 1])


def gaussian_halfwidth(residuals: np.ndarray, alpha: float = 0.2) -> float:
    """±z·σ half-width under a normality assumption — the comparison baseline.

    Kept so the Day-5 experiment can measure exactly what assuming normality
    costs on fat-tailed financial errors (it under-covers in the tail).
    """
    from scipy.stats import norm

    r = np.asarray(residuals, dtype=float).flatten()
    r = r[~np.isnan(r)]
    if len(r) < 2:
        raise ValueError("Need >= 2 residuals for a std estimate")
    return float(norm.ppf(1.0 - alpha / 2.0) * np.std(r, ddof=1))


def scale_for_horizon(halfwidth_1step: float, horizon: int) -> float:
    """√h scaling of a 1-step half-width (independent-errors assumption)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return float(halfwidth_1step * math.sqrt(horizon))


def empirical_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of realised values inside [lower, upper] — the honesty check."""
    a = np.asarray(actual, dtype=float).flatten()
    lo = np.asarray(lower, dtype=float).flatten()
    hi = np.asarray(upper, dtype=float).flatten()
    mask = ~np.isnan(a)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((a[mask] >= lo[mask]) & (a[mask] <= hi[mask])))


# ─────────────────────────────────────────────────────────────────────────────
# The predictor.py-facing bundle
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class IntervalForecast:
    """Point forecasts with conformal bands + a width-derived confidence label."""
    q80: float                    # 1-step 80% half-width (calibration units)
    q95: float                    # 1-step 95% half-width
    lower80: np.ndarray
    upper80: np.ndarray
    lower95: np.ndarray
    upper95: np.ndarray
    rel_width80: float            # q80 / reference price — drives the label
    confidence: str               # High / Medium / Low (kept for the template)
    confidence_score: int         # now = expected coverage of the NARROW band


def confidence_from_width(rel_width80: float) -> tuple[str, int]:
    """Map the 80% relative half-width to the legacy label + score fields.

    The label thresholds are documented, monotone, and tied to the MODEL's
    calibrated error (not the asset's volatility): a 1-step 80% band tighter
    than ±1.5% of price reads High, tighter than ±4% Medium, else Low. The
    score is no longer a made-up 85/70/55: it is 80 — the nominal coverage of
    the band being shown — degraded when the band is so wide it is useless
    (wider than ±8% of price at 80% confidence means "the model knows little").
    """
    if rel_width80 < 0.015:
        label = "High"
    elif rel_width80 < 0.04:
        label = "Medium"
    else:
        label = "Low"
    score = 80 if rel_width80 < 0.08 else max(50, int(round(80 - 400 * (rel_width80 - 0.08))))
    return label, score


def build_interval_forecast(
    residuals: np.ndarray,
    point_forecasts: np.ndarray,
    reference_price: float,
    horizons: np.ndarray | None = None,
) -> IntervalForecast:
    """Conformal bands around ``point_forecasts`` from holdout ``residuals``.

    ``residuals`` are 1-step-ahead errors (same units as the forecasts —
    predictor.py passes price-level test errors). ``horizons`` defaults to
    1..len(point_forecasts) for an autoregressive future path, giving √h-
    widening bands.
    """
    pf = np.asarray(point_forecasts, dtype=float).flatten()
    if horizons is None:
        horizons = np.arange(1, len(pf) + 1)
    h = np.asarray(horizons, dtype=float).flatten()
    if len(h) != len(pf):
        raise ValueError("horizons and point_forecasts must align")

    q80 = conformal_halfwidth(residuals, alpha=0.20)
    q95 = conformal_halfwidth(residuals, alpha=0.05)
    w80 = q80 * np.sqrt(h)
    w95 = q95 * np.sqrt(h)

    rel = q80 / float(reference_price) if reference_price else float("inf")
    label, score = confidence_from_width(rel)
    return IntervalForecast(
        q80=q80, q95=q95,
        lower80=pf - w80, upper80=pf + w80,
        lower95=pf - w95, upper95=pf + w95,
        rel_width80=rel, confidence=label, confidence_score=score,
    )
