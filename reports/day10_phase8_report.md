# Day 10 of 10 — StockAI Production Upgrade
**Phase 8 — Tests + README + demo + PROJECT COMPLETE**
**Date:** 2026-07-24

---

## Resume gap progress

**Gap:** "has an ML project" → "has a *defensible* ML project." Ten days of
honest measurement existed but were guarded by nothing and explained by a
README that still described the pre-sprint Alpha Vantage app. Today closes the
project with a 90-test offline suite that pins every methodological claim
(leakage fix, no-peeking, cost enforcement, baseline arithmetic, calibrated
intervals, API guardrails), a README rewritten as a mini research report that
leads with the leakage-fix story and the honest backtest table, a model card
that states the limitations plainly, and a runnable 60-second `demo.py` that
tells the whole story offline in four acts. This is also the closing day of
the 30-day three-repo sprint.

## Files touched

| File | Change |
|---|---|
| `tests/conftest.py` | new — offline fixtures (seeded geometric random walks, no network anywhere) |
| `tests/test_no_scaler_leakage.py` | new — 8 tests: AST-level regression that the scaler is fit on a train slice and `fit_transform` never returns, + the leak mechanism demonstrated |
| `tests/test_temporal_split.py` | new — 12 tests: expanding/rolling fold geometry, boundaries, rejection cases |
| `tests/test_walkforward_no_peeking.py` | new — 20 tests: runtime invariants, oracle-scores-zero alignment sanity, return-space metric hand-cases, the feature truncation proof (base + extended) |
| `tests/test_backtest_costs.py` | new — 12 tests: hand-computed equity curves, turnover-based costs, cost monotonicity, buy-and-hold single-commission |
| `tests/test_baselines.py` | new — 5 tests: persistence floor arithmetic, momentum provably past-only (future mutation test) |
| `tests/test_intervals.py` | new — 15 tests: conformal quantile by hand, small-n conservatism, √h widening, coverage ≈ nominal on synthetic fat-tailed residuals |
| `tests/test_api.py` | new — 13 tests: golden paths + guardrails (zero-cost 422, disclaimer mandatory, Day-9 `xgboost_tuned` regression, matrix symmetry) |
| `tests/test_loader.py` | new — 5 tests: cache-first, uppercase normalisation, never-silently-empty, Redis disabled state |
| `demo.py` | new — 60-second four-act offline demo (bug → walk-forward → backtest → intervals), ~6s runtime |
| `docs/DEMO.md` | new — talk-track + drill-down table for interview Q&A |
| `docs/MODEL_CARD.md` | new — intended use, data, protocol, results, limitations |
| `README.md` | full rewrite — mini research report; stale Alpha Vantage content replaced |
| `requirements.txt` | + `pytest==8.3.4` |
| `.gitignore` | fixed `pytest_c_` typo → `.pytest_cache/` |

## Setup

- CPU only; venv Python 3.11.9. Entire test suite and demo run **offline**
  from seeded synthetic series and the `data/eval/` CSV cache — no yfinance
  call anywhere in `pytest` or `demo.py` (cache-first loader).

## Experiments

### Experiment 1 — the suite catches what it exists to catch

**Hypothesis:** each regression test fails against the pre-sprint behaviour,
not just passes against the current code.

**Method:** wrote the leakage tests as AST checks (a `fit_transform` CALL node
anywhere in `predictor.py`/`stock_predictor.py` fails; `scaler.fit` must take
a Subscript/slice argument), plus a mechanism test showing a train-only scaler
maps the last train price to exactly 1.0 while the leaky fit maps it below —
the observable signature of the bug. First run: **3 failures, all mine** — the
initial regex matched `fit_transform` inside the *comments documenting the old
bug*, and one float compared exactly. Moved to AST (comments are invisible to
the parser) and `pytest.approx`.

**Result table:**

| Suite | Tests | Time | Network |
|---|---|---|---|
| test_no_scaler_leakage | 8 | <0.1s | none |
| test_temporal_split | 12 | <0.1s | none |
| test_walkforward_no_peeking | 20 | ~2s | none |
| test_backtest_costs | 12 | <0.1s | none |
| test_baselines | 5 | <0.1s | none |
| test_intervals | 15 | <0.1s | none |
| test_api | 13 | ~4s | none (loader monkeypatched) |
| test_loader | 5 | ~2s | none (yfinance stubbed) |
| **Total** | **90** | **~8s** | **none** |

**Interpretation:** the false-positive-on-comments failure is itself the
argument for AST-level regression tests: the string "fit_transform" *should*
appear in the file forever (it documents the fix); only the *call* must never
come back.

### Experiment 2 — the demo tells the honest story, verified live

**Hypothesis:** the whole sprint narrative can run offline in under 10 seconds
of compute.

**Method:** `demo.py` — Act 1 refits leaky-vs-honest scalers on cached AAPL
(leaky max $257.38 vs honest $195.89 — the model "knew" the top a year early);
Act 2 runs ARIMA + random-walk through the real 5-fold walk-forward live
(RMSE 0.013511 vs 0.013527 — fourth decimal); Act 3 runs the cost-aware
backtest (ARIMA long/flat +82.2%, Sharpe 1.71 vs buy-and-hold +100.2%,
Sharpe 1.96 on this window); Act 4 prints a 5-day conformal-band forecast.

**Result:** total runtime 6.1s offline. The demo's per-ticker numbers differ
from the 10-ticker sprint means (AAPL-only window) — the script says so and
cites the pooled numbers beside them, because a demo that quietly cherry-picks
its window would be the exact sin the sprint was about.

**Interpretation:** every claim in the talk-track is printed by the terminal
as it is spoken; nothing is asserted from memory.

### Experiment 3 — the README as a measured artifact

**Method:** rewrote README.md from the results CSVs only (leaderboard,
ablation, frontier, leakage comparison, coverage) — every number in every
table traces to a file in `results/`. Kept the live HF Space link. Structure:
story → headline tables → architecture stances → test map → run instructions →
day-by-day report index → limitations.

**Result:** the old README's claims (Alpha Vantage everywhere, "predicts
future stock prices", lru_cache) matched the pre-sprint app and none of the
sprint's findings; nothing of it survived except the license and the live-demo
link.

## Head-to-Head Comparison (final leaderboard, unchanged from Day 9)

| Strategy | Sharpe net | Dir-acc | Verdict |
|---|---|---|---|
| Buy-and-hold | **1.83** | — | the benchmark that matters |
| ARIMA (champion) | 1.34 | 0.534 | best forecaster, still loses |
| XGBoost tuned | 1.19 | 0.531 | best XGB, below always-up 0.549 |
| LSTM (returns) | 1.11 | 0.519 | random-walk parity |
| PatchTST | — | — | RMSE 15.9% worse than zero |
| LLM zero-shot (200 samples) | — | 0.500 | coin flip, p = 1.00 |

## Key Findings

1. **Write regression tests against the bug's mechanism, not its spelling.**
   The regex version of the leakage test failed on the fix's own comments;
   the AST version pins the only thing that matters — no `fit_transform`
   *call*, scaler fit on a *slice*.
2. **An offline suite is a design constraint that pays twice.** Forcing every
   test off the network produced monkeypatched loaders and seeded series —
   which is also exactly what makes the suite runnable in an interview, in CI,
   and in 8 seconds.
3. **The oracle test is the cheapest alignment audit.** A model handed the
   true test returns must score RMSE 0 and dir-acc 1.0; any off-by-one in the
   scorer's day-alignment breaks it. It passed — but it is the test I'd most
   want failing loudly during a refactor.
4. **What didn't work:** the first leakage-test draft (regex on source text) —
   see Experiment 1; fixed by moving to the AST. Also `pytest`'s default
   rootdir picked up the repo `venv/` until tests were pointed at `tests/`
   explicitly in the README instructions.

## Sample Outputs Saved

- `demo.py` terminal output reproduced in full in this report's Experiment 2
  (the demo *is* the sample output; it regenerates itself on every run).
- Suite transcript: 90 passed in ~8s (offline).

## Phase wrap-up: What was finalized

**Final approach.** The project closes as an honest-evaluation reference
implementation: walk-forward + persistence/always-up/buy-and-hold baselines +
turnover-costed backtests + split-conformal uncertainty, wrapped in FastAPI/
Docker/MLflow (Day 9), guarded by 90 offline tests, explained by a README that
is generated from its own results files, demonstrated by a 6-second offline
demo, and bounded by a model card that lists survivorship bias and the absence
of edge as first-class facts.

**Final metrics.** Honest post-leakage numbers only: champion ARIMA Sharpe
1.34 net of 5 bps vs buy-and-hold 1.83 across ~5,000 OOS days; dir-acc 0.534
vs always-up 0.549; RMSE(returns) at random-walk parity (−0.02%); conformal
coverage 80.8%/99.2% at nominal 80/95; LLM frontier check 0.500 on 200
samples; ablation span 0.517 → 0.535, every rung below the no-forecast
baseline.

**What carries forward.** The reusable pieces are the harness, not the models:
`src/backtest/walkforward.py` (structural no-peeking), `assert_no_lookahead`
(the truncation proof generalizes to any feature pipeline),
`src/backtest/trading.py` (benchmark-mandatory backtesting), and the
benchmark-first MLflow pattern. Future-sprint candidates from the backlog:
point-in-time universe (kill the survivorship bias), regime detection (HMM)
feeding the model, realistic execution (vectorbt), paper-trading loop.

**Resume gap progress.** The StockAI line item is now: *"Audited an LSTM
stock predictor, found and fixed train/test leakage in its normalisation,
rebuilt evaluation as cost-aware walk-forward against persistence and
buy-and-hold baselines, replaced heuristic confidence with measured-coverage
conformal intervals, shipped it as a tested (90 tests), containerized,
MLflow-tracked service — and reported the honest negative result."* Every
clause is backed by a file in this repo.

## Next Day

None — **PROJECT COMPLETE**, and with it the 30-day sprint
(FinTrack Jun 25–Jul 4, CineSemantics Jul 5–14, StockAI Jul 15–24).

## Code Changes

Branch `sprint/day10-2026-07-24`: 9 new test files (90 tests), `demo.py`,
`docs/DEMO.md`, `docs/MODEL_CARD.md`, full `README.md` rewrite,
`requirements.txt` + `.gitignore` touch-ups.
