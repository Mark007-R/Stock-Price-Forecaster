# Day 09 — Phase 7: Production wrapper (Docker · Redis · MLflow · ops dashboard)

**Date:** 2026-07-23 · **Day 09 of 10** · StockAI Production Upgrade

## Resume gap progress

**Gap:** "trained a model in a notebook" vs "shipped a measured ML service" — the repo had an
honest pipeline (Days 1–8) but no deployment story: no container, no shared cache, no run
tracking, no operational view of the walk-forward evidence.

**Today's contribution:** the honest pipeline is now production-shaped — a slim Docker image (no
TensorFlow; the serving champion is ARIMA) behind `docker compose up` with Redis as a shared
price cache, every walk-forward run recorded to MLflow with its buy-and-hold benchmark logged as
a first-class metric, per-request telemetry on the API, and a Streamlit ops dashboard whose every
chart goes through the same `src/` code path the API serves. Plus one real bug found by wiring it
together: the API 422'd every `xgboost_tuned` backtest since Day 6.

## Files touched

| File | Change |
|---|---|
| `src/data/loader.py` | +Redis read-through tier in front of the disk cache (`REDIS_URL`, lazy client, one-failure disable, `redis_status()`); promotes disk hits to Redis; TTL 7 d |
| `src/tracking/mlflow_runs.py` (new) | MLflow-tracked walk-forward runs: params, per-fold metrics as steps, strategy AND B&H aggregates, folds CSV + equity-curve PNG artifacts; CLI entry |
| `src/serving/api.py` | telemetry middleware → `logs/requests.jsonl`; `/health` reports Redis status; `/backtest` gains `log_mlflow`; **bugfix: `with_features` used `== "xgboost"`, breaking `xgboost_tuned` (lines ~245)**; version 0.9.0 |
| `src/serving/dashboard.py` (new) | Streamlit ops dashboard: leakage before/after, live walk-forward explorer (equity vs B&H, per-fold dir-acc), conformal coverage, MLflow run table, telemetry tail |
| `Dockerfile`, `.dockerignore`, `requirements-serving.txt` (new) | slim serving image — python:3.11-slim + libgomp, no TF/Prophet; healthcheck on `/health` |
| `docker-compose.yml` (new) | redis + api (:8000) + dashboard (:8501); `mlruns/`, `logs/`, `data/eval/` host-mounted |
| `experiments/day09_production.py` (new) | cache-tier latency bench + MLflow batch runner |
| `requirements.txt`, `.gitignore` | +redis/mlflow/fakeredis pins; ignore `logs/` |

## Setup

CPU only. Live Redis 7 (alpine container) on :6379 for the cache bench; MLflow 2.22 file store at
`mlruns/` (gitignored, host-mounted in compose); 10-ticker fixed set, 2021-01-01→2025-01-01,
expanding walk-forward, 5 folds, 5 bps/side — identical protocol to Days 2–8.

## Experiments

### A. What does each price-cache tier actually cost?

**Hypothesis:** Redis will NOT beat the local disk cache on latency; its value must be
justified by sharing, not speed.

| tier | reps | p50 ms | p95 ms | shared across containers |
|---|---|---|---|---|
| yfinance (cold fetch) | 1 | 2352.7 | 2352.7 | no |
| disk CSV cache | 20 | **4.9** | 6.4 | no |
| Redis GET (live server) | 20 | 5.4 | 7.7 | **yes** |

**Interpretation:** confirmed. Redis is ~10% *slower* than the local CSV for a single process
(both ~480× and ~435× faster than a cold yfinance pull). Redis earns its place only in the
compose stack, where the API and dashboard are separate containers with separate filesystems —
without it each would pay the 2.4 s yfinance hit per uncached series (and hit rate limits twice
as fast). The loader therefore treats Redis as strictly optional: unset `REDIS_URL` (or a dead
server) degrades silently to disk-then-network.

### B. MLflow-tracked walk-forward batch — 5 models × 10 tickers, 50 runs

Every run logs strategy **and** buy-and-hold metrics; the store cannot show one without the other.

| model | mean dir-acc | mean Sharpe (net) | mean B&H Sharpe | beats B&H | mean wall s |
|---|---|---|---|---|---|
| arima (champion) | 0.5345 | 1.3428 | 1.8302 | 1/10 | 11.8 |
| xgboost_tuned | 0.5313 | 1.1863 | 1.8302 | 2/10 | 1.4 |
| xgboost | 0.5164 | 1.2201 | 1.8302 | 1/10 | 1.7 |
| momentum | 0.5114 | 0.9041 | 1.8302 | 1/10 | 0.4 |
| persistence | 0.0000* | 0.0000 | 1.8302 | 0/10 | 1.1 |

*persistence predicts exactly 0 → never takes a directional position; dir-acc scores that as no
credit by design (Day 2 convention).

**Interpretation:** the tracked store reproduces the sprint's Day-3/Day-6 numbers exactly (ARIMA
1.34 vs 1.83; xgb_tuned dir-acc 0.531) — a regression check in itself. The 5 winning runs out of
50 are: momentum·XOM, arima·XOM, xgb_tuned·XOM, xgb_tuned·KO, xgboost·MSFT — **4 of 5 "wins"
land on XOM and KO, the two tickers where buy-and-hold itself went nowhere** (B&H Sharpe 0.17 /
0.19). Where the benchmark is strong, nothing beats it; the "edge" only appears where there was
nothing to beat.

### C. API production smoke — and the bug the wrapper caught

With Redis up: `/health` → `price_cache_redis: "connected"`; `/predict` AAPL h=5 serves
arima(1,0,0) with √h-widened conformal bands; `/backtest {log_mlflow: true}` returns
`mlflow_run_id` and the run appears in the store with identical metrics to the CLI path
(dir-acc 0.5481, Sharpe 1.160 vs B&H 1.960 — matches run 42dc4f79 from batch B).

**Found while smoke-testing:** `/backtest` with `xgboost_tuned` → 422
(`xgboost needs ctx built with make_context(..., with_features=True)`). Root cause: Day 5 wired
`with_features=(req.model == "xgboost")`; Day 6 registered `xgboost_tuned` and no test covered
the API path. Fixed to `startswith("xgboost")`. Telemetry captured the failure as it happened
(`{"path": "/backtest", "status": 422, "latency_ms": 31.5}`) — the middleware paid for itself
within minutes of existing. Day 10's `test_api.py` gets a regression case per registry model.

### D. Ops dashboard (Streamlit, :8501)

Rendered and verified live in a browser: §1 leakage before/after per ticker (median RMSE
inflation 3.6%, LSTM beats persistence on 0/10 after the fix); §2 walk-forward explorer —
ARIMA·AAPL equity vs B&H (+82.2% vs +100.2%, Sharpe 1.71 vs 1.96), per-fold dir-acc vs always-up
0.562; §3 conformal coverage table (Day 5); §4 MLflow table ("5 of 50 runs beat their benchmark")
+ telemetry tail with live p50/p95/error-rate. LSTM/PatchTST are deliberately excluded from the
live explorer (TF + minutes per fit is the wrong tool for a dashboard request; their story is in
the Day-2/7 CSVs).

## Head-to-Head Comparison (running leaderboard — unchanged by design)

Day 9 adds no new model. The leaderboard remains: B&H Sharpe 1.83 > ARIMA 1.34 > xgb_tuned 1.19
(dir-acc 0.531) > xgb 1.22 > momentum 0.90; always-up dir-acc 0.549 still unbeaten. Today's
delta is that this table is now *reproducible by anyone with `docker compose up` and queryable
via MLflow* rather than living only in CSVs.

## Key Findings

1. **Redis is not a speed layer here — it's a sharing layer.** p50 5.4 ms vs disk 4.9 ms
   (single process, measured), but it is the only tier that survives container boundaries. An
   ops decision justified by measurement, not fashion — and the loader degrades gracefully when
   it's absent.
2. **The tracked-run store is benchmark-first by construction.** Logging `bh_sharpe_net` and
   `beats_bh_sharpe` on every run makes the honest comparison the default MLflow view. First
   full sweep: 5/50 runs beat buy-and-hold, and 4 of those 5 are on the two tickers where
   buy-and-hold itself was flat — "edge" concentrated exactly where there was nothing to beat.
3. **Production wrapping found a real bug on day one:** every `xgboost_tuned` API backtest had
   been broken since Day 6 (exact-match `with_features` gate). Zero tests covered the API path —
   the strongest possible motivation for Day 10's test suite.
4. **What didn't work:** first `docker build` sat ~45 min with an empty layer cache (flaky
   network stalling the base-image pull) — killed and restarted with plain progress; host pip
   also needed a retry on a stalled wheel download. Slim image (no TF/Prophet) was the right
   call regardless: the serving champion is ARIMA, and TensorFlow would have tripled both image
   size and pull pain for models the API marks as research-only.

## Sample Outputs Saved

- `results/phase7_cache_bench.csv`, `results/phase7_mlflow_runs.csv`,
  `results/phase7_mlflow_summary.csv`
- `results/plots/day09_mlflow_sharpe.png` (champion vs B&H per ticker, from tracked runs)
- `results/samples/day09_mlflow_run_samples.json` (4 tracked-run summaries)
- `results/metrics.json` → `day09`
- `mlruns/` (51 runs, gitignored) · `logs/requests.jsonl` (telemetry, gitignored)

## Next Day

Day 10 — Phase 8: PROJECT COMPLETE. Test suite (`test_no_scaler_leakage.py`,
`test_temporal_split.py`, `test_walkforward_no_peeking.py`, `test_backtest_costs.py`,
`test_baselines.py`, `test_api.py` — including the xgboost_tuned regression), README rewritten
around the leakage-fix story + honest backtest table, 60-second demo, and the 30-day sprint arc
closer.

## Code Changes

- `src/data/loader.py` (+~90 lines): Redis tier + status; no behavior change without `REDIS_URL`.
- `src/tracking/mlflow_runs.py` (new, ~210 lines): tracked walk-forward runner, CLI.
- `src/serving/api.py` (+~55 lines): telemetry middleware, `log_mlflow`, health upgrade,
  xgboost_tuned bugfix.
- `src/serving/dashboard.py` (new, ~250 lines): four-section ops dashboard.
- `Dockerfile` / `docker-compose.yml` / `requirements-serving.txt` / `.dockerignore` (new).
- `experiments/day09_production.py` (new, ~190 lines): stages `cache` / `mlflow`.
