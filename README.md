# Stock-Price-Forecaster

**An LSTM stock-price app, audited and rebuilt around one question: is any of
it real?** The answer — measured, walk-forward, net of transaction costs — is
the honest result this repo now reports: *the best forecaster in it loses to
buy-and-hold, and every number here says so out loud.*

> 🔗 **Live demo:** https://iambatman07-stockai.hf.space · [HF Space](https://huggingface.co/spaces/IamBatman07/StockAI)
>
> ⚡ **60-second local demo:** `python demo.py` (offline, ~8s — talk-track in [docs/DEMO.md](docs/DEMO.md))

---

## The story

This project shipped as a Flask + Streamlit LSTM predictor with a good-looking
RMSE. A 10-day upgrade sprint (2026-07-15 → 07-24) treated it the way a quant
would treat a stranger's backtest:

1. **Day 1 — the audit found a leak.** `MinMaxScaler().fit_transform()` ran on
   the **full price series before the train/test split** — the test window's
   min/max leaked into training normalisation, understating reported error by
   up to 23% per ticker. The first commit of the sprint was the fix (fit on the
   train slice only, in both `predictor.py` and `stock_predictor.py`), and the
   honest post-fix numbers are the only ones reported anywhere in this repo.
   A regression test now fails if the leak ever returns.
2. **Days 2–4 — reframe and re-measure.** Predicting absolute prices flatters
   any model (persistence beat the LSTM's price-RMSE on 10/10 tickers), so the
   target became next-day *returns*, scored **walk-forward** (5 expanding
   folds, no peeking — enforced at runtime) against persistence and always-up
   baselines, with a **cost-aware long/flat backtest vs buy-and-hold**.
3. **Days 5–7 — production + depth.** `src/` layout, FastAPI service,
   conformal prediction intervals replacing a made-up "confidence" number,
   Optuna sweep, portfolio backtest, a PatchTST transformer.
4. **Day 8 — the frontier check.** A frontier LLM, asked for next-day
   direction on 200 anonymized samples: exactly 100/200. A coin flip, at 2s
   and ~2¢ per flip.
5. **Days 9–10 — ship it honestly.** Docker + MLflow + ops dashboard + 90
   offline tests + this README.

## Headline results

**Walk-forward, 10 tickers (AAPL MSFT SPY GOOGL AMZN META NVDA JPM XOM KO),
2021–2025, ~5,000 out-of-sample days, 5 bps/side costs:**

| Strategy | Sharpe (net) | Total return (mean) | Dir-acc | RMSE(ret) vs random walk |
|---|---|---|---|---|
| **Buy-and-hold** | **1.83** | **+178.7%** | — | — |
| always-up (bet up daily) | 1.83 | +178.7% | 0.549 | — |
| ARIMA (champion forecaster) | 1.34 | +118.1% | 0.534 | −0.02% |
| XGBoost (18 causal features) | 1.22 | +84.1% | 0.516 | +8.0% |
| LSTM on returns | 1.11 | +79.6% | 0.519 | +0.1% |
| momentum (r̂ₜ = rₜ₋₁) | 0.90 | +54.6% | 0.511 | +41% |
| persistence (r̂ = 0) | 0.00 | 0.0% | takes no position | floor |

**No model beats buy-and-hold.** That's the finding, and it survived an Optuna
sweep (the optimizer's global best *was* the random walk — it shrank predictions
5.6× toward zero), calendar/regime features, multi-horizon targets, a
transformer (PatchTST: RMSE 15.9% *worse* than predicting zero), and 25
portfolio configurations (23/25 lost to equal-weight buy-and-hold; the 2 wins
were on tickers where buy-and-hold went nowhere).

### The leakage fix, quantified (Day 1)

| | Leaky scaler (as shipped) | Honest scaler (fixed) |
|---|---|---|
| Scaler sees | full series incl. test min/max | train window only |
| AAPL price-RMSE | 5.29 | 6.50 (+23.1%) |
| NVDA price-RMSE | 6.28 | 7.69 (+22.4%) |
| Mean across 10 tickers | — | +5.2% reported-error inflation |
| LSTM beats persistence on price-RMSE | — | **0/10 tickers** |

### The ablation (5,000 pooled OOS days)

| Rung | Adds | Dir-acc | Sharpe (net) |
|---|---|---|---|
| 1 | persistence floor | — | 0.00 |
| 2 | + returns target (LSTM) | 0.517 | 1.03 |
| 3 | + 18 causal features (XGBoost) | 0.516 | 1.22 |
| 4 | + champion selection (ARIMA) | 0.535 | 1.34 |
| 5 | + Optuna tuning + time-decay | 0.531 | 1.19 |
| — | *always-up baseline* | *0.549* | *1.83* |

Six days of ML bought 1.8 points of directional accuracy — and never caught
the baseline that doesn't forecast at all.

### Frontier comparison (Day 8)

| System | Dir-acc (200 OOS samples) | Latency/pred | Cost/pred |
|---|---|---|---|
| Claude (zero-shot, anonymized panels) | 0.500 — CI95 [0.43, 0.57], p = 1.00 vs coin | ~2,000 ms | ~$0.02 |
| Best specialized (XGB tuned, same days) | 0.560 — CI overlaps the coin too | 16 ms | $0 |
| Full pipeline (5,000-day walk-forward) | 0.535 | 16 ms | $0 |

Its confidence was *anti-calibrated* (most-confident calls: 43.8%), and 200
samples can't statistically crown anyone — which is exactly why every claim
above runs on 5,000-day walk-forwards instead.

### Calibrated uncertainty (Day 5)

The volatility-only "confidence: 85" heuristic is gone. Intervals are
**split-conformal** on the model's own held-out errors: measured coverage
**80.8% / 99.2%** at nominal 80/95 (the Gaussian alternative missed the 95%
tail by 2×). This is what the app now ships: forecasts that admit what they
don't know, with a claim the future can check.

---

## Architecture

```
├── demo.py                     # the 60-second honest-evaluation demo (offline)
├── app.py / predictor.py       # original Flask UI (leakage-fixed, conformal bands)
├── stock_predictor.py          # original Streamlit variant (leakage-fixed)
├── historical.py               # technical indicators — the ONE implementation,
│                               #   imported by features + API alike
├── src/
│   ├── data/loader.py          # cached yfinance loader (disk + optional Redis)
│   ├── features/engineer.py    # 18+11 causal features + empirical no-look-ahead PROOF
│   ├── models/                 # persistence / arima / xgb / lstm / patchtst registry
│   │   └── intervals.py        # split-conformal prediction intervals
│   ├── backtest/
│   │   ├── walkforward.py      # folds + no-peeking invariants + return-space scoring
│   │   ├── trading.py          # long/flat with turnover costs vs buy-and-hold
│   │   └── portfolio.py        # multi-ticker portfolio engine (Day 7)
│   ├── serving/api.py          # FastAPI :8000 — /predict /backtest /indicators /correlation
│   ├── serving/dashboard.py    # Streamlit ops dashboard (the sprint's evidence, live)
│   └── tracking/mlflow_runs.py # benchmark-first MLflow logging
├── tests/                      # 90 offline tests — see below
├── experiments/                # day01..day09 runnable experiment scripts
├── results/                    # every metric CSV + plots + samples the reports cite
├── reports/                    # daily research reports (day01..day10)
└── docs/                       # TS_AUDIT, DATA_LEAKAGE, TARGET_REFRAMING, MODEL_CARD, DEMO
```

**Design stances**

- **The benchmark is load-bearing.** `/backtest` refuses zero-cost runs at the
  schema level and always returns buy-and-hold beside the strategy; the MLflow
  tracker cannot log a Sharpe without `bh_sharpe_net` next to it.
- **Honesty ships in the payload.** Every `/predict` and `/backtest` response
  carries a `disclaimer` field with the sprint's verdict.
- **Proof over promise.** Feature causality is not asserted in comments — it's
  *demonstrated* by rebuilding features on a truncated series and requiring
  bit-identical values (`assert_no_lookahead`), and pinned by tests.

## Tests (90, all offline, ~8s)

```bash
pytest tests -q
```

| File | Guards |
|---|---|
| `test_no_scaler_leakage.py` | AST-level regression: the scaler is fit on a train *slice*; `fit_transform` never returns |
| `test_temporal_split.py` | fold geometry: contiguous, non-overlapping, train strictly before test |
| `test_walkforward_no_peeking.py` | runtime invariants, oracle-scores-zero sanity, the feature truncation proof |
| `test_backtest_costs.py` | hand-computed equity curves; costs on turnover; higher costs strictly hurt |
| `test_baselines.py` | persistence floor arithmetic; momentum provably uses only past prices |
| `test_intervals.py` | conformal quantile by hand; small-n conservatism; coverage ≈ nominal on synthetic residuals |
| `test_api.py` | golden paths + guardrails (zero-cost refused, disclaimer present, Day-9 `xgboost_tuned` regression) |
| `test_loader.py` | cache-first, never-silently-empty, Redis optional |

## Run it

```bash
pip install -r requirements.txt

python demo.py                                # the 60-second story
pytest tests -q                               # the 90 guards
python app.py                                 # Flask UI on :5000
uvicorn src.serving.api:app --port 8000       # honest API on :8000
streamlit run src/serving/dashboard.py        # ops dashboard
docker compose up                             # API + Redis + MLflow + dashboard
```

Example API call:

```bash
curl -s -X POST localhost:8000/backtest -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "model": "arima", "cost_bps": 5}'
```

## The sprint, day by day

| Day | Phase | Report |
|---|---|---|
| 1 | Audit + scaler-leakage fix + honest baselines | [reports/day01](reports/day01_phase1_report.md) |
| 2 | Returns target + walk-forward harness | [reports/day02](reports/day02_phase2_report.md) |
| 3 | 5-family bake-off + cost-aware backtest | [reports/day03](reports/day03_phase2_report.md) |
| 4 | Calendar/regime features + multi-horizon (both hurt) | [reports/day04](reports/day04_phase2_report.md) |
| 5 | `src/` refactor + FastAPI + conformal intervals | [reports/day05](reports/day05_phase3_report.md) |
| 6 | Optuna sweep (optimum = the random walk) + failure modes | [reports/day06](reports/day06_phase4_report.md) |
| 7 | Portfolio backtest + PatchTST (negative result) | [reports/day07](reports/day07_phase5_report.md) |
| 8 | Frontier LLM comparison (coin flip) + ablation | [reports/day08](reports/day08_phase6_report.md) |
| 9 | Docker + Redis + MLflow + ops dashboard | [reports/day09](reports/day09_phase7_report.md) |
| 10 | Tests + docs + demo — **project complete** | [reports/day10](reports/day10_phase8_report.md) |

## Limitations

Survivorship-biased ticker universe (chosen in 2026, all survivors); US
large-cap daily bars only; long/flat execution model (no shorting, slippage
model is a flat bps charge); interval coverage assumes exchangeable errors.
Details in [docs/MODEL_CARD.md](docs/MODEL_CARD.md). **Nothing in this repo is
investment advice — the repo's own measurements argue against using it as any.**

## License

MIT — see [LICENSE](LICENSE).
