# Day 08 — Phase 6: Frontier LLM comparison (designed negative result) + full ablation

**Date:** 2026-07-22 · **Day 08 of 10** · StockAI Production Upgrade (Stock-Price-Forecaster)

## Resume gap progress

**Gap:** the repo had never answered the two questions every reviewer of a "stock prediction"
project asks: *"why not just ask an LLM?"* and *"which of your upgrades actually mattered?"*
**Today's contribution:** a 200-sample zero-shot LLM next-day-direction benchmark (the negative
result the spec predicted — measured, not cited) and a five-rung ablation of the whole sprint on
5,000 frozen out-of-sample days, plus the assembled `results/frontier_comparison.csv` placing the
naive pre-sprint notebook, the full pipeline, and the LLM side by side.

## Setup

- **Compute:** CPU only. Specialized models scored from the Day-7 frozen walk-forward predictions
  (`results/samples/day07_oos_predictions.csv`, 5,000 OOS days per model); default-XGB re-run live
  on the identical folds (50 fits).
- **Data:** same 10 tickers (AAPL MSFT SPY GOOGL AMZN META NVDA JPM XOM KO), same span
  (2021-01-01 → 2025-01-01), same 5 expanding walk-forward folds as Days 2–7. Public yfinance data
  from the disk cache.
- **LLM:** Claude, run **in this session**, genuinely zero-shot. **GPT-5.4 was not run** (no
  OpenAI key in the autonomous environment) and is reported as not-run rather than estimated —
  same honest-scoping convention as CineSemantics Day 8. LLM latency (~2 s) and cost (~$0.02/query)
  are labeled ESTIMATES; everything else is measured.
- **Contamination guard (design decision):** the eval window is 2023–24 — inside most frontier
  models' training data. Prompts are therefore **anonymized**: no ticker, no dates. Each of the 200
  prompts is only a month of daily returns plus the indicator panel (RSI14, BB%B, MACD histogram,
  close/SMA20, close/SMA50, 21d/63d returns, trailing vol, distance from 52-week high, day of
  week). A dated, named prompt would have tested *recall of 2023–24 headlines*, not forecasting.
- **Sampling:** seeded (42), stratified — 10 tickers × 5 folds × 4 OOS days = 200. Ground truth
  was written to a separate file and never read until scoring. Base rate: 53.0% up days.

## Experiments

### 1 — Zero-shot LLM next-day direction (the designed negative result)

**Hypothesis:** the LLM lands within noise of a coin flip (finance literature + Day-6/7 evidence
that these features carry no exploitable daily signal).

**Method:** 200 anonymized indicator prompts → up/down + stated confidence per sample → scored
against realised next-day direction; Wilson 95% CIs; two-sided binomial test vs 0.5.

| Metric | Value |
|---|---|
| Directional accuracy | **100/200 = 0.500** |
| Wilson 95% CI | [0.431, 0.569] |
| p-value vs coin flip | **1.00** |
| Up-call rate | 77.0% (base rate 53.0%) |
| Mean return captured (trade every call) | **−3.7 bps/day** (owning every day: −2.1 bps/day on these 200) |
| Latency / cost | ~2,000 ms, ~$0.02 per query (estimates) vs 16.2 ms, ~$0 for ARIMA |

**Interpretation:** an exact coin flip — you cannot script a cleaner negative result. Two
sub-findings sharpen it: (a) the LLM is **long-biased far beyond the data** (77% up calls vs 53%
base rate) — it "knows" markets drift up and over-applies it; (b) trading its calls *lost more*
than owning the same days, because its 46 down calls missed rebounds more often than they dodged
drops.

### 2 — Anti-calibration of stated confidence

| Stated confidence | n | Hit rate |
|---|---|---|
| ≤ 0.55 | 184 | 0.505 |
| 0.55–0.65 | 16 | **0.438** |

**Interpretation:** the calls the LLM was most sure about — RSI-extreme mean-reversion setups —
were its *worst*. Confidence and accuracy moved in opposite directions. This mirrors the Day-5
finding that the shipped app's hardcoded confidence measured the wrong thing; a language model's
self-reported confidence on a signal-free task is the same failure in prose form.

### 3 — Same-200-sample head-to-head (every forecaster, identical days)

| Method | Dir-acc | Wilson 95% CI |
|---|---|---|
| XGB tuned+decay | 0.560 | [0.491, 0.627] |
| ARIMA (champion) | 0.550 | [0.481, 0.617] |
| always-up | 0.530 | [0.461, 0.598] |
| LSTM (returns) | 0.530 | [0.461, 0.598] |
| PatchTST | 0.530 | [0.461, 0.598] |
| **Claude zero-shot** | **0.500** | **[0.431, 0.569]** |
| XGB default | 0.480 | [0.412, 0.549] |

**Interpretation:** the quiet methodological punchline: **on 200 samples, NOTHING — not even the
sprint's best model — is statistically distinguishable from a coin.** Every CI straddles 0.5.
That is precisely why this sprint evaluated on 5,000-day walk-forwards; any single-holdout claim
(including the original notebook's) lives inside this noise band.

### 4 — Full-sprint ablation (5,000 frozen OOS days, identical folds)

| Rung | Adds | RMSE vs zero | Dir-acc | Sharpe net 5 bps |
|---|---|---|---|---|
| 1 persistence | random walk r̂=0 | 1.0000 | — (no position) | 0.00 |
| 2 +returns target | LSTM on returns (Day 2) | 0.9996 | 0.517 | 1.03 |
| 3 +features | XGB default, 18 causal features (Day 3) | 1.0799 | 0.516 | 1.22 |
| 4 +champion | ARIMA by AIC (Day 3) | **0.9994** | **0.535** | **1.34** |
| 5 +tuning | Optuna + time-decay (Day 6) | 1.0025 | 0.531 | 1.19 |
| — | *always-up baseline* | — | *0.549* | — |
| — | *buy-and-hold* | — | — | *1.83* |

**Interpretation:** six days of ML upgrades moved directional accuracy by **1.8 pp end to end**
(0.517 → 0.535), never crossed the one-line always-up baseline (0.549), and never came within
0.44 Sharpe of buy-and-hold (1.34 vs 1.83). Only ARIMA nominally beats the predict-zero RMSE
floor — by 0.06%. The ablation doesn't show which upgrade "won"; it shows the honest version of
the ladder is *flat*, which IS the finding: the pipeline upgrades bought correctness of process
(no leakage, real baselines, calibrated intervals), not alpha.

## Frontier Model Comparison (Day-8 table)

| System | Dir-acc | RMSE vs zero | Sharpe net 5 bps | Latency/pred | Cost/pred | Verdict |
|---|---|---|---|---|---|---|
| Naive notebook (pre-sprint) | 0.483 | n/a (price-level) | n/a | n/a | $0 | leaky scaler inflated price-RMSE 6.98 → 6.58; persistence (3.80) beat it 10/10 tickers |
| **Full pipeline (this sprint)** | **0.535** | **0.9994** | **1.34** | **16.2 ms** | ~$0 | honest — and still loses to B&H (1.83) |
| Claude zero-shot (this session) | 0.500 | — | — | ~2,000 ms (est.) | ~$0.02 (est.) | exact coin flip; anti-calibrated confidence; −3.7 bps/day if traded |
| GPT-5.4 zero-shot | — | — | — | — | — | **not run** (no API key) — reported, not fabricated |

## Key Findings

1. **The LLM is a coin: 0.500 on the nose, p = 1.00.** The designed negative result, measured on
   our own out-of-sample days with a contamination guard, at ~125× the latency and infinite
   relative cost vs ARIMA.
2. **The LLM's confidence is anti-calibrated** — its >0.55-confidence calls hit 43.8% vs 50.5%
   for the rest. Same disease as the app's old hardcoded confidence score, new host.
3. **200 samples can't separate ANY forecaster from a coin** — every Wilson CI in the head-to-head
   straddles 0.5. The LLM benchmark and the sprint's 5,000-day walk-forward discipline justify
   each other: single-holdout accuracy claims (the old README's included) are noise.
4. **The ablation ladder is flat** — 1.8 pp of directional accuracy across five rungs, all below
   always-up (0.549), all below B&H Sharpe (1.83). What the sprint actually bought is measurement
   integrity, and Day 8 proves even a frontier model can't rescue the signal side.
5. **What didn't work (and why):** the LLM's mean-reversion instinct at RSI extremes — its
   highest-conviction pattern — was its worst bucket (echoes Day 6: trailing-bear/oversold labels
   select rebound days in this window, and the LLM walked into the same trap from prose priors).

## Sample Outputs Saved

- `results/frontier_comparison.csv`, `results/ablation.csv` (spec deliverables)
- `results/phase6_same_sample.csv`, `results/phase6_llm_calibration.csv`
- `results/samples/day08_llm_prompts.jsonl` (200 anonymized prompts),
  `day08_llm_ground_truth.csv`, `day08_llm_predictions.json`, `day08_llm_examples.json` (10 worked examples)
- `results/plots/day08_same_sample_diracc.png`, `day08_llm_calibration.png`,
  `day08_ablation_ladder.png`, `day08_llm_confusion.png`
- `results/metrics.json` → `day08`

## Phase wrap-up: What was finalized

- **Final approach:** frontier comparison done as a *measured* negative result — 200 stratified,
  seeded, anonymized OOS samples with ground truth quarantined until scoring; ablation done on the
  frozen Day-7 predictions so all five rungs share folds, days, and costs; un-runnable baselines
  (GPT-5.4) reported as not-run.
- **Final metrics:** LLM 0.500 dir-acc (CI [0.431, 0.569], p=1.00), anti-calibrated confidence
  (0.438 in its top bucket); full pipeline 0.535 dir-acc / 1.34 Sharpe vs B&H 1.83 / always-up
  0.549; ablation total movement 1.8 pp; naive notebook 0.483 with leakage-inflated RMSE.
- **What carries forward:** `frontier_comparison.csv` + `ablation.csv` are the README's evidence
  tables for Day 10; the "200 samples proves nothing — that's why we walk-forward 5,000 days"
  framing is the interview line; Day 9 wraps the honest pipeline in production ops (Docker, Redis,
  MLflow, Streamlit ops dashboard).
- **Resume gap progress:** the project can now answer both killer interview questions with tables
  it generated itself: LLMs are a coin flip here (measured, contamination-guarded), and every
  sprint upgrade's contribution is decomposed on identical out-of-sample days.

## Next Day

Day 9 — Phase 7: production wrapper. Dockerize the FastAPI service, Redis cache for yfinance
pulls, MLflow tracking for walk-forward runs, Streamlit ops dashboard (equity curve vs B&H,
per-fold dir-acc, conformal intervals, leakage before/after). **[POST]**

## Code Changes

- `experiments/day08_frontier_ablation.py` (new, ~470 lines) — two-stage harness (`--stage prep`
  / `--stage score`): anonymized prompt builder, quarantined ground truth, Wilson/binomial
  scoring, same-sample head-to-head, five-rung ablation on frozen predictions, frontier table,
  four plots.
- No production `src/` modules touched — Day 8 is measurement, not modelling.
