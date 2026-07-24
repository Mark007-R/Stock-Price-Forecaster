# The 60-second demo — talk-track

Run it (offline, ~8s of compute inside a ~60s narration):

```bash
python demo.py
```

Four acts, one sentence of setup each. The script prints everything the
narration references — nothing is claimed that the terminal isn't showing.

---

**Setup (5s).** "This repo started as an LSTM stock predictor with a great
RMSE. The sprint's job was to find out whether any of it was real."

**Act 1 — the bug (15s).** "First thing the audit found: the scaler was fit on
the *whole* price series before the train/test split. Watch — the leaky scaler's
max is $257, the honest one's is $195. The model was told where prices would
top out a year in advance. Fixing that was commit one; reported error had been
understated by up to 23% per ticker."

**Act 2 — walk-forward (15s).** "Second: one 80/20 holdout is one lucky draw.
Everything is now scored walk-forward — refit on the past, score on the next
block, slide. Here's the punchline: the champion model's return-RMSE and the
random walk's differ in the fourth decimal. And the 'always-up' baseline
matches its directional accuracy."

**Act 3 — the backtest (15s).** "Third, the only question that matters: after
5 basis points of costs, the champion's Sharpe loses to buying and holding.
That held across all ten tickers, five model families, an Optuna sweep, and a
transformer. The honest result *is* the result."

**Act 4 — the interval (10s).** "So what ships? Forecasts that admit what they
don't know: 80% conformal bands from the model's own calibrated errors —
measured coverage 80.8% on nominal 80. The old code shipped a 'confidence: 85'
that was never a probability of anything."

**Close (5s).** "Ninety tests keep it honest — including one that fails if
anyone ever fits a scaler on the full series again."

---

## If they want to go deeper

| Question | Show |
|---|---|
| "How do I know the folds don't peek?" | `pytest tests/test_walkforward_no_peeking.py -v` — includes the truncation proof: features rebuilt on a shortened series must be bit-identical |
| "What did tuning buy?" | `results/ablation.csv` — 5 rungs, dir-acc 0.517 → 0.535, all below always-up 0.549 |
| "Would an LLM do better?" | `results/frontier_comparison.csv` — 200 anonymized samples, dir-acc 0.500, p = 1.00 vs a coin |
| "Is it deployable anyway?" | `docker compose up` → FastAPI :8000 (every response carries the disclaimer field) + MLflow + ops dashboard |
| "The live API?" | `uvicorn src.serving.api:app --port 8000`, then POST `/backtest` — zero-cost backtests are refused by the schema |
