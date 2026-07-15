import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Blueprint, render_template, request, jsonify
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import warnings
import logging
import time
import threading
import uuid

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predict_bp = Blueprint('predict', __name__)

_cache = {}
CACHE_EXPIRY = 3600

# Job store: job_id -> { "status": "running"|"done"|"error", "result": {}, "error": "" }
_jobs = {}


def fetch_stock_data(ticker, period="1y"):
    cache_key = f"{ticker}_{period}"
    current_time = time.time()
    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if current_time - cached_time < CACHE_EXPIRY:
            logger.info(f"Returning cached data for {ticker}")
            return cached_data.copy()
    logger.info(f"Fetching data for {ticker} from yfinance...")
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise Exception(f"No data found for ticker '{ticker}'. Please verify the symbol is correct.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for col in df.columns:
        df[col] = df[col].astype(float)
    _cache[cache_key] = (df, current_time)
    logger.info(f"Fetched {len(df)} days of data for {ticker}")
    return df


def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def run_prediction(job_id, ticker, num_days):
    """Runs in a background thread. Stores result in _jobs when done."""
    try:
        data = fetch_stock_data(ticker, period="1y")

        if len(data) < 60:
            raise Exception(f"Insufficient data for {ticker}: {len(data)} days found, need 60+.")

        data = data.dropna()
        close_prices = data["Close"].values.astype(float).reshape(-1, 1)

        time_step = 30

        # ── Fit the scaler on the TRAIN slice ONLY (no look-ahead leakage) ──
        # Previously: scaler.fit_transform(close_prices) ran on the FULL series
        # BEFORE the train/test split, leaking the test set's min/max into the
        # training normalisation and inflating the reported RMSE/MAE/MAPE.
        # Fix: choose the temporal split on the raw price series, fit the scaler
        # on the training prices only, then transform the whole series with it.
        raw_split = int(len(close_prices) * 0.8)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(close_prices[:raw_split])          # train-only fit — no peeking
        scaled_data = scaler.transform(close_prices)

        def create_dataset(arr, ts):
            X, y = [], []
            for i in range(len(arr) - ts - 1):
                X.append(arr[i:(i + ts), 0])
                y.append(arr[i + ts, 0])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        X, y = create_dataset(scaled_data, time_step)

        if len(X) < 30:
            raise Exception(f"Not enough data to train. Found {len(data)} days.")

        X = X.reshape(X.shape[0], X.shape[1], 1)
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]

        logger.info(f"[{job_id}] Training on {len(X_train)} samples...")
        model = create_lstm_model((time_step, 1))
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=16,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stop]
        )
        logger.info(f"[{job_id}] Training complete.")

        predictions   = scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = float(np.sqrt(np.mean((predictions - y_test_actual) ** 2)))
        mae  = float(np.mean(np.abs(predictions - y_test_actual)))
        mape = float(np.mean(np.abs((y_test_actual - predictions) / (y_test_actual + 1e-8))) * 100)

        # Future predictions
        last_seq = scaled_data[-time_step:].reshape(1, time_step, 1).astype(np.float32)
        future_scaled = []
        for _ in range(num_days):
            next_val = float(model.predict(last_seq, verbose=0)[0][0])
            future_scaled.append(next_val)
            last_seq = np.append(last_seq[:, 1:, :], [[[next_val]]], axis=1).astype(np.float32)

        future_preds_inv = scaler.inverse_transform(
            np.array(future_scaled, dtype=np.float32).reshape(-1, 1)
        )
        future_dates = [data.index[-1] + timedelta(days=i + 1) for i in range(num_days)]

        current_price = float(data["Close"].iloc[-1])
        volatility    = float(data["Close"].pct_change().std() * 100)

        if volatility < 2:
            confidence, confidence_score = "High", 85
        elif volatility < 4:
            confidence, confidence_score = "Medium", 70
        else:
            confidence, confidence_score = "Low", 55

        # Build plain Python lists
        actual_list       = [round(float(v), 2) for v in y_test_actual.flatten()]
        predicted_list    = [round(float(v), 2) for v in predictions.flatten()]
        dates_list        = data.index[-len(y_test_actual):].strftime('%Y-%m-%d').tolist()
        future_preds_list = [round(float(v), 2) for v in future_preds_inv.flatten()]
        future_dates_list = [d.strftime('%Y-%m-%d') for d in future_dates]

        # Pre-zip for Jinja2 (Jinja2 doesn't have zip built-in)
        actual_predicted_zip = list(zip(dates_list, actual_list, predicted_list))
        future_zip           = list(zip(future_dates_list, future_preds_list))

        _jobs[job_id] = {
            "status": "done",
            "result": {
                "ticker":                ticker,
                "current_price":         round(current_price, 2),
                "last_date":             data.index[-1].strftime('%Y-%m-%d'),
                "actual":                actual_list,
                "predicted":             predicted_list,
                "dates":                 dates_list,
                "future_preds":          future_preds_list,
                "future_dates":          future_dates_list,
                "actual_predicted_zip":  actual_predicted_zip,
                "future_zip":            future_zip,
                "rmse":                  round(rmse, 2),
                "mae":                   round(mae, 2),
                "mape":                  round(mape, 2),
                "confidence":            confidence,
                "confidence_score":      confidence_score,
                "data_points":           len(data),
                "volatility":            round(volatility, 2),
            }
        }
        logger.info(f"[{job_id}] Job complete.")

    except Exception as e:
        logger.error(f"[{job_id}] Failed: {str(e)}", exc_info=True)
        _jobs[job_id] = {"status": "error", "error": str(e)}


# ── Routes ─────────────────────────────────────────────────────────────────────

@predict_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("home.html")

    ticker = request.form.get("ticker", "AAPL").upper().strip()
    if not ticker or len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
        return render_template("home.html",
            error="Please enter a valid stock ticker (e.g., AAPL, MSFT, GOOGL)")

    try:
        num_days = int(request.form.get("num_days", 30))
        if num_days < 1 or num_days > 365:
            return render_template("home.html", error="Number of days must be between 1 and 365")
    except ValueError:
        return render_template("home.html", error="Invalid number of days")

    # Kick off background job, return loading page immediately
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running"}
    thread = threading.Thread(target=run_prediction, args=(job_id, ticker, num_days), daemon=True)
    thread.start()

    return render_template("loading.html", job_id=job_id, ticker=ticker)


@predict_bp.route('/predict/status/<job_id>')
def predict_status(job_id):
    """Polled every 3s by loading.html"""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "Job not found"}), 404
    if job["status"] == "error":
        return jsonify({"status": "error", "error": job["error"]})
    return jsonify({"status": job["status"]})


@predict_bp.route('/predict/result/<job_id>')
def predict_result(job_id):
    """Called by loading.html when status == done"""
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return render_template("home.html", error="Result not ready or job not found.")
    result = job["result"]
    _jobs.pop(job_id, None)  # free memory
    return render_template("home.html", **result)