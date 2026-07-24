from flask import Blueprint, render_template, request
import pandas as pd
from datetime import datetime
import yfinance as yf
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

correlation_bp = Blueprint('correlation', __name__)

# In-memory cache
_cache = {}
CACHE_EXPIRY = 3600  # 1 hour

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data using yfinance with caching"""
    import time
    cache_key = f"{ticker}_{start_date}_{end_date}"
    current_time = time.time()

    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if current_time - cached_time < CACHE_EXPIRY:
            logger.info(f"Returning cached data for {ticker}")
            return cached_data.copy()

    logger.info(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d",
                     progress=False, auto_adjust=True)

    if df.empty:
        raise Exception(f"No data found for ticker '{ticker}'. Please verify the symbol is correct.")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    _cache[cache_key] = (df, current_time)
    logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
    return df

@correlation_bp.route('/correlation', methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        tickers_input = request.form.get("tickers", "").upper()
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

        if len(tickers) < 2:
            return render_template("correlation.html",
                error="Please enter at least 2 stock tickers separated by commas")

        if len(tickers) > 10:
            return render_template("correlation.html",
                error="Please enter no more than 10 tickers")

        for ticker in tickers:
            if len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
                return render_template("correlation.html",
                    error=f"Invalid ticker format: {ticker}")

        start_date = request.form.get("start_date", "2021-01-01")
        end_date = request.form.get("end_date", datetime.today().strftime('%Y-%m-%d'))

        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            if start_dt >= end_dt:
                return render_template("correlation.html",
                    error="Start date must be before end date")
            if (end_dt - start_dt).days < 30:
                return render_template("correlation.html",
                    error="Date range must be at least 30 days for meaningful correlation analysis")
        except ValueError:
            return render_template("correlation.html", error="Invalid date format")

        try:
            logger.info(f"Starting correlation analysis for {tickers}")
            close_data = pd.DataFrame()
            failed_tickers = []

            for ticker in tickers:
                try:
                    df = fetch_stock_data(ticker, start_date, end_date)
                    if not df.empty and 'Close' in df.columns:
                        close_data[ticker] = df['Close']
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    logger.error(f"Failed to fetch {ticker}: {str(e)}")
                    failed_tickers.append(ticker)

            if close_data.empty or len(close_data.columns) < 2:
                error_msg = "Not enough valid data found for correlation analysis."
                if failed_tickers:
                    error_msg += f" Failed tickers: {', '.join(failed_tickers)}"
                return render_template("correlation.html", error=error_msg)

            close_data = close_data.dropna(how="all")
            valid_columns = [
                column
                for column in close_data.columns
                if close_data[column].dropna().shape[0] >= 2
            ]
            close_data = close_data[valid_columns]

            if close_data.empty or len(close_data.columns) < 2:
                return render_template("correlation.html",
                    error="Not enough overlapping data found for the selected tickers and date range")

            corr_matrix = close_data.corr(min_periods=2)
            while len(corr_matrix.columns) >= 2 and corr_matrix.isna().to_numpy().any():
                missing_counts = corr_matrix.isna().sum()
                drop_column = missing_counts.sort_values(ascending=False).index[0]
                corr_matrix = corr_matrix.drop(index=drop_column, columns=drop_column)

            if corr_matrix.empty or len(corr_matrix.columns) < 2:
                return render_template("correlation.html",
                    error="Not enough overlapping data found for the selected tickers and date range")

            close_data = close_data[corr_matrix.columns]

            stats = {}
            for ticker in close_data.columns:
                ticker_data = close_data[ticker].dropna()
                if len(ticker_data) > 0:
                    returns = ticker_data.pct_change().dropna()
                    stats[ticker] = {
                        "current_price": round(float(ticker_data.iloc[-1]), 2),
                        "price_change": round(((float(ticker_data.iloc[-1]) / float(ticker_data.iloc[0])) - 1) * 100, 2),
                        "volatility": round(float(returns.std()) * 100, 2),
                        "mean_price": round(float(ticker_data.mean()), 2),
                        "sharpe_ratio": round(
                            (float(returns.mean()) / float(returns.std())) * (252 ** 0.5), 2
                        ) if returns.std() > 0 else 0
                    }

            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    t1 = corr_matrix.columns[i]
                    t2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    if pd.isna(correlation):
                        continue
                    corr_pairs.append({
                        'ticker1': t1,
                        'ticker2': t2,
                        'correlation': round(float(correlation), 3)
                    })

            corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

            warning = None
            if failed_tickers:
                warning = f"Warning: Could not fetch data for: {', '.join(failed_tickers)}"

            logger.info(f"Correlation analysis completed with {len(close_data)} data points")

            return render_template("correlation.html",
                tickers=list(close_data.columns),
                corr_matrix=corr_matrix.round(3).values.tolist(),
                labels=corr_matrix.columns.tolist(),
                stats=stats,
                corr_pairs=corr_pairs[:5],
                start_date=start_date,
                end_date=end_date,
                warning=warning,
                data_points=len(close_data)
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Correlation analysis error: {error_msg}")
            return render_template("correlation.html", error=f"Error: {error_msg}")

    return render_template("correlation.html")
