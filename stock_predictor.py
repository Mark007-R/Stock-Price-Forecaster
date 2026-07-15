import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Import the SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Streamlit Page Config
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# API Keys (Replace with your own)
ALPHA_VANTAGE_API_KEY = "UV0H8FN0FJWS5WVK"
NEWS_API_KEY = "bfad50dc845d4d4b8a5565533dd4e70e"

st.sidebar.title("🔍 Navigation")
selected_page = st.sidebar.selectbox("Select a Page", [
    "📈 Stock Predictor", "📉 Historical Analysis", "📊 Stock Correlation Dashboard"])

# Add custom CSS for stylish page
st.markdown("""
    <style>
        /* General Background and Container Styling */
        .css-1d391kg {
            background-color: #f4f4f4;  /* Light grey background */
            color: black;
        }
        .block-container {
            padding: 2rem;
        }

        /* Header and Title Styling */
        .stTitle {
            color: black;
            font-weight: bold;
        }
        .stSubheader {
            color: black;
        }

        /* Button Styling */
        .stButton>button {
            background-color:#f0f0f0;  /* Grey background */
            color: black;
            border-radius: 10px;
            font-weight: bold;
            padding: 0.5rem 1rem;
        }

        /* Metric Styling */
        .stMetric {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 1rem;
            color: black;
        }

        /* Sidebar Styling */
        .css-18e3th9 {
            background-color: #f4f4f4;
            color: black;
        }

        /* Links and Text Styling */
        a {
            color: #1f77b4;  /* Blue for links */
        }
        a:hover {
            text-decoration: underline;
        }

        /* Specific Element Styling */
        .css-1j2nfhk {
            background-color: #f4f4f4;
        }
    </style>
""", unsafe_allow_html=True)

# Helper function to fetch stock data using Alpha Vantage
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data_alpha(ticker, outputsize='full'):
    """Fetch daily stock data from Alpha Vantage"""
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            if "Note" in data:
                raise Exception("API rate limit reached. Please wait a minute and try again.")
            elif "Error Message" in data:
                raise Exception(f"Invalid ticker symbol: {ticker}")
            else:
                raise Exception(f"Unable to fetch data: {data.get('Information', 'Unknown error')}")
        
        # Convert to DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        return df
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

# Helper function to get current price
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_current_price_alpha(ticker):
    """Get current stock price from Alpha Vantage"""
    try:
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            return float(data["Global Quote"]["05. price"])
        return None
    except Exception as e:
        return None

# Helper function to fetch intraday data for different intervals
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_intraday_data_alpha(ticker, interval='60min', outputsize='full'):
    """Fetch intraday stock data from Alpha Vantage"""
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url, timeout=10)
        data = response.json()
        
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            raise Exception("Unable to fetch intraday data")
        
        time_series = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        return df
    except Exception as e:
        raise Exception(f"Error fetching intraday data: {str(e)}")

# ----------------------- Page 1: Stock Predictor -----------------------
if selected_page == "📈 Stock Predictor":

    # Sidebar - Stock News
    st.sidebar.title("📰 Live Stock News")
    news_url = f"https://newsapi.org/v2/everything?q=stocks&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(news_url, timeout=10)
        news_data = response.json()
        analyzer = SentimentIntensityAnalyzer()

        if "articles" in news_data:
            for article in news_data["articles"][:5]:
                sentiment = analyzer.polarity_scores(
                    article["title"])["compound"]
                emoji = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😞"
                st.sidebar.markdown(
                    f"**{emoji} [{article['title']}]({article['url']})**")

                # Display image if available
                if article.get("urlToImage"):
                    st.sidebar.image(
                        article["urlToImage"], use_container_width=True)

                st.sidebar.caption(
                    f"🗞 {article['source']['name']} | 🕒 {article['publishedAt'][:10]}")
    except Exception as e:
        st.sidebar.error("⚠️ Unable to fetch live news!")


    # Main App Title
    st.title("📈 Stock Price Predictor")

    # User Inputs
    ticker = st.text_input(
        "Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL").upper()
    num_days = st.slider("Days for Prediction:", min_value=1,
                         max_value=100, value=30)

    # Fetch Live Stock Price
    if ticker:
        current_price = get_current_price_alpha(ticker)
        if current_price:
            st.metric(label="💰 Current Price", value=f"${current_price:.2f}")
        else:
            st.info("💡 Enter a valid ticker and click 'Predict' to see current price")

    # Predict Button
    if st.button("🚀 Predict Stock Price"):
        with st.spinner("🔄 Fetching data and training model..."):
            try:
                # Fetch stock data from Alpha Vantage
                data = fetch_stock_data_alpha(ticker, outputsize='full')
                
                # Filter to last 2 years for faster processing
                two_years_ago = datetime.today() - timedelta(days=365 * 2)
                data = data[data.index >= two_years_ago]
                
                # Display the actual current price from fetched data
                latest_price = data["Close"].iloc[-1]
                st.success(f"✅ Latest available price: ${latest_price:.2f} (as of {data.index[-1].strftime('%Y-%m-%d')})")

                # Check if we have enough data
                if len(data) < 100:
                    st.error("⚠️ Not enough historical data for this ticker. Please try another stock.")
                    st.stop()

                # Normalize Data — fit the scaler on the TRAIN slice ONLY.
                # Previously: scaler.fit_transform(data[['Close']]) ran on the
                # FULL series before the split below, leaking the test set's
                # min/max into training and inflating the reported metrics.
                # Fix: pick the temporal split on the raw prices, fit on train
                # prices only, then transform the whole series with that scaler.
                raw_split = int(len(data) * 0.8)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(data[['Close']].iloc[:raw_split])   # train-only fit — no peeking
                scaled_data = scaler.transform(data[['Close']])

                # Prepare Data for LSTM
                def create_dataset(data, time_step=60):
                    X, y = [], []
                    for i in range(len(data) - time_step - 1):
                        X.append(data[i:(i + time_step), 0])
                        y.append(data[i + time_step, 0])
                    return np.array(X), np.array(y)

                time_step = 60
                X, y = create_dataset(scaled_data, time_step)
                X = X.reshape(X.shape[0], X.shape[1], 1)

                # Train-Test Split (temporal — matches the scaler split above)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                # Build LSTM Model
                model = Sequential([
                    LSTM(50, return_sequences=True,
                         input_shape=(X_train.shape[1], 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

                # Predictions
                predictions = model.predict(X_test, verbose=0)
                predictions = scaler.inverse_transform(predictions)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Predict Future Stock Prices
                last_data = scaled_data[-time_step:].reshape(1, time_step, 1)
                future_predictions = []
                for _ in range(num_days):
                    next_pred = model.predict(last_data, verbose=0)
                    future_predictions.append(next_pred[0][0])
                    last_data = np.append(last_data[:, 1:, :], [
                                          [next_pred[0]]], axis=1)

                future_predictions = scaler.inverse_transform(
                    np.array(future_predictions).reshape(-1, 1))

                # 📊 Prediction Results
                results_df = pd.DataFrame({
                    "Date": data.index[-len(y_test_actual):].strftime('%Y-%m-%d'),
                    "Actual Price": y_test_actual.flatten(),
                    "Predicted Price": predictions.flatten()
                })
                st.subheader("📊 Prediction Results")
                st.dataframe(results_df.style.format(
                    {"Actual Price": "${:,.2f}", "Predicted Price": "${:,.2f}"}))

                # Download CSV
                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(label="📥 Download CSV", data=csv_data,
                                   file_name=f"{ticker}_predictions.csv", mime="text/csv")

                # 📉 Prediction Graph
                st.subheader(f"📉 {ticker} Stock Price Prediction")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index[-len(y_test_actual):], y=y_test_actual.flatten(),
                                         mode="lines", name="Actual Price", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=data.index[-len(y_test_actual):], y=predictions.flatten(),
                                         mode="lines", name="Predicted Price", line=dict(color="red")))
                st.plotly_chart(fig, use_container_width=True)

                # Future Prediction Table & Graph
                future_dates = [data.index[-1] +
                                timedelta(days=i) for i in range(1, num_days + 1)]
                future_df = pd.DataFrame(
                    {"Date": future_dates, "Predicted Price": future_predictions.flatten()})

                st.subheader(f"🔮 {num_days}-Day Future Prediction")
                st.dataframe(future_df.style.format(
                    {"Predicted Price": "${:,.2f}"}))

                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted Price"],
                                                mode="lines+markers", name="Future Prediction", line=dict(color="green")))
                st.plotly_chart(fig_future, use_container_width=True)

                # 📊 Candlestick Chart with Indicators
                st.subheader("📊 Candlestick Chart with Indicators")
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index,
                    open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                    increasing_line_color='green', decreasing_line_color='red'
                )])

                # Simple Moving Average (SMA)
                data["SMA50"] = data["Close"].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["SMA50"], mode="lines", name="SMA 50", line=dict(color="orange")))

                # Exponential Moving Average (EMA)
                data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["EMA20"], mode="lines", name="EMA 20", line=dict(color="purple")))

                # Bollinger Bands
                rolling_mean = data["Close"].rolling(window=20).mean()
                rolling_std = data["Close"].rolling(window=20).std()
                data["Upper Band"] = rolling_mean + (rolling_std * 2)
                data["Lower Band"] = rolling_mean - (rolling_std * 2)
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["Upper Band"], mode="lines", name="Upper Bollinger Band", line=dict(color="gray", dash="dash")))
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["Lower Band"], mode="lines", name="Lower Bollinger Band", line=dict(color="gray", dash="dash")))

                fig.update_layout(xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
                st.info("💡 Tip: Make sure you're using a valid stock ticker symbol. Alpha Vantage free tier allows 25 requests/day.")

# ----------------------- Page 2: Historical Analysis -----------------------
elif selected_page == "📉 Historical Analysis":
    st.title("📉 Historical Stock Analysis")
    hist_ticker = st.text_input("Enter Stock Ticker:", "MSFT").upper()
    hist_period = st.selectbox(
        "Select Period", ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "Full History"])

    if st.button("📊 Analyze"):
        try:
            with st.spinner("Fetching historical data..."):
                # Fetch full data from Alpha Vantage
                df = fetch_stock_data_alpha(hist_ticker, outputsize='full')
                
                # Filter based on period
                today = datetime.today()
                if hist_period == "1 Month":
                    start_date = today - timedelta(days=30)
                elif hist_period == "3 Months":
                    start_date = today - timedelta(days=90)
                elif hist_period == "6 Months":
                    start_date = today - timedelta(days=180)
                elif hist_period == "1 Year":
                    start_date = today - timedelta(days=365)
                elif hist_period == "2 Years":
                    start_date = today - timedelta(days=730)
                else:  # Full History
                    start_date = df.index.min()
                
                df = df[df.index >= start_date]
                
                if df.empty:
                    st.error("⚠️ No data found for this ticker.")
                    st.stop()
                
                st.subheader(f"Data for {hist_ticker}")
                st.dataframe(df)

                st.subheader("📈 Closing Price Trend")
                st.line_chart(df["Close"])

                # Moving Averages
                df["SMA20"] = df["Close"].rolling(window=20).mean()
                df["SMA50"] = df["Close"].rolling(window=50).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color="orange")))
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="red")))
                fig.update_layout(
                    title=f"{hist_ticker} - SMA Analysis", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume Analysis
                st.subheader("📊 Volume Analysis")
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
                fig_volume.update_layout(title=f"{hist_ticker} - Trading Volume", xaxis_title="Date", yaxis_title="Volume")
                st.plotly_chart(fig_volume, use_container_width=True)
                
        except Exception as e:
            st.error(f"⚠️ Failed to retrieve historical data: {str(e)}")
            st.info("💡 Alpha Vantage free tier allows 25 requests per day. Try again later if limit reached.")

# ----------------------- Page 3: Stock Correlation Dashboard -----------------------
elif selected_page == "📊 Stock Correlation Dashboard":
    st.title("📊 Stock Correlation Dashboard")

    # User Inputs
    tickers_input = st.text_input(
        "Enter Stock Tickers (comma separated, e.g., AAPL, MSFT, TSLA):", "AAPL, MSFT").upper()
    tickers = [ticker.strip() for ticker in tickers_input.split(",")]

    # Fetch Data
    if len(tickers) > 1:
        if st.button("📊 Analyze Correlation"):
            try:
                with st.spinner("Fetching data and calculating correlations..."):
                    close_data = pd.DataFrame()
                    
                    # Fetch data for each ticker
                    for ticker in tickers:
                        try:
                            df = fetch_stock_data_alpha(ticker, outputsize='full')
                            # Filter to last 2 years
                            two_years_ago = datetime.today() - timedelta(days=730)
                            df = df[df.index >= two_years_ago]
                            close_data[ticker] = df['Close']
                            time.sleep(12)  # Alpha Vantage rate limit: 5 calls/min for free tier
                        except Exception as e:
                            st.warning(f"⚠️ Could not fetch data for {ticker}: {str(e)}")
                            continue
                    
                    if close_data.empty or len(close_data.columns) < 2:
                        st.error("⚠️ Not enough valid tickers to calculate correlation.")
                        st.stop()
                    
                    # Drop NaN values for correlation calculation
                    close_data = close_data.dropna()

                    # Calculate Correlation Matrix
                    corr_matrix = close_data.corr()

                    # Plot Correlation Heatmap
                    st.subheader("📊 Correlation Heatmap")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
                                linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation'}, 
                                vmin=-1, vmax=1, center=0)
                    plt.title("Stock Price Correlation Matrix")
                    st.pyplot(fig)
                    
                    # Display correlation values
                    st.subheader("📈 Correlation Values")
                    st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1))
                    
                    # Normalized Price Comparison
                    st.subheader("📉 Normalized Price Comparison")
                    normalized_data = close_data / close_data.iloc[0] * 100
                    fig_norm = go.Figure()
                    for ticker in close_data.columns:
                        fig_norm.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data[ticker], 
                                                      mode="lines", name=ticker))
                    fig_norm.update_layout(title="Normalized Stock Prices (Base 100)", 
                                          xaxis_title="Date", yaxis_title="Normalized Price")
                    st.plotly_chart(fig_norm, use_container_width=True)
                    
                    st.info("⏱️ Note: Alpha Vantage free tier has a rate limit of 25 requests/day and 5 requests/minute. Delays are added to respect this limit.")

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter more than one stock ticker.")