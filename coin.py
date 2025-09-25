import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Forex Trade Prediction", layout="wide")
st.title("🔮 Forex Trade Prediction")
st.write("Predict future cryptocurrency prices")

# ---------------- Technical Indicators ---------------- #
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi.index = series.index
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def add_indicators(df):
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['MACD_signal'] = compute_macd(df['Close'])
    return df.dropna()

# ---------------- Coin Selection ---------------- #
coin_map = {
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'DOGE': 'DOGE-USD', 'PEPE': 'PEPE-USD',
    'XRP': 'XRP-USD', 'BNB': 'BNB-USD', 'ADA': 'ADA-USD', 'SOL': 'SOL-USD'
}

symbol = st.selectbox("Select a coin:", list(coin_map.keys()))
ticker = coin_map[symbol]

# ---------------- Load Data ---------------- #
@st.cache_data(show_spinner=False)
def load_data(ticker, symbol):
    file_path = f"data/{symbol}_data.csv"
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(file_path):
        df = yf.download(ticker, start="2018-01-01")
        df.to_csv(file_path, index_label='Date')
    else:
        try:
            df = pd.read_csv(file_path, parse_dates=True, index_col='Date')
            if df.empty or 'Close' not in df.columns:
                raise ValueError("Invalid CSV")
        except:
            df = yf.download(ticker, start="2018-01-01")
            df.to_csv(file_path, index_label='Date')
    df = add_indicators(df)
    return df

with st.spinner("Loading data..."):
    df = load_data(ticker, symbol)
st.success("Data loaded successfully!")

# ---------------- Prediction Settings ---------------- #
col1, col2 = st.columns(2)
with col1:
    timeframe = st.radio("Predict by days or months?", ('Days', 'Months'))
with col2:
    value = st.number_input("Number of days or months:", min_value=1, max_value=365, value=30, step=1)

future_days = value * 30 if timeframe == 'Months' else value
label = f"{value} Months" if timeframe == 'Months' else f"{future_days} Days"

# ---------------- Predict Button ---------------- #
if st.button("🔮 Predict Prices"):
    # ---------------- Prepare Data ---------------- #
    data = df[['Close', 'RSI', 'MACD', 'MACD_signal']].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    seq_len = 60
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    # ---------------- Build / Load Model ---------------- #
    model_path = f"saved_models/{symbol}_model.h5"
    os.makedirs("saved_models", exist_ok=True)

    @st.cache_resource(show_spinner=False)
    def get_model(X, y, model_path):
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.4),
                LSTM(128),
                Dropout(0.4),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
            model.save(model_path)
        return model

    with st.spinner("Training / Loading model..."):
        model = get_model(X, y, model_path)
    st.success("Model ready!")

    # ---------------- Predict Future ---------------- #
    last_seq = scaled_data[-seq_len:]
    predicted_prices = []

    for _ in range(future_days):
        input_seq = last_seq.reshape(1, seq_len, X.shape[2])
        pred = model.predict(input_seq, verbose=0)
        predicted_prices.append(pred[0, 0])
        new_row = np.append(pred[0, 0], last_seq[-1, 1:])
        last_seq = np.vstack((last_seq, new_row))[-seq_len:]

    padded = np.zeros((len(predicted_prices), data.shape[1]))
    padded[:, 0] = predicted_prices
    predicted_prices = scaler.inverse_transform(padded)[:, 0]

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price (USD)': predicted_prices})

    # ---------------- Plot ---------------- #
    st.subheader(f"{symbol} Price Prediction - Next {label}")
    fig, ax = plt.subplots(figsize=(12, 6))  # 👈 Medium size graph
    ax.plot(data.index[-100:], data['Close'][-100:], label='Last 100 Days')
    ax.plot(future_dates, predicted_prices, label='Predicted', color='red')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ---------------- Display Predictions ---------------- #
    st.subheader("📈 Sample Predicted Prices")
    st.dataframe(future_df.tail(min(10, future_days)))

    # ---------------- Download CSV ---------------- #
    csv = future_df.to_csv(index=False)
    st.download_button("💾 Download Predictions as CSV", data=csv, file_name=f"{symbol}_predicted.csv")
