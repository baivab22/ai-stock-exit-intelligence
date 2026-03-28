import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from utils.features import add_features
from utils.preprocessing import scale_features
from utils.sequences import create_sequences

# --------------------------------------
# Page config
# --------------------------------------
st.set_page_config(page_title="AI Trading System", layout="wide")

st.markdown("<h1 style='text-align: center;'>📈 AI Stock Exit Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Early Warning Signals for Market Downside Risk</h4>", unsafe_allow_html=True)

# --------------------------------------
# Load data and model
# --------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("filtered_1year_data.csv")
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("thesis_exit_model.keras", compile=False)

df = load_data()
model = load_model()

# --------------------------------------
# Sidebar controls
# --------------------------------------
st.sidebar.header("⚙️ Controls")
symbols = df['symbol'].unique()
stock = st.sidebar.selectbox("Select Stock", symbols)

date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])
threshold = st.sidebar.slider("Sell Threshold", 0.1, 0.9, 0.4)
initial_investment = st.sidebar.number_input("Investment ($)", value=1000)

# --------------------------------------
# Filter data
# --------------------------------------
stock_df = df[df['symbol'] == stock].copy()
stock_df = stock_df[
    (stock_df['date'] >= pd.to_datetime(date_range[0])) &
    (stock_df['date'] <= pd.to_datetime(date_range[1]))
]
stock_df = stock_df.sort_values('date')
stock_df = add_features(stock_df).dropna()

# Features to use
FEATURE_COLS = [
    'open','high','low','close','volume','per_change','traded_amount',
    'return_1','return_5','ma_5','ma_10','ma_20',
    'volatility_5','momentum_5','rsi','macd','bb_width','vol_ratio','day_of_week','month'
]

stock_df, scaler = scale_features(stock_df, FEATURE_COLS)

# --------------------------------------
# Create sequences and predict
# --------------------------------------
WINDOW_SIZE = 20
X, dates, prices = create_sequences(stock_df, FEATURE_COLS, WINDOW_SIZE)
probs = model.predict(X).flatten()
signals = (probs > threshold).astype(int)

# --------------------------------------
# Price chart with signals
# --------------------------------------
st.subheader("📈 Price & AI Signals")
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price', line=dict(color='cyan')))
sell_idx = np.where(signals == 1)[0]
fig.add_trace(go.Scatter(
    x=np.array(dates)[sell_idx],
    y=np.array(prices)[sell_idx],
    mode='markers',
    name='SELL',
    marker=dict(color='red', size=8)
))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------
# Metrics
# --------------------------------------
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Probability", f"{np.mean(probs):.2f}")
col2.metric("Sell Signals", int(np.sum(signals)))
col3.metric("Sell %", f"{np.mean(signals)*100:.2f}%")

# --------------------------------------
# Strategy vs Buy & Hold
# --------------------------------------
returns = np.diff(prices) / prices[:-1]
returns = np.append(returns, 0)
strategy = [0 if s == 1 else r for s, r in zip(signals, returns)]
strategy_curve = np.cumsum(strategy)
benchmark_curve = np.cumsum(returns)

st.subheader("📉 Strategy vs Buy & Hold")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=strategy_curve, name="AI Strategy"))
fig2.add_trace(go.Scatter(y=benchmark_curve, name="Buy & Hold"))
fig2.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------
# Risk dashboard
# --------------------------------------
st.subheader("⚠️ Risk Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Strategy Return", f"{strategy_curve[-1]:.2f}")
col2.metric("Benchmark Return", f"{benchmark_curve[-1]:.2f}")
col3.metric("Risk Reduction", f"{benchmark_curve.min() - strategy_curve.min():.2f}")

# --------------------------------------
# Signals table
# --------------------------------------
st.subheader("📅 Daily Signals")
table = pd.DataFrame({
    "Date": dates,
    "Price": prices,
    "Probability": probs,
    "Signal": ["SELL 🔴" if s==1 else "HOLD 🟢" for s in signals]
})
st.dataframe(table.tail(50), use_container_width=True)

# --------------------------------------
# Investment simulator
# --------------------------------------
st.subheader("💰 Portfolio Simulator")
value = initial_investment
portfolio = []
for r in strategy:
    value *= (1 + r)
    portfolio.append(value)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=portfolio, name="Portfolio"))
fig3.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig3, use_container_width=True)
st.success(f"Final Portfolio Value: ${value:.2f}")