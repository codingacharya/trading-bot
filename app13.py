import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator, SMAIndicator
import datetime

st.set_page_config("NIFTY Options Strategy", layout="wide")

# =====================================================
# LOAD STOCK LIST
# =====================================================
@st.cache_data
def load_stocks():
    df = pd.read_csv("nifty_stocks.csv")
    if "Stock" not in df.columns:
        st.error("CSV must contain column: Stock")
        st.stop()
    return df["Stock"].dropna().unique().tolist()

stocks = load_stocks()

# =====================================================
# FETCH DATA FROM YAHOO FINANCE (SAFE 1-D FIX)
# =====================================================
@st.cache_data
def fetch_data(symbol):
    df = yf.download(
        symbol,
        period="5d",
        interval="5m",
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return None

    # FIX: Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Force OHLC to 1-D numeric series
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Stock"] = symbol
    return df

# =====================================================
# INDICATORS (100% 1-D SAFE)
# =====================================================
def add_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    bb60 = BollingerBands(close, 60)
    bb105 = BollingerBands(close, 105)
    bb150 = BollingerBands(close, 150)

    df["BB60"] = (bb60.bollinger_hband() - bb60.bollinger_lband()) / close * 100
    df["BB105"] = (bb105.bollinger_hband() - bb105.bollinger_lband()) / close * 100
    df["BB150"] = (bb150.bollinger_hband() - bb150.bollinger_lband()) / close * 100

    df["RSI20"] = RSIIndicator(close, 20).rsi()
    df["WILLR28"] = WilliamsRIndicator(high, low, close, 28).williams_r()

    dmi6 = ADXIndicator(high, low, close, 6)
    dmi20 = ADXIndicator(high, low, close, 20)

    df["+DI6"] = dmi6.adx_pos()
    df["-DI6"] = dmi6.adx_neg()
    df["+DI20"] = dmi20.adx_pos()
    df["-DI20"] = dmi20.adx_neg()

    df["MA8"] = SMAIndicator(close, 8).sma_indicator()
    return df

# =====================================================
# MAIN PIPELINE
# =====================================================
frames = []

for stock in stocks:
    data = fetch_data(stock)
    if data is not None:
        frames.append(add_indicators(data))

if not frames:
    st.error("No data fetched from Yahoo Finance")
    st.stop()

df = pd.concat(frames, ignore_index=True)

# =====================================================
# TIME FILTER (After 10:00 AM)
# =====================================================
df["Time"] = df["Datetime"].dt.time
df = df[df["Time"] >= datetime.time(10, 0)]

# =====================================================
# CALL SIDE CONDITIONS
# =====================================================
df["CALL_ENTRY"] = (
    (df["BB60"] <= 35) &
    (df["RSI20"].between(65, 100)) &
    (df["WILLR28"].between(-20, 0)) &
    (df["+DI6"] >= 40) & (df["-DI6"] <= 12) &
    (df["+DI20"] >= 35) & (df["-DI20"] <= 15)
)

# =====================================================
# PUT SIDE CONDITIONS
# =====================================================
df["PUT_ENTRY"] = (
    (df["BB60"] <= 35) &
    (df["RSI20"].between(1, 40)) &
    (df["WILLR28"].between(-100, -80)) &
    (df["-DI6"] >= 35) & (df["+DI6"] <= 15) &
    (df["-DI20"] >= 30) & (df["+DI20"] <= 15)
)

# =====================================================
# EXIT CONDITIONS
# =====================================================
df["CALL_EXIT"] = (
    (abs(df["+DI20"] - df["-DI20"]) < 10) |
    (df["Close"] < df["MA8"])
)

df["PUT_EXIT"] = (
    (abs(df["+DI20"] - df["-DI20"]) < 10) |
    (df["Close"] > df["MA8"])
)

# =====================================================
# ENTRY DURATION (MINUTES)
# =====================================================
df["Entry_Time"] = df["Datetime"].where(df["CALL_ENTRY"] | df["PUT_ENTRY"])
df["Duration_Min"] = (
    df.groupby("Stock")["Entry_Time"]
    .transform(lambda x: (x - x.min()).dt.total_seconds() / 60)
)

# =====================================================
# DASHBOARD
# =====================================================
st.title("📊 NIFTY CALL & PUT Strategy Dashboard")

cols = [
    "Stock", "BB60", "BB105", "BB150",
    "WILLR28", "RSI20",
    "+DI6", "-DI6", "+DI20", "-DI20",
    "Entry_Time", "Duration_Min"
]

tab1, tab2 = st.tabs(["📈 CALL SIDE", "📉 PUT SIDE"])

with tab1:
    st.subheader("CALL SIDE – ACTIVE SIGNALS")
    call_df = df[df["CALL_ENTRY"] & ~df["CALL_EXIT"]].sort_values("BB60")
    st.dataframe(call_df[cols], use_container_width=True)

with tab2:
    st.subheader("PUT SIDE – ACTIVE SIGNALS")
    put_df = df[df["PUT_ENTRY"] & ~df["PUT_EXIT"]].sort_values("BB60")
    st.dataframe(put_df[cols], use_container_width=True)

st.caption("📡 Data: Yahoo Finance | ⏱ From 10:00 AM | 🔁 Multiple entries allowed")
