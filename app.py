import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

from data_loader import load_headlines, get_price_history
from sentiment import score_headlines
from backtest import join_sentiment_prices, simple_long_only_backtest, correlation_table
from viz import plot_equity_curve

st.set_page_config(page_title="StockSage — AI Market Sentiment & Strategy", page_icon="🧠", layout="wide")
st.title("🧠 StockSage — AI Market Sentiment & Strategy Analyzer")
st.caption("Quantifies sentiment from headlines and correlates it with short-term stock returns. Educational demo — not investment advice.")

st.markdown("**Step 1.** Upload a CSV with columns: `date, ticker, headline` (see sample).")
up = st.file_uploader("Upload headlines CSV", type=["csv"])
use_sample = st.checkbox("Use included sample_headlines.csv")
if not up and not use_sample:
    st.info("Upload a file or tick the sample to continue.")
    st.stop()

# Load headlines
try:
    df_head = load_headlines("sample_headlines.csv" if use_sample else up)
except Exception as e:
    st.error(f"Could not read headlines file: {e}")
    st.stop()

st.success(f"Loaded {len(df_head):,} headlines across {df_head['ticker'].nunique()} tickers.")
st.dataframe(df_head.head(10))

# Score sentiment
row_sent, daily_sent = score_headlines(df_head)
st.subheader("📊 Daily Sentiment by Ticker")
st.dataframe(daily_sent.sort_values(["date","ticker"]).head(20))

# Price history window
min_date = df_head["date"].min()
max_date = df_head["date"].max()
pad_days = 30
start = (min_date - timedelta(days=pad_days)).date()
end = (max_date + timedelta(days=10)).date()

tickers = df_head["ticker"].unique().tolist()
with st.spinner("Downloading price history..."):
    px = get_price_history(tickers, start, end)

if px.empty:
    st.error("No price data returned. Check ticker symbols and internet access.")
    st.stop()

st.subheader("💵 Price Sample")
st.dataframe(px.sort_values(["ticker","date"]).head(10))

# Join & analytics
joined = join_sentiment_prices(daily_sent, px)

st.subheader("🔗 Joined Sentiment + Prices (sample)")
st.dataframe(joined.sort_values(["ticker","date"]).head(20))

st.sidebar.header("Backtest Settings")
horizon = st.sidebar.selectbox("Horizon", ["Return_1d","Return_3d","Return_5d"], index=0)
thresh = st.sidebar.slider("Sentiment threshold (long if avg_sent > threshold)", min_value=-0.5, max_value=0.5, value=0.05, step=0.01)

res = simple_long_only_backtest(joined, horizon=horizon, thresh=thresh)
st.subheader("🧪 Backtest Summary")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Signals", f"{res['n_signals']}")
c2.metric("Cumulative Return", f"{res['cumulative_return']*100:,.2f}%")
c3.metric("Avg Trade Return", f"{res['avg_trade_return']*100:,.2f}%")
c4.metric("Hit Rate", f"{res['hit_rate']*100:,.1f}%")

if not res["daily_curve"].empty:
    fig = plot_equity_curve(res["daily_curve"])
    st.pyplot(fig)

st.subheader("📈 Correlation: Sentiment vs. Future Returns")
corr = correlation_table(joined)
if corr.empty:
    st.write("Not enough data to compute correlation.")
else:
    st.dataframe(corr.style.format("{:.3f}"))

st.download_button(
    "Download Joined Dataset (CSV)",
    data=joined.to_csv(index=False).encode("utf-8"),
    file_name="stocksage_joined.csv",
    mime="text/csv"
)

st.caption("Limitations: toy strategy; does not account for transaction costs, slippage, or look-ahead bias. For education only.")
