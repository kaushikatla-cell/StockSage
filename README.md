# 🧠 StockSage — AI Market Sentiment & Strategy Analyzer

**Live Demo (after deploy):** `https://stocksage.streamlit.app`  
Quantifies sentiment from financial headlines/posts and correlates it with short-term stock returns. Includes a Streamlit dashboard, a simple backtest, and clean, modular Python code.

## ✨ Features
- Upload headlines (`date, ticker, headline`)
- VADER sentiment scoring per headline → daily average per ticker
- Auto-downloads price history with yfinance
- Correlates sentiment with 1/3/5-day forward returns
- Simple long-only backtest (avg_sent > threshold)
- Export joined dataset for deeper research

## 📂 Structure
