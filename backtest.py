import pandas as pd
import numpy as np

def flatten_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns returned by yfinance (e.g., ('Close', 'AAPL'))
    into a normal DataFrame with columns ['date', 'ticker', 'Close', 'Return_1d', ...]
    """
    # Handle MultiIndex columns (from yfinance)
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.stack(level=1).reset_index()
        prices = prices.rename(columns={"level_1": "ticker"})
    else:
        # If already flat, just ensure consistent naming
        if "Date" in prices.columns:
            prices = prices.rename(columns={"Date": "date"})
    return prices


def join_sentiment_prices(daily_sent: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sentiment data with stock prices on date and ticker.
    Handles both MultiIndex and single-level cases.
    """
    prices = flatten_prices(prices)

    # Ensure datetime and consistent columns
    if "date" not in prices.columns and "Date" in prices.columns:
        prices = prices.rename(columns={"Date": "date"})

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.tz_localize(None)
    daily_sent["date"] = pd.to_datetime(daily_sent["date"], errors="coerce").dt.tz_localize(None)

    # Drop duplicates before merging
    prices = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
    daily_sent = daily_sent.drop_duplicates(subset=["date", "ticker"], keep="last")

    # Perform safe merge
    try:
        df = pd.merge(prices, daily_sent, on=["date", "ticker"], how="left")
    except Exception:
        # Fallback: if ticker missing in one side, convert both to string and retry
        prices["ticker"] = prices["ticker"].astype(str)
        daily_sent["ticker"] = daily_sent["ticker"].astype(str)
        df = pd.merge(prices, daily_sent, on=["date", "ticker"], how="left")

    return df


def simple_long_only_backtest(df: pd.DataFrame, horizon="Return_1d", thresh=0.05) -> dict:
    """
    Long if avg_sent > thresh; hold for horizon (1d/3d/5d).
    Equally weighted across tickers & signals appearing that day.
    """
    d = df.copy()
    if "avg_sent" not in d.columns or horizon not in d.columns:
        return {"n_signals":0,"cumulative_return":0,"avg_trade_return":0,"hit_rate":0,"daily_curve":pd.DataFrame()}

    d["signal"] = (d["avg_sent"] > thresh).astype(int)
    d["trade_ret"] = d[horizon].fillna(0) * d["signal"]
    daily = d.groupby("date", as_index=False)["trade_ret"].mean()
    daily["equity_curve"] = (1 + daily["trade_ret"]).cumprod()

    cumret = daily["equity_curve"].iloc[-1] - 1 if len(daily) else 0.0
    avg_trade_ret = d.loc[d["signal"]==1, horizon].mean() if (d["signal"]==1).any() else 0.0
    hit_rate = (d.loc[d["signal"]==1, horizon] > 0).mean() if (d["signal"]==1).any() else 0.0

    return {
        "n_signals": int((d["signal"]==1).sum()),
        "cumulative_return": float(cumret),
        "avg_trade_return": float(avg_trade_ret if pd.notnull(avg_trade_ret) else 0.0),
        "hit_rate": float(hit_rate if pd.notnull(hit_rate) else 0.0),
        "daily_curve": daily
    }


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correlate same-day avg_sent with future returns.
    """
    sub = df[["avg_sent","Return_1d","Return_3d","Return_5d"]].dropna()
    if sub.empty:
        return pd.DataFrame()
    corr = sub.corr().loc[["avg_sent"], ["Return_1d","Return_3d","Return_5d"]]
    corr = corr.rename(index={"avg_sent":"Sentiment vs Returns"})
    return corr
