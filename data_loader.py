import pandas as pd
import yfinance as yf

REQUIRED_COLS = ["date", "ticker", "headline"]

def load_headlines(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Headlines CSV missing columns: {missing}. Required: {REQUIRED_COLS}")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["headline"] = df["headline"].astype(str).str.strip()
    return df

def get_price_history(tickers, start, end) -> pd.DataFrame:
    """
    Returns a dataframe with columns: date, ticker, Close (adjusted), Return_1d...Return_5d
    """
    frames = []
    for tk in sorted(set(tickers)):
        px = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
        if px.empty:
            continue
        px = px[["Close"]].copy()
        px["Return_1d"] = px["Close"].pct_change(1).shift(-1)
        px["Return_3d"] = px["Close"].pct_change(3).shift(-3)
        px["Return_5d"] = px["Close"].pct_change(5).shift(-5)
        px["ticker"] = tk
        px = px.reset_index().rename(columns={"Date":"date"})
        px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
        frames.append(px)
    if not frames:
        return pd.DataFrame(columns=["date","ticker","Close","Return_1d","Return_3d","Return_5d"])
    out = pd.concat(frames, ignore_index=True)
    return out
