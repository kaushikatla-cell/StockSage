import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def score_headlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds compound sentiment score per headline and daily averages per ticker.
    """
    df = df.copy()
    df["compound"] = df["headline"].astype(str).apply(lambda x: _analyzer.polarity_scores(x)["compound"])
    # aggregate per date,ticker
    daily = (df.groupby(["date","ticker"], as_index=False)
               .agg(avg_sent=("compound","mean"),
                    n_items=("compound","count")))
    return df, daily
