import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(daily_curve: pd.DataFrame, title="Strategy Equity Curve"):
    fig, ax = plt.subplots()
    ax.plot(daily_curve["date"], daily_curve["equity_curve"])
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (Start=1.0)")
    fig.tight_layout()
    return fig
