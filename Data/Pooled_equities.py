from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "BAC",
    "XOM", "CVX", "JNJ", "PG", "UNH", "HD", "KO", "PEP", "WMT",
]


def fetch_equities_returns(
    tickers: List[str] = TICKERS,
    start: str = "2005-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.dropna(how="any")
    log_returns = np.log(prices / prices.shift(1)).iloc[1:]
    log_returns = log_returns.dropna(how="any")
    return log_returns
