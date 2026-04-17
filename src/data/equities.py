from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from ..config import TICKERS


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
    return log_returns.dropna(how="any")


def fetch_individual_returns(
    tickers: List[str] = TICKERS,
    start: str = "2005-01-01",
    end: str = "2025-12-31",
) -> Dict[str, pd.Series]:
    """Per-equity daily log-returns keyed by ticker."""
    result: Dict[str, pd.Series] = {}
    for t in tickers:
        prices = yf.download(t, start=start, end=end, progress=False)["Close"].squeeze()
        log_ret = np.log(prices / prices.shift(1)).iloc[1:].dropna()
        log_ret.name = t
        result[t] = log_ret
    return result
