from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

def fetch_individual_returns(
    tickers: List[str] = TICKERS,
    start: str = "2005-01-01",
    end: str = "2025-12-31",
) -> Dict[str, pd.Series]:
    result: Dict[str, pd.Series] = {}
    for t in tickers:
        prices = yf.download(t, start=start, end=end, progress=False)["Close"].squeeze()
        log_ret = np.log(prices / prices.shift(1)).iloc[1:].dropna()
        log_ret.name = t
        result[t] = log_ret
    return result
