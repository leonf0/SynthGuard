from typing import List

import torch

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQUENCE_LENGTH: int = 252
WINDOW: int = 30
SEED_WINDOW: int = 60

REGIME_NAMES: List[str] = ["low_vol", "mid_vol", "high_vol"]
N_REGIMES: int = 3

TICKERS: List[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "BAC",
    "XOM", "CVX", "JNJ", "PG", "UNH", "HD", "KO", "PEP", "WMT",
]
