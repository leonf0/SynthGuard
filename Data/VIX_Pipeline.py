from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def fetch_vix_data(
    start: str = "2005-01-01", end: str = "2025-12-31"
) -> Tuple[pd.Series, pd.Series]:
    vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"].squeeze()
    tnx = yf.download("^TNX", start=start, end=end, progress=False)["Close"].squeeze()
    vix.name, tnx.name = "VIX", "TNX"
    return vix, tnx


def align_and_fill(
    vix: pd.Series, tnx: pd.Series, ffill_limit: int = 3, max_missing_frac: float = 0.02
) -> Tuple[pd.Series, pd.Series]:
    common_idx = vix.index.union(tnx.index).sort_values()
    vix = vix.reindex(common_idx).ffill(limit=ffill_limit)
    tnx = tnx.reindex(common_idx).ffill(limit=ffill_limit)
    mask = vix.notna() & tnx.notna()
    missing_frac = 1.0 - mask.mean()
    if missing_frac > max_missing_frac:
        raise ValueError(
            f"DataQualityError: {missing_frac:.2%} missing after alignment (limit {max_missing_frac:.0%})"
        )
    vix, tnx = vix[mask], tnx[mask]
    return vix, tnx


def build_vix_features(
    vix: pd.Series, tnx: pd.Series
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, pd.DatetimeIndex]:
    vix_ema = vix.ewm(span=5, adjust=False).mean()
    vix_21d_chg = vix_ema.pct_change(periods=21)
    tnx_21d_chg = tnx.diff(periods=21)

    features = pd.DataFrame(
        {"vix_ema": vix_ema, "vix_21d_chg": vix_21d_chg, "tnx_21d_chg": tnx_21d_chg}
    )
    features = features.iloc[21:] 
    features = features.dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    return features, X, scaler, features.index


def fetch_vix_pipeline(
    start: str = "2005-01-01", end: str = "2025-12-31"
) -> Tuple[pd.Series, pd.DataFrame, np.ndarray, StandardScaler, pd.DatetimeIndex]:
    vix_raw, tnx_raw = fetch_vix_data(start, end)
    vix, tnx = align_and_fill(vix_raw, tnx_raw)
    features_df, X, scaler, dates = build_vix_features(vix, tnx)
    return vix, features_df, X, scaler, dates
