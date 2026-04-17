from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..config import REGIME_NAMES


def train_gbm(
    returns_df: pd.DataFrame, regime_labels: pd.Series
) -> Dict[str, Tuple[float, float]]:
    dt = 1.0 / 252.0
    common = returns_df.index.intersection(regime_labels.index)
    returns_aligned = returns_df.loc[common]
    regimes_aligned = regime_labels.loc[common]

    params: Dict[str, Tuple[float, float]] = {}
    for regime_name in REGIME_NAMES:
        mask = regimes_aligned == regime_name
        pooled = returns_aligned.loc[mask].values.ravel()
        pooled = pooled[np.isfinite(pooled)]
        mu = pooled.mean() / dt
        sigma = pooled.std() / np.sqrt(dt)
        params[regime_name] = (mu, sigma)
    return params


def gbm_generate_next(
    window: np.ndarray,
    regime: int,
    gbm_params: Dict[str, Tuple[float, float]],
) -> float:
    dt = 1.0 / 252.0
    mu, sigma = gbm_params[REGIME_NAMES[regime]]
    return mu * dt + sigma * np.sqrt(dt) * np.random.randn()
