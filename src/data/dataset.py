from typing import List, Tuple

import numpy as np
import pandas as pd

from ..config import REGIME_NAMES, SEQUENCE_LENGTH, WINDOW


def build_training_dataset(
    returns_df: pd.DataFrame,
    regime_labels: pd.Series,
    window: int = WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  
    regime_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
    common_dates = returns_df.index.intersection(regime_labels.index)
    returns_aligned = returns_df.loc[common_dates]
    regimes_aligned = regime_labels.loc[common_dates]

    X_windows: List[np.ndarray] = []
    X_regimes: List[int] = []
    y_targets: List[float] = []

    for col in returns_aligned.columns:
        r = returns_aligned[col].values
        reg = regimes_aligned.values
        for t in range(window, len(r) - 1):
            label = reg[t]
            if label not in regime_to_idx:
                continue
            X_windows.append(r[t - window : t])
            X_regimes.append(regime_to_idx[label])
            y_targets.append(r[t + 1])

    return (
        np.array(X_windows, dtype=np.float32),
        np.array(X_regimes, dtype=np.int64),
        np.array(y_targets, dtype=np.float32),
    )


def build_sequence_dataset(
    returns_df: pd.DataFrame,
    regime_labels: pd.Series,
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:

    regime_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
    common = returns_df.index.intersection(regime_labels.index)
    returns_aligned = returns_df.loc[common]
    regimes_aligned = regime_labels.loc[common].map(regime_to_idx)

    sequences: List[np.ndarray] = []
    seq_regimes: List[int] = []

    for col in returns_aligned.columns:
        r = returns_aligned[col].values.astype(np.float32)
        reg = regimes_aligned.values
        n = len(r)
        if n < seq_len:
            continue
        for start in range(0, n - seq_len + 1, stride):
            window_r = r[start : start + seq_len]
            window_reg = reg[start : start + seq_len]
            if np.any(~np.isfinite(window_r)) or np.any(pd.isna(window_reg)):
                continue
            modal_regime = int(pd.Series(window_reg).mode().iloc[0])
            sequences.append(window_r)
            seq_regimes.append(modal_regime)

    return (
        np.stack(sequences, axis=0).astype(np.float32),
        np.array(seq_regimes, dtype=np.int64),
    )
