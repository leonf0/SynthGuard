from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def fit_regime_gmm(
    X: np.ndarray,
    features_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> Tuple[GaussianMixture, pd.Series]:
    gmm = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        init_params="k-means++",
        max_iter=300,
        tol=1e-4,
        n_init=3,
        reg_covar=1e-6,
        random_state=42,
    )
    gmm.fit(X)

    raw_labels = gmm.predict(X)

    vix_ema_values = features_df["vix_ema"].values
    component_mean_vix = np.array(
        [vix_ema_values[raw_labels == c].mean() for c in range(N_REGIMES)]
    )
    sorted_order = np.argsort(component_mean_vix)  
    index_to_regime = {sorted_order[i]: REGIME_NAMES[i] for i in range(N_REGIMES)}

    regime_labels = pd.Series(
        [index_to_regime[l] for l in raw_labels], index=dates, name="regime"
    )
    return gmm, regime_labels


def smooth_regime_labels(labels: pd.Series, min_run: int = 5) -> pd.Series:
    smoothed = labels.copy()
    vals = smoothed.values
    n = len(vals)
    i = 0
    while i < n:
        j = i
        while j < n and vals[j] == vals[i]:
            j += 1
        run_len = j - i
        if run_len < min_run:
            left = vals[i - 1] if i > 0 else vals[j] if j < n else vals[i]
            right = vals[j] if j < n else vals[i - 1] if i > 0 else vals[i]
            replacement = left if left == right else left  # majority of surrounding
            vals[i:j] = replacement
        i = j
    smoothed[:] = vals
    return smoothed
