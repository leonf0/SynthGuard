from typing import Dict

import numpy as np
import pandas as pd

from ..config import REGIME_NAMES


def train_garch(
    returns_df: pd.DataFrame, regime_labels: pd.Series
) -> Dict[str, Dict[str, float]]:
    from arch import arch_model

    common = returns_df.index.intersection(regime_labels.index)
    returns_aligned = returns_df.loc[common]
    regimes_aligned = regime_labels.loc[common]

    params: Dict[str, Dict[str, float]] = {}
    for regime_name in REGIME_NAMES:
        mask = regimes_aligned == regime_name
        pooled = returns_aligned.loc[mask].values.ravel()
        pooled = pooled[np.isfinite(pooled)]

        am = arch_model(
            pooled * 100, vol="Garch", p=1, q=1, dist="StudentsT", mean="Zero"
        )
        res = am.fit(disp="off", show_warning=False)
        alpha_val = res.params.get("alpha[1]", 0.05)
        beta_val = res.params.get("beta[1]", 0.90)
        if alpha_val + beta_val >= 1.0:
            beta_val = 0.999 - alpha_val
        params[regime_name] = {
            "omega": res.params["omega"] / 1e4,
            "alpha": alpha_val,
            "beta": beta_val,
            "nu": res.params.get("nu", 30.0),
        }
    return params


def garch_generate_next(
    window: np.ndarray,
    regime: int,
    garch_params: Dict[str, Dict[str, float]],
) -> float:

    p = garch_params[REGIME_NAMES[regime]]
    omega, alpha, beta, nu = p["omega"], p["alpha"], p["beta"], p["nu"]

    sigma2 = np.var(window)
    for r in window:
        sigma2 = omega + alpha * r**2 + beta * sigma2

    eps = np.random.standard_t(df=max(nu, 2.01))
    return np.sqrt(sigma2) * eps
