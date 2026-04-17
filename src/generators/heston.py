from typing import Dict

import numpy as np
import pandas as pd

from ..config import REGIME_NAMES


def train_heston(
    returns_df: pd.DataFrame, regime_labels: pd.Series
) -> Dict[str, Dict[str, float]]:
    """Calibrate Heston parameters per regime via method of moments."""
    common = returns_df.index.intersection(regime_labels.index)
    returns_aligned = returns_df.loc[common]
    regimes_aligned = regime_labels.loc[common]

    params: Dict[str, Dict[str, float]] = {}
    for regime_name in REGIME_NAMES:
        mask = regimes_aligned == regime_name
        pooled = returns_aligned.loc[mask].values.ravel()
        pooled = pooled[np.isfinite(pooled)]

        mu = float(np.mean(pooled)) * 252
        daily_var = pooled**2
        theta = float(np.clip(np.mean(daily_var) * 252, 1e-6, 0.5))

        if len(daily_var) > 2:
            ac1 = np.clip(np.corrcoef(daily_var[:-1], daily_var[1:])[0, 1], 0.001, 0.999)
            kappa = float(-252 * np.log(ac1))
        else:
            kappa = 5.0
        kappa = float(np.clip(kappa, 0.1, 20.0))

        xi = float(np.std(daily_var)) * np.sqrt(252) * 2.0 / max(np.sqrt(theta), 1e-4)
        xi = float(np.clip(xi, 0.05, 2.0))

        if len(pooled) > 2:
            dvar = np.diff(daily_var)
            rho = float(np.clip(np.corrcoef(pooled[:-1], dvar)[0, 1], -0.99, 0.99))
        else:
            rho = -0.7

        if 2 * kappa * theta <= xi**2:
            xi = float(np.sqrt(2 * kappa * theta * 0.95))

        params[regime_name] = {
            "mu": mu, "kappa": kappa, "theta": theta,
            "xi": xi, "rho": rho, "v0": theta,
        }
    return params


def heston_generate_next(
    window: np.ndarray,
    regime: int,
    heston_params: Dict[str, Dict[str, float]],
) -> float:
    p = heston_params[REGIME_NAMES[regime]]
    mu, kappa, theta, xi, rho = p["mu"], p["kappa"], p["theta"], p["xi"], p["rho"]
    dt = 1.0 / 252.0

    v_t = float(np.var(window[-5:])) * 252 if len(window) >= 5 else p["v0"]
    v_t = max(v_t, 1e-8)

    z1 = np.random.randn()
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn()

    v_pos = max(v_t, 0.0)
    dv = kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z2
    v_next = max(v_t + dv, 1e-8)
    r = mu * dt + np.sqrt(v_pos * dt) * z1
    return float(r)
