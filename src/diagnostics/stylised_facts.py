from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..config import REGIME_NAMES


def _hill_alpha(x: np.ndarray) -> Tuple[float, bool]:
    x = x[np.isfinite(x)]
    x = np.abs(x[x != 0])
    n = len(x)
    k = int(np.floor(n ** 0.6))
    if k < 4:
        return np.nan, True
    x_sorted = np.sort(x)[::-1]
    log_ratios = np.log(x_sorted[:k] / x_sorted[k])
    H_k = np.mean(log_ratios)
    if H_k <= 0:
        return np.nan, True
    return 1.0 / H_k, False


def _leverage_stat(returns: np.ndarray, max_tau: int = 10) -> float:
    returns = returns[np.isfinite(returns)]
    L_plus = 0.0
    for tau in range(1, max_tau + 1):
        if len(returns) <= tau + 1:
            break
        r_t = returns[: len(returns) - tau]
        r2_t_tau = returns[tau:] ** 2
        if np.std(r_t) < 1e-12 or np.std(r2_t_tau) < 1e-12:
            continue
        corr = np.corrcoef(r_t, r2_t_tau)[0, 1]
        if np.isfinite(corr):
            L_plus += corr
    return L_plus


def _leverage_bootstrap_sigma(
    returns: np.ndarray, n_boot: int = 500, max_tau: int = 10
) -> float:
    L_plus_samples = []
    for _ in range(n_boot):
        shuffled = np.random.permutation(returns)
        L_plus_samples.append(_leverage_stat(shuffled, max_tau))
    return float(np.std(L_plus_samples))


def _acf_array(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    x = x - np.mean(x)
    n = len(x)
    var = np.var(x)
    if var < 1e-16:
        return np.zeros(max_lag)
    usable_lags = min(max_lag, n - 2)
    acf = np.zeros(max_lag)
    for lag in range(1, usable_lags + 1):
        acf[lag - 1] = np.mean(x[: n - lag] * x[lag:]) / var
    return acf


def _power_law(tau: np.ndarray, a: float, beta: float) -> np.ndarray:
    return a * tau ** (-beta)


def _exponential(tau: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(-b * tau)


def compute_benchmark(
    individual_returns: Dict[str, pd.Series],
    regime_labels: pd.Series,
) -> Dict[str, Dict]:
    regime_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
    regime_labels_int = regime_labels.map(regime_to_idx).dropna().astype(int)

    benchmark: Dict[str, Dict] = {}

    for regime_name in REGIME_NAMES:
        regime_idx = regime_to_idx[regime_name]

        hill_alphas_upper = []
        hill_alphas_lower = []

        for ticker, series in individual_returns.items():
            common = series.index.intersection(regime_labels_int.index)
            if len(common) < 30:
                continue
            ret = series.loc[common].values
            reg = regime_labels_int.loc[common].values
            regime_ret = ret[reg == regime_idx]
            regime_ret = regime_ret[np.isfinite(regime_ret)]
            if len(regime_ret) < 30:
                continue

            pos = regime_ret[regime_ret > 0]
            if len(pos) >= 10:
                alpha, caution = _hill_alpha(pos)
                if not caution and np.isfinite(alpha):
                    hill_alphas_upper.append(alpha)

            neg = np.abs(regime_ret[regime_ret < 0])
            if len(neg) >= 10:
                alpha, caution = _hill_alpha(neg)
                if not caution and np.isfinite(alpha):
                    hill_alphas_lower.append(alpha)

        hill_combined = [
            (u + l) / 2.0 for u, l in zip(hill_alphas_upper, hill_alphas_lower)
        ] or [3.5]

        all_regime_ret = []
        for ticker, series in individual_returns.items():
            common = series.index.intersection(regime_labels_int.index)
            if len(common) < 30:
                continue
            ret = series.loc[common].values
            reg = regime_labels_int.loc[common].values
            regime_ret = ret[reg == regime_idx]
            regime_ret = regime_ret[np.isfinite(regime_ret)]
            all_regime_ret.extend(regime_ret.tolist())
        all_regime_ret_arr = np.array(all_regime_ret, dtype=np.float64)

        L_plus_real = _leverage_stat(all_regime_ret_arr)

        subsample_size = min(len(all_regime_ret_arr), 5000)
        boot_sample = np.random.choice(
            all_regime_ret_arr, size=subsample_size, replace=False
        )
        L_plus_sigma = _leverage_bootstrap_sigma(boot_sample, n_boot=500)

        benchmark[regime_name] = {
            "hill_alpha_mean": float(np.mean(hill_combined)),
            "hill_alpha_std": float(np.std(hill_combined)) if len(hill_combined) > 1 else 1.0,
            "leverage_L_plus": float(L_plus_real),
            "leverage_sigma": float(L_plus_sigma) if L_plus_sigma > 0 else 0.1,
        }

    return benchmark
