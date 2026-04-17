import warnings
from typing import Dict, List

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from ..config import REGIME_NAMES
from .stylized_facts import _acf_array, _exponential, _hill_alpha, _leverage_stat, _power_law


TEST_IDS = [
    "hill_tail_index", "jarque_bera", "ljung_box_raw",
    "ljung_box_squared", "arch_lm", "leverage", "acf_absolute_returns",
]


def _test_hill(returns: np.ndarray, benchmark_regime: Dict) -> Dict:
    pos = returns[returns > 0]
    neg = np.abs(returns[returns < 0])

    alpha_upper, caution_upper = _hill_alpha(pos) if len(pos) >= 5 else (np.nan, True)
    alpha_lower, caution_lower = _hill_alpha(neg) if len(neg) >= 5 else (np.nan, True)
    caution = caution_upper or caution_lower

    if np.isfinite(alpha_upper) and np.isfinite(alpha_lower):
        alpha_syn = (alpha_upper + alpha_lower) / 2.0
    elif np.isfinite(alpha_upper):
        alpha_syn = alpha_upper
    elif np.isfinite(alpha_lower):
        alpha_syn = alpha_lower
    else:
        return {
            "test_id": "hill_tail_index", "verdict": "fail",
            "polarity": "benchmark_comparison", "statistic": None,
            "p_value": None, "p_value_bh": None,
            "benchmark": benchmark_regime["hill_alpha_mean"],
            "auxiliary": {"caution": True},
        }

    alpha_real = benchmark_regime["hill_alpha_mean"]
    sigma = max(benchmark_regime["hill_alpha_std"], 0.1)
    diff = abs(alpha_syn - alpha_real)

    if diff <= 1.0 * sigma:
        verdict = "pass"
    elif diff <= 2.0 * sigma:
        verdict = "marginal"
    else:
        verdict = "fail"

    return {
        "test_id": "hill_tail_index", "verdict": verdict,
        "polarity": "benchmark_comparison",
        "statistic": float(alpha_syn), "p_value": None, "p_value_bh": None,
        "benchmark": float(alpha_real),
        "auxiliary": {
            "alpha_upper": float(alpha_upper) if np.isfinite(alpha_upper) else None,
            "alpha_lower": float(alpha_lower) if np.isfinite(alpha_lower) else None,
            "diff_sigma": float(diff / sigma),
            "caution": caution,
        },
    }


def _test_jarque_bera(returns: np.ndarray) -> Dict:
    jb_stat, p_value = sp_stats.jarque_bera(returns)
    if p_value < 0.05:
        verdict = "pass"
    elif p_value < 0.20:
        verdict = "marginal"
    else:
        verdict = "fail"
    return {
        "test_id": "jarque_bera", "verdict": verdict,
        "polarity": "should_reject",
        "statistic": float(jb_stat), "p_value": float(p_value), "p_value_bh": None,
        "benchmark": None,
        "auxiliary": {
            "skewness": float(sp_stats.skew(returns)),
            "kurtosis": float(sp_stats.kurtosis(returns)),
        },
    }


def _test_lb_raw(returns: np.ndarray) -> Dict:
    demeaned = returns - np.mean(returns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = acorr_ljungbox(demeaned, lags=10, return_df=True)
    p_value = float(result["lb_pvalue"].iloc[-1])
    lb_stat = float(result["lb_stat"].iloc[-1])
    if p_value > 0.05:
        verdict = "pass"
    elif p_value >= 0.01:
        verdict = "marginal"
    else:
        verdict = "fail"
    return {
        "test_id": "ljung_box_raw", "verdict": verdict,
        "polarity": "should_not_reject",
        "statistic": lb_stat, "p_value": p_value, "p_value_bh": None,
        "benchmark": None, "auxiliary": {},
    }


def _test_lb_squared(returns: np.ndarray) -> Dict:
    squared = returns ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = acorr_ljungbox(squared, lags=20, return_df=True)
    p_value = float(result["lb_pvalue"].iloc[-1])
    lb_stat = float(result["lb_stat"].iloc[-1])
    if p_value < 0.05:
        verdict = "pass"
    elif p_value < 0.20:
        verdict = "marginal"
    else:
        verdict = "fail"
    return {
        "test_id": "ljung_box_squared", "verdict": verdict,
        "polarity": "should_reject",
        "statistic": lb_stat, "p_value": p_value, "p_value_bh": None,
        "benchmark": None, "auxiliary": {},
    }


def _test_arch_lm(returns: np.ndarray) -> Dict:
    demeaned = returns - np.mean(returns)
    lm_stat, lm_p, f_stat, f_p = het_arch(demeaned, nlags=5)
    p_value = float(lm_p)
    if p_value < 0.05:
        verdict = "pass"
    elif p_value < 0.20:
        verdict = "marginal"
    else:
        verdict = "fail"
    return {
        "test_id": "arch_lm", "verdict": verdict,
        "polarity": "should_reject",
        "statistic": float(lm_stat), "p_value": p_value, "p_value_bh": None,
        "benchmark": None,
        "auxiliary": {"f_stat": float(f_stat), "f_pvalue": float(f_p)},
    }


def _test_leverage(returns: np.ndarray, benchmark_regime: Dict) -> Dict:
    L_plus_syn = _leverage_stat(returns)

    n_boot = 500
    boot_L = np.array([
        _leverage_stat(np.random.permutation(returns)) for _ in range(n_boot)
    ])
    boot_L = boot_L[np.isfinite(boot_L)]
    p_value = float(np.mean(boot_L <= L_plus_syn)) if len(boot_L) else 1.0

    L_plus_real = benchmark_regime["leverage_L_plus"]
    sigma = max(benchmark_regime["leverage_sigma"], 1e-4)

    if L_plus_syn < L_plus_real - 0.5 * sigma:
        verdict = "pass"
    elif L_plus_syn <= 0:
        verdict = "marginal"
    else:
        verdict = "fail"

    return {
        "test_id": "leverage", "verdict": verdict,
        "polarity": "benchmark_comparison",
        "statistic": float(L_plus_syn), "p_value": p_value, "p_value_bh": None,
        "benchmark": float(L_plus_real),
        "auxiliary": {"sigma": float(sigma), "bootstrap_n": int(len(boot_L))},
    }


def _test_acf_abs(returns: np.ndarray) -> Dict:
    abs_ret = np.abs(returns)
    acf = _acf_array(abs_ret, max_lag=50)
    lags = np.arange(1, 51, dtype=np.float64)

    n = len(returns)
    sig_threshold = 2.0 / np.sqrt(n)
    has_structure_beyond_5 = np.any(np.abs(acf[5:]) > sig_threshold)

    power_law_ok = False
    beta_hat = np.nan
    aic_pow = np.inf
    try:
        positive_mask = acf > 0
        if np.sum(positive_mask) >= 5:
            lags_pos = lags[positive_mask]
            acf_pos = acf[positive_mask]
            popt_pow, _ = curve_fit(
                _power_law, lags_pos, acf_pos, p0=[0.5, 0.3],
                bounds=([0, 0.01], [5.0, 3.0]), maxfev=5000,
            )
            residuals_pow = acf_pos - _power_law(lags_pos, *popt_pow)
            rss_pow = float(np.sum(residuals_pow ** 2))
            n_fit = len(acf_pos)
            aic_pow = n_fit * np.log(rss_pow / n_fit + 1e-16) + 2 * 2
            beta_hat = float(popt_pow[1])
            power_law_ok = True
    except (RuntimeError, ValueError):
        pass

    exp_ok = False
    aic_exp = np.inf
    try:
        positive_mask = acf > 0
        if np.sum(positive_mask) >= 5:
            lags_pos = lags[positive_mask]
            acf_pos = acf[positive_mask]
            popt_exp, _ = curve_fit(
                _exponential, lags_pos, acf_pos, p0=[0.5, 0.1],
                bounds=([0, 0.001], [5.0, 5.0]), maxfev=5000,
            )
            residuals_exp = acf_pos - _exponential(lags_pos, *popt_exp)
            rss_exp = float(np.sum(residuals_exp ** 2))
            n_fit = len(acf_pos)
            aic_exp = n_fit * np.log(rss_exp / n_fit + 1e-16) + 2 * 2
            exp_ok = True
    except (RuntimeError, ValueError):
        pass

    delta_aic = aic_exp - aic_pow

    if not has_structure_beyond_5:
        verdict = "fail"
    elif power_law_ok and delta_aic > 0 and 0.2 <= beta_hat <= 0.6:
        verdict = "pass"
    elif power_law_ok and delta_aic > 0 and not (0.2 <= beta_hat <= 0.6):
        verdict = "marginal"
    elif exp_ok and delta_aic < -2:
        verdict = "fail"
    elif exp_ok and -2 <= delta_aic < 0:
        verdict = "marginal"
    else:
        verdict = "marginal"

    return {
        "test_id": "acf_absolute_returns", "verdict": verdict,
        "polarity": "benchmark_comparison",
        "statistic": float(beta_hat) if np.isfinite(beta_hat) else None,
        "p_value": None, "p_value_bh": None,
        "benchmark": None,
        "auxiliary": {
            "aic_power_law": float(aic_pow) if np.isfinite(aic_pow) else None,
            "aic_exponential": float(aic_exp) if np.isfinite(aic_exp) else None,
            "delta_aic": float(delta_aic) if np.isfinite(delta_aic) else None,
            "power_law_fit": power_law_ok,
            "has_structure_beyond_5": bool(has_structure_beyond_5),
        },
    }


def _apply_bh_correction(test_results: List[Dict], alpha_fdr: float = 0.10) -> None:
    BH_TESTS = {"ljung_box_raw", "ljung_box_squared", "arch_lm", "jarque_bera", "leverage"}
    eligible = [t for t in test_results if t["test_id"] in BH_TESTS and t["p_value"] is not None]
    if not eligible:
        return
    p_values = np.array([t["p_value"] for t in eligible])
    m = len(p_values)
    ranks = np.argsort(np.argsort(p_values)) + 1
    raw_adjusted = p_values * m / ranks

    order = np.argsort(ranks)[::-1]
    running_min = 1.0
    adjusted = np.empty(m)
    for idx in order:
        running_min = min(running_min, raw_adjusted[idx])
        adjusted[idx] = running_min

    for i, t in enumerate(eligible):
        t["p_value_bh"] = float(np.clip(adjusted[i], 0, 1))


def run_test_suite(
    returns_252: np.ndarray,
    regime_labels_252: np.ndarray,
    benchmark: Dict[str, Dict],
) -> Dict:
    result: Dict = {}

    for regime_idx, regime_name in enumerate(REGIME_NAMES):
        mask = regime_labels_252 == regime_idx
        regime_ret = returns_252[mask]
        sample_count = int(np.sum(mask))

        bm = benchmark[regime_name]

        if sample_count < 20:
            tests_result = {
                tid: {
                    "test_id": tid, "verdict": "fail",
                    "polarity": "N/A", "statistic": None,
                    "p_value": None, "p_value_bh": None,
                    "benchmark": None,
                    "auxiliary": {
                        "insufficient_data": True,
                        "sample_count": sample_count,
                    },
                }
                for tid in TEST_IDS
            }
            result[regime_name] = {"sample_count": sample_count, "tests": tests_result}
            continue

        tests: List[Dict] = [
            _test_hill(regime_ret, bm),
            _test_jarque_bera(regime_ret),
            _test_lb_raw(regime_ret),
            _test_lb_squared(regime_ret),
            _test_arch_lm(regime_ret),
            _test_leverage(regime_ret, bm),
            _test_acf_abs(regime_ret),
        ]
        _apply_bh_correction(tests)

        result[regime_name] = {
            "sample_count": sample_count,
            "tests": {t["test_id"]: t for t in tests},
        }

    failing = [
        {"regime": rn, "test": tid, "verdict": t["verdict"]}
        for rn in REGIME_NAMES if rn in result
        for tid, t in result[rn]["tests"].items()
        if t["verdict"] in ("fail", "marginal")
    ]

    return {"regime_results": result, "failing_tests_summary": failing}


def print_test_suite(suite_result: Dict) -> None:
    regime_results = suite_result["regime_results"]
    col_width = 14
    header_tests = [
        ("hill_tail_index", "Hill"),
        ("jarque_bera", "JB"),
        ("ljung_box_raw", "LB-Raw"),
        ("ljung_box_squared", "LB-Sq"),
        ("arch_lm", "ARCH-LM"),
        ("leverage", "Leverage"),
        ("acf_absolute_returns", "ACF-Abs"),
    ]

    print("\n" + "=" * 120)
    print(f"{'Regime':<14}", end="")
    for _, short_name in header_tests:
        print(f"{short_name:^{col_width}}", end="")
    print(f"  {'N':>5}")
    print("-" * 120)

    verdict_symbols = {"pass": "PASS", "marginal": "MARG", "fail": "FAIL"}

    for regime_name in REGIME_NAMES:
        rr = regime_results[regime_name]
        print(f"{regime_name:<14}", end="")
        for tid, _ in header_tests:
            t = rr["tests"][tid]
            v = verdict_symbols.get(t["verdict"], "???")
            stat = t["statistic"]
            cell = f"{v}({stat:.2f})" if stat is not None else v
            print(f"{cell:^{col_width}}", end="")
        print(f"  {rr['sample_count']:>5}")
    print("=" * 120)

    failing = suite_result["failing_tests_summary"]
    n_fail = sum(1 for f in failing if f["verdict"] == "fail")
    n_marg = sum(1 for f in failing if f["verdict"] == "marginal")
    print(
        f"\nSummary: {n_fail} failures, {n_marg} marginal across "
        f"{len(REGIME_NAMES)} regimes × {len(header_tests)} tests = "
        f"{len(REGIME_NAMES) * len(header_tests)} cells"
    )
    if n_fail > 0:
        print("\nFail details:")
        for f in failing:
            if f["verdict"] == "fail":
                print(f"  {f['regime']:>12} × {f['test']}")
    print()
