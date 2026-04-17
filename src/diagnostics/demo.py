from typing import Dict

import numpy as np
import pandas as pd

from ..config import REGIME_NAMES, SEED_WINDOW, SEQUENCE_LENGTH, WINDOW
from ..generators import cvae_generate_next
from ..regimes import generate_regime_sequence
from .stylized_facts import compute_benchmark
from .tests import print_test_suite, run_test_suite


def run_cvae_demo(
    train_all_dict: Dict,
    individual_returns: Dict[str, pd.Series],
) -> Dict:
    regime_labels = train_all_dict["regime_labels"]
    trans_matrix = train_all_dict["transition_matrix"]
    cvae_model = train_all_dict["cvae_model"]
    cvae_res = train_all_dict["cvae_residual_std"]

    print("=== Computing benchmark from real equity data ===")
    benchmark = compute_benchmark(individual_returns, regime_labels)
    for rn, bm in benchmark.items():
        print(
            f"  {rn}: Hill α={bm['hill_alpha_mean']:.2f}±{bm['hill_alpha_std']:.2f}, "
            f"L+={bm['leverage_L_plus']:.4f}±{bm['leverage_sigma']:.4f}"
        )

    regime_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
    regime_labels_int = regime_labels.map(regime_to_idx).dropna().astype(int)

    ticker = list(individual_returns.keys())[0]
    series = individual_returns[ticker]
    common = series.index.intersection(regime_labels_int.index)
    series_aligned = series.loc[common]
    reg_aligned = regime_labels_int.loc[common]

    seed_start = np.random.randint(0, len(series_aligned) - SEED_WINDOW)
    seed = series_aligned.values[seed_start : seed_start + SEED_WINDOW].astype(np.float32)
    initial_regime = int(reg_aligned.values[seed_start + SEED_WINDOW - 1])
    print(
        f"\n=== Seed: {ticker}, start idx {seed_start}, "
        f"initial regime = {REGIME_NAMES[initial_regime]} ==="
    )

    print("=== Generating 252-step CVAE sequence ===")
    regime_seq = generate_regime_sequence(trans_matrix, initial_regime, SEQUENCE_LENGTH)

    buffer = list(seed)
    for t in range(SEQUENCE_LENGTH):
        w = np.array(buffer[-WINDOW:], dtype=np.float32)
        r = cvae_generate_next(w, int(regime_seq[t]), cvae_model, cvae_res)
        if not np.isfinite(r):
            r = 0.0
        buffer.append(float(r))
    returns_252 = np.array(buffer[-SEQUENCE_LENGTH:], dtype=np.float32)

    regime_prev = {
        name: np.sum(regime_seq == i) / SEQUENCE_LENGTH
        for i, name in enumerate(REGIME_NAMES)
    }
    print(
        f"  Generated sequence: std={np.std(returns_252):.6f}, "
        f"mean={np.mean(returns_252):.6f}"
    )
    print(f"  Regime prevalence: {regime_prev}")

    std_r = np.std(returns_252)
    if std_r < 1e-8:
        print("  BLOCKED: near-zero variance")
        return {}
    if np.any(~np.isfinite(returns_252)):
        print("  BLOCKED: NaN/Inf detected")
        return {}
    if std_r > 5.0:
        print("  WARNING: extreme variance")

    print("\n=== Running Layer 2 test suite (7 tests × 3 regimes) ===")
    suite_result = run_test_suite(returns_252, regime_seq, benchmark)
    print_test_suite(suite_result)

    return suite_result
