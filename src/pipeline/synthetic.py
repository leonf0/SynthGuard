from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import N_REGIMES, REGIME_NAMES, SEED_WINDOW, SEQUENCE_LENGTH
from ..discriminator.fake import FAKE_GENERATORS
from ..generators import (
    cvae_generate_next,
    garch_generate_next,
    gbm_generate_next,
    generate_sequences,
    heston_generate_next,
    tft_generate_next,
)
from ..regimes import generate_regime_sequence
from .rollout import autoregressive_generate

N_TOTAL = 20_000
N_REAL = 10_000
N_REAL_POOLED = 5_000
N_REAL_INDIVIDUAL = 5_000
N_MODEL_SYNTHETIC = 5_000
N_FAKE = 5_000
N_FAKE_METHODS = 12


def _sample_real_window(
    returns_series: pd.Series,
    regime_labels_int: pd.Series,
    length: int = SEQUENCE_LENGTH,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    common = returns_series.index.intersection(regime_labels_int.index)
    if len(common) < length:
        return None
    ret_aligned = returns_series.loc[common].values
    reg_aligned = regime_labels_int.loc[common].values
    max_start = len(ret_aligned) - length
    if max_start < 1:
        return None
    start = np.random.randint(0, max_start)
    r = ret_aligned[start : start + length].astype(np.float32)
    g = reg_aligned[start : start + length]
    if np.any(np.isnan(r)) or np.any(np.isnan(g)):
        return None
    return r, g.astype(np.float32)


def _sample_real_sequences(
    source_series: List[pd.Series],
    regime_labels_int: pd.Series,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    returns_out = np.empty((n, SEQUENCE_LENGTH), dtype=np.float32)
    regimes_out = np.empty((n, SEQUENCE_LENGTH), dtype=np.float32)
    collected = 0
    while collected < n:
        series = source_series[np.random.randint(len(source_series))]
        result = _sample_real_window(series, regime_labels_int)
        if result is not None:
            returns_out[collected], regimes_out[collected] = result
            collected += 1
    return returns_out, regimes_out


def _build_autoregressive_synthetic(
    train_all_dict: Dict,
    individual_returns: Dict[str, pd.Series],
    regime_labels: pd.Series,
    per_model_map: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    trans_matrix = train_all_dict["transition_matrix"]
    gbm_params = train_all_dict["gbm_params"]
    garch_params = train_all_dict["garch_params"]
    heston_params = train_all_dict["heston_params"]
    tft_model = train_all_dict["tft_model"]
    tft_res = train_all_dict["tft_residual_std"]
    cvae_model = train_all_dict["cvae_model"]
    cvae_res = train_all_dict["cvae_residual_std"]

    generators: List[Tuple[str, Callable[[np.ndarray, int], float]]] = [
        ("gbm", lambda w, r: gbm_generate_next(w, r, gbm_params)),
        ("garch", lambda w, r: garch_generate_next(w, r, garch_params)),
        ("heston", lambda w, r: heston_generate_next(w, r, heston_params)),
        ("tft", lambda w, r: tft_generate_next(w, r, tft_model, tft_res)),
        ("cvae", lambda w, r: cvae_generate_next(w, r, cvae_model, cvae_res)),
    ]

    regime_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
    regime_labels_int = regime_labels.map(regime_to_idx).dropna().astype(int)

    valid_tickers = [
        t for t, s in individual_returns.items()
        if len(s.index.intersection(regime_labels_int.index)) >= SEED_WINDOW + 1
    ]

    total = sum(per_model_map[name] for name, _ in generators)
    returns_out = np.empty((total, SEQUENCE_LENGTH), dtype=np.float32)
    regimes_out = np.empty((total, SEQUENCE_LENGTH), dtype=np.float32)
    idx = 0

    for name, gen_fn in generators:
        count = per_model_map[name]
        print(f"  Generating {count} sequences with {name}...")
        generated = 0
        while generated < count:
            ticker = valid_tickers[np.random.randint(len(valid_tickers))]
            series = individual_returns[ticker]
            common_dates = series.index.intersection(regime_labels_int.index)
            series_aligned = series.loc[common_dates]
            reg_aligned = regime_labels_int.loc[common_dates]

            if len(series_aligned) < SEED_WINDOW + 1:
                continue
            start = np.random.randint(0, len(series_aligned) - SEED_WINDOW)
            seed = series_aligned.values[start : start + SEED_WINDOW].astype(np.float32)
            if np.any(~np.isfinite(seed)):
                continue

            initial_regime = int(reg_aligned.values[start + SEED_WINDOW - 1])
            regime_seq = generate_regime_sequence(
                trans_matrix, initial_regime, SEQUENCE_LENGTH
            )
            ret_seq = autoregressive_generate(seed, regime_seq, gen_fn)

            if np.any(~np.isfinite(ret_seq)):
                continue

            returns_out[idx] = ret_seq
            regimes_out[idx] = regime_seq.astype(np.float32)
            idx += 1
            generated += 1

    return returns_out, regimes_out


def _build_diffusion_synthetic(
    train_all_dict: Dict,
    regime_labels: pd.Series,
    n_total: int,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample full 252-length sequences directly from the sequence-level diffusion model.

    Regime allocation follows the empirical regime distribution. Each sampled sequence
    is assigned a constant regime path equal to its conditioning regime (the diffusion
    model conditions on a single regime, not a regime path).
    """
    diff_model = train_all_dict["diffusion_model"]
    regime_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
    regime_dist = regime_labels.map(regime_to_idx).dropna().astype(int)
    regime_probs = np.array(
        [(regime_dist == i).mean() for i in range(N_REGIMES)], dtype=np.float64
    )
    regime_probs /= regime_probs.sum()

    counts = np.random.multinomial(n_total, regime_probs)
    returns_out_parts: List[np.ndarray] = []
    regimes_out_parts: List[np.ndarray] = []

    for regime_idx, count in enumerate(counts):
        if count == 0:
            continue
        print(f"  [Diffusion] sampling {count} sequences for regime {REGIME_NAMES[regime_idx]}...")
        remaining = int(count)
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            seqs = generate_sequences(
                diff_model, regime=regime_idx, n_sequences=this_batch, cfg_weight=2.0
            )
            returns_out_parts.append(seqs.astype(np.float32))
            regimes_out_parts.append(
                np.full((this_batch, SEQUENCE_LENGTH), regime_idx, dtype=np.float32)
            )
            remaining -= this_batch

    returns_out = np.concatenate(returns_out_parts, axis=0)
    regimes_out = np.concatenate(regimes_out_parts, axis=0)

    # Keep only finite sequences; re-sample if we filter any out
    finite_mask = np.all(np.isfinite(returns_out), axis=1)
    if not finite_mask.all():
        n_bad = (~finite_mask).sum()
        print(f"  [Diffusion] discarding {n_bad} non-finite sequences")
        returns_out = returns_out[finite_mask]
        regimes_out = regimes_out[finite_mask]

    return returns_out, regimes_out


def _build_model_synthetic(
    train_all_dict: Dict,
    individual_returns: Dict[str, pd.Series],
    regime_labels: pd.Series,
    n: int = N_MODEL_SYNTHETIC,
) -> Tuple[np.ndarray, np.ndarray]:

    per_model_autoregressive = {
        "gbm": 1000,
        "garch": 1000,
        "heston": 1000,
        "tft": 900,
        "cvae": 1000,
    }
    n_diffusion = n - sum(per_model_autoregressive.values())  # 100 by default

    auto_r, auto_g = _build_autoregressive_synthetic(
        train_all_dict, individual_returns, regime_labels, per_model_autoregressive
    )
    diff_r, diff_g = _build_diffusion_synthetic(
        train_all_dict, regime_labels, n_diffusion
    )

    returns_out = np.concatenate([auto_r, diff_r], axis=0)
    regimes_out = np.concatenate([auto_g, diff_g], axis=0)

    # Pad / trim to exactly n in case diffusion dropped non-finite samples
    if len(returns_out) < n:
        deficit = n - len(returns_out)
        print(f"  Backfilling {deficit} sequences from GBM...")
        extra_r, extra_g = _build_autoregressive_synthetic(
            train_all_dict, individual_returns, regime_labels, {"gbm": deficit}
        )
        returns_out = np.concatenate([returns_out, extra_r], axis=0)
        regimes_out = np.concatenate([regimes_out, extra_g], axis=0)
    elif len(returns_out) > n:
        returns_out = returns_out[:n]
        regimes_out = regimes_out[:n]

    return returns_out, regimes_out


def _distribute_counts(total: int, n_buckets: int) -> List[int]:
    base = total // n_buckets
    remainder = total % n_buckets
    return [base + (1 if i < remainder else 0) for i in range(n_buckets)]


def _build_fake_synthetic(
    trans_matrix: np.ndarray, n: int = N_FAKE
) -> Tuple[np.ndarray, np.ndarray]:
    per_method = _distribute_counts(n, N_FAKE_METHODS)
    returns_out = np.empty((n, SEQUENCE_LENGTH), dtype=np.float32)
    regimes_out = np.empty((n, SEQUENCE_LENGTH), dtype=np.float32)
    idx = 0
    for gen_fn, count in zip(FAKE_GENERATORS, per_method):
        for _ in range(count):
            returns_out[idx] = gen_fn(SEQUENCE_LENGTH)
            initial_regime = np.random.randint(N_REGIMES)
            regimes_out[idx] = generate_regime_sequence(
                trans_matrix, initial_regime, SEQUENCE_LENGTH
            ).astype(np.float32)
            idx += 1
    return returns_out, regimes_out


def build_discriminator_dataset(
    train_all_dict: Dict,
    individual_returns: Dict[str, pd.Series],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    regime_labels = train_all_dict["regime_labels"]
    trans_matrix = train_all_dict["transition_matrix"]
    regime_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
    regime_labels_int = regime_labels.map(regime_to_idx).dropna().astype(int)

    print("=== Sampling 5,000 real sequences (pooled) ===")
    pooled_df = pd.DataFrame(individual_returns).dropna()
    pooled_series_list = [pooled_df[col] for col in pooled_df.columns]
    real_pooled_r, real_pooled_g = _sample_real_sequences(
        pooled_series_list, regime_labels_int, N_REAL_POOLED
    )

    print("=== Sampling 5,000 real sequences (individual) ===")
    individual_series_list = list(individual_returns.values())
    real_indiv_r, real_indiv_g = _sample_real_sequences(
        individual_series_list, regime_labels_int, N_REAL_INDIVIDUAL
    )

    print("=== Generating 5,000 model-based synthetic sequences ===")
    model_r, model_g = _build_model_synthetic(
        train_all_dict, individual_returns, regime_labels, N_MODEL_SYNTHETIC
    )

    print("=== Generating 5,000 obviously-fake sequences ===")
    fake_r, fake_g = _build_fake_synthetic(trans_matrix, N_FAKE)

    X_returns = np.concatenate([real_pooled_r, real_indiv_r, model_r, fake_r], axis=0)
    X_regimes = np.concatenate([real_pooled_g, real_indiv_g, model_g, fake_g], axis=0)
    y = np.concatenate([
        np.zeros(N_REAL, dtype=np.int64),
        np.ones(N_MODEL_SYNTHETIC + N_FAKE, dtype=np.int64),
    ])

    perm = np.random.permutation(len(y))
    X_returns, X_regimes, y = X_returns[perm], X_regimes[perm], y[perm]

    print(
        f"=== Discriminator dataset: {X_returns.shape[0]} instances, "
        f"class balance: {y.mean():.2%} synthetic ==="
    )
    return X_returns, X_regimes, y
