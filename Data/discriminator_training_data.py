from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEED_WINDOW = 60
N_TOTAL = 20_000
N_REAL = 10_000
N_REAL_POOLED = 5_000
N_REAL_INDIVIDUAL = 5_000
N_MODEL_SYNTHETIC = 5_000
N_FAKE = 5_000
N_FAKE_METHODS = 12
N_MODELS = 6


def _regime_to_int(labels: pd.Series) -> pd.Series:
    mapping = {name: i for i, name in enumerate(REGIME_NAMES)}
    return labels.map(mapping)


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


def _autoregressive_generate(
    seed: np.ndarray,
    regime_sequence: np.ndarray,
    gen_fn: Callable[[np.ndarray, int], float],
    window: int = WINDOW,
) -> np.ndarray:

    buffer = list(seed)
    for t in range(SEQUENCE_LENGTH):
        w = np.array(buffer[-window:], dtype=np.float32)
        r = gen_fn(w, int(regime_sequence[t]))
        if not np.isfinite(r):
            r = 0.0
        buffer.append(float(r))
    return np.array(buffer[-SEQUENCE_LENGTH:], dtype=np.float32)


def _distribute_counts(total: int, n_buckets: int) -> List[int]:

    base = total // n_buckets
    remainder = total % n_buckets
    return [base + (1 if i < remainder else 0) for i in range(n_buckets)]


def _build_model_synthetic(
    train_all_dict: Dict,
    individual_returns: Dict[str, pd.Series],
    regime_labels: pd.Series,
    n: int = N_MODEL_SYNTHETIC,
) -> Tuple[np.ndarray, np.ndarray]:

    trans_matrix = train_all_dict["transition_matrix"]
    gbm_params = train_all_dict["gbm_params"]
    garch_params = train_all_dict["garch_params"]
    heston_params = train_all_dict["heston_params"]
    tft_model = train_all_dict["tft_model"]
    tft_res = train_all_dict["tft_residual_std"]
    cvae_model = train_all_dict["cvae_model"]
    cvae_res = train_all_dict["cvae_residual_std"]
    diff_model = train_all_dict["diffusion_model"]
    diff_res = train_all_dict["diffusion_residual_std"]

    generators: List[Tuple[str, Callable[[np.ndarray, int], float]]] = [
        ("gbm", lambda w, r: gbm_generate_next(w, r, gbm_params)),
        ("garch", lambda w, r: garch_generate_next(w, r, garch_params)),
        ("heston", lambda w, r: heston_generate_next(w, r, heston_params)),
        ("tft", lambda w, r: tft_generate_next(w, r, tft_model, tft_res)),
        ("cvae", lambda w, r: cvae_generate_next(w, r, cvae_model, cvae_res)),
        ("diffusion", lambda w, r: diffusion_generate_next(w, r, diff_model, diff_res)),
    ]

    regime_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
    regime_labels_int = regime_labels.map(regime_to_idx).dropna().astype(int)

    valid_tickers = []
    for t, s in individual_returns.items():
        common = s.index.intersection(regime_labels_int.index)
        if len(common) >= SEED_WINDOW + 1:
            valid_tickers.append(t)

    per_model = per_model_map = {
        "gbm": 1000,
        "garch": 1000,
        "heston": 1000,
        "tft": 900,
        "cvae": 1000,
        "diffusion": 100,
}

    returns_out = np.empty((n, SEQUENCE_LENGTH), dtype=np.float32)
    regimes_out = np.empty((n, SEQUENCE_LENGTH), dtype=np.float32)
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
            regime_seq = generate_regime_sequence(trans_matrix, initial_regime, SEQUENCE_LENGTH)
            ret_seq = _autoregressive_generate(seed, regime_seq, gen_fn)

            if np.any(~np.isfinite(ret_seq)):
                continue

            returns_out[idx] = ret_seq
            regimes_out[idx] = regime_seq.astype(np.float32)
            idx += 1
            generated += 1

    return returns_out, regimes_out


def _fake_constant_drift(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    drift = np.random.uniform(0.0005, 0.005)
    sigma = np.random.uniform(0.00005, 0.0005)
    return (np.full(n, drift) + np.random.randn(n) * sigma).astype(np.float32)


def _fake_iid_gaussian(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    sigma = np.random.uniform(0.005, 0.03)
    return (np.random.randn(n) * sigma).astype(np.float32)


def _fake_perfect_alternation(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    amplitude = np.random.uniform(0.003, 0.02)
    signs = np.array([(-1) ** i for i in range(n)], dtype=np.float64)
    if np.random.rand() < 0.3:
        for i in range(1, n - 1):
            if np.random.rand() < 0.15:
                signs[i] = signs[i + 1] if np.random.rand() < 0.5 else signs[i - 1]
    return (signs * amplitude).astype(np.float32)


def _fake_linear_trend(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    start_val = np.random.uniform(-0.01, -0.001)
    end_val = np.random.uniform(0.001, 0.01)
    return np.linspace(start_val, end_val, n).astype(np.float32)


def _fake_sinusoidal(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    amplitude = np.random.uniform(0.005, 0.02)
    period = np.random.uniform(10, 40)
    t = np.arange(n)
    return (amplitude * np.sin(2 * np.pi * t / period)).astype(np.float32)


def _fake_non_negative(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    sigma = np.random.uniform(0.005, 0.03)
    return np.abs(np.random.randn(n) * sigma).astype(np.float32)


def _fake_frequent_jumps(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    magnitude = np.random.uniform(0.05, 0.15)
    freq = int(np.random.uniform(3, 10))
    seq = np.zeros(n, dtype=np.float32)
    for i in range(0, n, max(freq, 1)):
        seq[i] = magnitude * np.random.choice([-1.0, 1.0])
    return seq


def _fake_piecewise_deterministic(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    boundaries = sorted(np.random.choice(range(1, n - 1), size=2, replace=False))
    segments = [0] + list(boundaries) + [n]
    seq = np.empty(n, dtype=np.float32)
    for i in range(3):
        val = np.random.uniform(-0.005, 0.005)
        seq[segments[i] : segments[i + 1]] = val
    return seq


def _fake_quantized(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    n_levels = np.random.randint(3, 6)
    levels = np.random.uniform(-0.02, 0.02, size=n_levels).astype(np.float32)
    return np.random.choice(levels, size=n).astype(np.float32)


def _fake_exploding_variance(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    base_sigma = np.random.uniform(0.0005, 0.002)
    growth = np.random.uniform(0.00005, 0.0002)
    sigmas = base_sigma + growth * np.arange(n)
    return (np.random.randn(n) * sigmas).astype(np.float32)


def _fake_mechanical_mean_reversion(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    coeff = np.random.uniform(-0.99, -0.80)
    x = np.random.uniform(-0.01, 0.01)
    seq = np.empty(n, dtype=np.float32)
    for i in range(n):
        seq[i] = x
        x = coeff * x
    return seq


def _fake_copy_pasted_blocks(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    block_len = np.random.randint(20, 81)
    block = (np.random.randn(block_len) * 0.01).astype(np.float32)
    n_repeats = int(np.ceil(n / block_len))
    tiled = np.tile(block, n_repeats)
    return tiled[:n]


FAKE_GENERATORS = [
    _fake_constant_drift,
    _fake_iid_gaussian,
    _fake_perfect_alternation,
    _fake_linear_trend,
    _fake_sinusoidal,
    _fake_non_negative,
    _fake_frequent_jumps,
    _fake_piecewise_deterministic,
    _fake_quantized,
    _fake_exploding_variance,
    _fake_mechanical_mean_reversion,
    _fake_copy_pasted_blocks,
]


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

    perm = np.random.permutation(N_TOTAL)
    X_returns, X_regimes, y = X_returns[perm], X_regimes[perm], y[perm]

    print(f"=== Discriminator dataset: {X_returns.shape[0]} instances, "
          f"class balance: {y.mean():.2%} synthetic ===")
    return X_returns, X_regimes, y
