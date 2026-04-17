"""Twelve obviously-fake sequence generators for discriminator training."""
import numpy as np

from ..config import SEQUENCE_LENGTH


def fake_constant_drift(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    drift = np.random.uniform(0.0005, 0.005)
    sigma = np.random.uniform(0.00005, 0.0005)
    return (np.full(n, drift) + np.random.randn(n) * sigma).astype(np.float32)


def fake_iid_gaussian(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    sigma = np.random.uniform(0.005, 0.03)
    return (np.random.randn(n) * sigma).astype(np.float32)


def fake_perfect_alternation(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    amplitude = np.random.uniform(0.003, 0.02)
    signs = np.array([(-1) ** i for i in range(n)], dtype=np.float64)
    if np.random.rand() < 0.3:
        for i in range(1, n - 1):
            if np.random.rand() < 0.15:
                signs[i] = signs[i + 1] if np.random.rand() < 0.5 else signs[i - 1]
    return (signs * amplitude).astype(np.float32)


def fake_linear_trend(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    start_val = np.random.uniform(-0.01, -0.001)
    end_val = np.random.uniform(0.001, 0.01)
    return np.linspace(start_val, end_val, n).astype(np.float32)


def fake_sinusoidal(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    amplitude = np.random.uniform(0.005, 0.02)
    period = np.random.uniform(10, 40)
    t = np.arange(n)
    return (amplitude * np.sin(2 * np.pi * t / period)).astype(np.float32)


def fake_non_negative(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    sigma = np.random.uniform(0.005, 0.03)
    return np.abs(np.random.randn(n) * sigma).astype(np.float32)


def fake_frequent_jumps(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    magnitude = np.random.uniform(0.05, 0.15)
    freq = int(np.random.uniform(3, 10))
    seq = np.zeros(n, dtype=np.float32)
    for i in range(0, n, max(freq, 1)):
        seq[i] = magnitude * np.random.choice([-1.0, 1.0])
    return seq


def fake_piecewise_deterministic(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    boundaries = sorted(np.random.choice(range(1, n - 1), size=2, replace=False))
    segments = [0] + list(boundaries) + [n]
    seq = np.empty(n, dtype=np.float32)
    for i in range(3):
        val = np.random.uniform(-0.005, 0.005)
        seq[segments[i] : segments[i + 1]] = val
    return seq


def fake_quantized(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    n_levels = np.random.randint(3, 6)
    levels = np.random.uniform(-0.02, 0.02, size=n_levels).astype(np.float32)
    return np.random.choice(levels, size=n).astype(np.float32)


def fake_exploding_variance(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    base_sigma = np.random.uniform(0.0005, 0.002)
    growth = np.random.uniform(0.00005, 0.0002)
    sigmas = base_sigma + growth * np.arange(n)
    return (np.random.randn(n) * sigmas).astype(np.float32)


def fake_mechanical_mean_reversion(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    coeff = np.random.uniform(-0.99, -0.80)
    x = np.random.uniform(-0.01, 0.01)
    seq = np.empty(n, dtype=np.float32)
    for i in range(n):
        seq[i] = x
        x = coeff * x
    return seq


def fake_copy_pasted_blocks(n: int = SEQUENCE_LENGTH) -> np.ndarray:
    block_len = np.random.randint(20, 81)
    block = (np.random.randn(block_len) * 0.01).astype(np.float32)
    n_repeats = int(np.ceil(n / block_len))
    return np.tile(block, n_repeats)[:n]


FAKE_GENERATORS = [
    fake_constant_drift,
    fake_iid_gaussian,
    fake_perfect_alternation,
    fake_linear_trend,
    fake_sinusoidal,
    fake_non_negative,
    fake_frequent_jumps,
    fake_piecewise_deterministic,
    fake_quantized,
    fake_exploding_variance,
    fake_mechanical_mean_reversion,
    fake_copy_pasted_blocks,
]
