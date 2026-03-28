from typing import Callable

import numpy as np
import pandas as pd

def estimate_transition_matrix(labels: pd.Series) -> np.ndarray:
    regime_to_idx = {name: i for i, name in enumerate(REGIME_NAMES)}
    indices = np.array([regime_to_idx[l] for l in labels if l in regime_to_idx])
    T = np.zeros((N_REGIMES, N_REGIMES))
    for t in range(len(indices) - 1):
        T[indices[t], indices[t + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums
    return T


def make_markov_step(transition_matrix: np.ndarray) -> Callable[[int], int]:
    def step(current_regime: int) -> int:
        return np.random.choice(N_REGIMES, p=transition_matrix[current_regime])
    return step


def generate_regime_sequence(
    transition_matrix: np.ndarray, initial_regime: int, length: int = SEQUENCE_LENGTH
) -> np.ndarray:
    step = make_markov_step(transition_matrix)
    regimes = np.empty(length, dtype=int)
    regimes[0] = initial_regime
    for t in range(1, length):
        regimes[t] = step(regimes[t - 1])
    return regimes
