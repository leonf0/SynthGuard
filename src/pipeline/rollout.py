from typing import Callable

import numpy as np

from ..config import SEQUENCE_LENGTH, WINDOW


def autoregressive_generate(
    seed: np.ndarray,
    regime_sequence: np.ndarray,
    gen_fn: Callable[[np.ndarray, int], float],
    window: int = WINDOW,
    length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Roll a next-step generator forward `length` steps starting from `seed`."""
    buffer = list(seed)
    for t in range(length):
        w = np.array(buffer[-window:], dtype=np.float32)
        r = gen_fn(w, int(regime_sequence[t]))
        if not np.isfinite(r):
            r = 0.0
        buffer.append(float(r))
    return np.array(buffer[-length:], dtype=np.float32)
