from __future__ import annotations

import numpy as np

RAW_FEATURES = 6


def enrich(X: np.ndarray) -> np.ndarray:
    extra = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return np.hstack([X, extra])
