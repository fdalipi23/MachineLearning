from __future__ import annotations
import numpy as np


class ActionSmoother:
    """
    Simple exponential moving average for actions.
    Helps reduce steering jitter.
    """

    def __init__(self, policy, alpha: float = 0.7):
        self.policy = policy
        self.alpha = alpha
        self.prev = np.zeros(2, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raw = self.policy(x)

        smooth = (
            self.alpha * raw
            + (1.0 - self.alpha) * self.prev
        )

        self.prev = smooth.astype(np.float32)

        return self.prev