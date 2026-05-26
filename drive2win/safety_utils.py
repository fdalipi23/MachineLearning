import numpy as np


def limit_steering(steering: float,
                   limit: float = 0.85) -> float:

    return float(
        np.clip(steering, -limit, limit)
    )


def minimum_throttle(throttle: float,
                     minimum: float = 0.20) -> float:

    if throttle < minimum:
        return minimum

    return float(throttle)