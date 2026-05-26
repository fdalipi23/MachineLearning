from __future__ import annotations
import numpy as np


FEATURE_NAMES = [
    "speed",
    "heading_error",
    "checkpoint_distance",
    "ray_0_front",
    "ray_1_+45",
    "ray_2_+90",
    "ray_3_+135",
    "ray_4_back",
    "ray_5_-135",
    "ray_6_-90",
    "ray_7_-45",
    "ground_friction",
]


def compare_inputs(
    live_input: np.ndarray,
    recorded_input: np.ndarray
):

    print("\n=== INPUT CHECK ===\n")

    for i, name in enumerate(FEATURE_NAMES):

        live_val = float(live_input[i])
        rec_val = float(recorded_input[i])

        diff = live_val - rec_val

        print(
            f"{i:02d} | "
            f"{name:22s} | "
            f"live={live_val:8.3f} | "
            f"recorded={rec_val:8.3f} | "
            f"diff={diff:8.3f}"
        )

    max_diff = np.max(
        np.abs(live_input - recorded_input)
    )

    print("\nMaximum difference:")
    print(f"{max_diff:.6f}")


if __name__ == "__main__":

    # fake example vectors for testing
    live = np.random.randn(12).astype(np.float32)

    recorded = (
        live
        + np.random.normal(0, 0.01, 12)
    ).astype(np.float32)

    compare_inputs(live, recorded)