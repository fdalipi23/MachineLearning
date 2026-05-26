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

N_FEATURES = 12
N_ACTIONS = 2

SPD_MAX = 25.0
DIST_MAX = 100.0
RAY_MAX = 50.0


def normalize_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)

    single = states.ndim == 1
    if single:
        states = states[None, :]

    out = np.zeros((states.shape[0], 12), dtype=np.float32)

    out[:, 0] = states[:, 0] / SPD_MAX
    out[:, 1] = states[:, 1] / np.pi
    out[:, 2] = states[:, 2] / DIST_MAX
    out[:, 3:11] = states[:, 3:11] / RAY_MAX
    out[:, 11] = states[:, 11]

    out = np.clip(out, -1.5, 1.5)

    return out[0] if single else out


def sensors_to_input(sensors: dict) -> np.ndarray:
    nav = sensors.get("navigation") or {}
    ground = sensors.get("ground") or {}

    speed = sensors.get("speed", 0.0)
    heading_error = sensors.get("heading_error", nav.get("heading_error", 0.0))
    checkpoint_distance = sensors.get("checkpoint_distance", nav.get("distance", 0.0))
    rays = sensors.get("rays", [50.0] * 8)
    ground_friction = sensors.get("ground_friction", ground.get("friction", 1.0))

    if rays is None or len(rays) < 8:
        rays = [50.0] * 8

    raw = np.array(
        [
            speed,
            heading_error,
            checkpoint_distance,
            rays[0],
            rays[1],
            rays[2],
            rays[3],
            rays[4],
            rays[5],
            rays[6],
            rays[7],
            ground_friction,
        ],
        dtype=np.float32,
    )

    return normalize_states(raw)


def clip_action(action):
    action = np.asarray(action, dtype=np.float32)
    throttle = float(np.clip(action[0], -1.0, 1.0))
    steering = float(np.clip(action[1], -1.0, 1.0))
    return throttle, steering