from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action


def make_policy(weights_path: str):
    weights = nn_mod.load(weights_path)

    memory = {
        "prev_action": np.zeros(2, dtype=np.float32),
        "stuck_frames": 0,
        "escape_reverse": 0,
        "escape_forward": 0,
        "escape_dir": 1.0,
    }

    def get_heading_error(sensors):
        nav = sensors.get("navigation") or {}
        return float(
            sensors.get(
                "heading_error",
                nav.get("heading_error", 0.0)
            )
        )

    def get_checkpoint_distance(sensors):
        nav = sensors.get("navigation") or {}
        return float(
            sensors.get(
                "checkpoint_distance",
                nav.get("distance", 999.0)
            )
        )

    def choose_escape_dir(rays):
        # +1 means steer right, -1 means steer left
        front_left = rays[1] if len(rays) > 1 else 50.0
        left = rays[2] if len(rays) > 2 else 50.0
        rear_left = rays[3] if len(rays) > 3 else 50.0

        rear_right = rays[5] if len(rays) > 5 else 50.0
        right = rays[6] if len(rays) > 6 else 50.0
        front_right = rays[7] if len(rays) > 7 else 50.0

        left_score = front_left + left + rear_left
        right_score = front_right + right + rear_right

        if left_score > right_score:
            return -1.0
        return 1.0

    def policy(state):
        sensors = state.get("sensors") or {}

        x = sensors_to_input(sensors)
        action = np.array(nn_mod.forward(x, weights), dtype=np.float32)

        throttle, steering = clip_action(action)

        heading_error = get_heading_error(sensors)
        checkpoint_distance = get_checkpoint_distance(sensors)
        speed = float(sensors.get("speed", 0.0))

        rays = sensors.get("rays", [50.0] * 8)
        if rays is None or len(rays) < 8:
            rays = [50.0] * 8

        front = float(rays[0])
        front_left = float(rays[1])
        front_right = float(rays[7])
        front_arc = min(front, front_left, front_right)

        wall_in_front = front_arc < 5.0

        # Escape phase 1: reverse away from wall
        if memory["escape_reverse"] > 0:
            memory["escape_reverse"] -= 1
            d = memory["escape_dir"]

            if memory["escape_reverse"] == 0:
                memory["escape_forward"] = 22

            return -1.0, 0.9 * d

        # Escape phase 2: push forward in opposite direction
        if memory["escape_forward"] > 0:
            memory["escape_forward"] -= 1
            d = memory["escape_dir"]
            return 0.85, -0.7 * d

        # Smooth raw neural output
        prev = memory["prev_action"]
        alpha = 0.75

        throttle = alpha * throttle + (1.0 - alpha) * prev[0]
        steering = alpha * steering + (1.0 - alpha) * prev[1]

        # Checkpoint steering correction
        if checkpoint_distance < 70:
            steering = steering - 0.55 * heading_error

        # Wall avoidance before getting fully stuck
        if front < 6.0:
            avoid_dir = choose_escape_dir(rays)
            steering = 0.85 * avoid_dir
            throttle = min(throttle, 0.45)

        elif front < 10.0:
            avoid_dir = choose_escape_dir(rays)
            steering = steering + 0.45 * avoid_dir
            throttle = min(throttle, 0.65)

        # Slow down when direction error is large
        if abs(heading_error) > 0.75:
            throttle = min(throttle, 0.55)

        if abs(heading_error) > 1.10:
            throttle = min(throttle, 0.35)

        # Detect real wall stuck
        if speed < 1.2 and wall_in_front:
            memory["stuck_frames"] += 1
        else:
            memory["stuck_frames"] = 0

        if memory["stuck_frames"] > 10:
            memory["stuck_frames"] = 0
            memory["escape_dir"] = choose_escape_dir(rays)
            memory["escape_reverse"] = 30
            memory["escape_forward"] = 0
            return -1.0, 0.9 * memory["escape_dir"]

        # Minimum movement
        if throttle < 0.20:
            throttle = 0.20

        throttle = float(np.clip(throttle, -1.0, 1.0))
        steering = float(np.clip(steering, -0.85, 0.85))

        memory["prev_action"][:] = [throttle, steering]

        print(
            f"speed={speed:.2f} "
            f"heading={heading_error:.2f} "
            f"front={front:.2f} "
            f"throttle={throttle:.2f} "
            f"steering={steering:.2f}"
        )

        return throttle, steering

    return policy