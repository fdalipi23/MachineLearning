from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action


def make_policy(weights_path: str):
    weights = nn_mod.load(weights_path)

    memory = {
        "prev_action": np.zeros(2, dtype=np.float32),
        "stuck_frames": 0,
        "escape_frames": 0,
    }

    def policy(state):
        sensors = state.get("sensors") or {}

        x = sensors_to_input(sensors)
        action = np.array(nn_mod.forward(x, weights), dtype=np.float32)

        throttle, steering = clip_action(action)

        # 1. Smooth steering/throttle so bot does not move randomly
        prev = memory["prev_action"]
        alpha = 0.75
        throttle = alpha * throttle + (1 - alpha) * prev[0]
        steering = alpha * steering + (1 - alpha) * prev[1]

        # 2. Correct steering using heading_error
        heading_error = sensors.get("heading_error", 0.0)
        checkpoint_distance = sensors.get("checkpoint_distance", 999.0)

        if checkpoint_distance < 60:
            steering = steering - 0.45 * heading_error

        # 3. Prevent very weak movement
        if throttle < 0.15:
            throttle = 0.25

        # 4. If stuck, reverse briefly and turn
        speed = sensors.get("speed", 0.0)

        if memory["escape_frames"] > 0:
            memory["escape_frames"] -= 1
            return -0.7, 0.6

        if speed < 0.3:
            memory["stuck_frames"] += 1
        else:
            memory["stuck_frames"] = 0

        if memory["stuck_frames"] > 25:
            memory["stuck_frames"] = 0
            memory["escape_frames"] = 20
            return -0.7, 0.6

        throttle = float(np.clip(throttle, -1.0, 1.0))
        steering = float(np.clip(steering, -0.9, 0.9))

        memory["prev_action"][:] = [throttle, steering]

        return throttle, steering

    return policy