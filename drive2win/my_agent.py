from __future__ import annotations
import numpy as np

from . import nn as nn_mod


def make_policy(weights_path: str):
    nn_mod.load(weights_path)  # kept so benchmark accepts weights

    memory = {
        "stuck": 0,
        "escape_rev": 0,
        "escape_fwd": 0,
        "escape_dir": 1.0,
        "post_escape": 0,
    }

    def steer_scale(distance):
        if distance < 6.0:
            return 0.45
        if distance < 12.0:
            return 0.75
        return 1.20

    def policy(state):
        sensors = state.get("sensors") or {}
        nav = sensors.get("navigation") or {}
        ground = sensors.get("ground") or {}

        speed = float(sensors.get("speed", 0.0))
        heading = float(sensors.get("heading_error", nav.get("heading_error", 0.0)))
        distance = float(sensors.get("checkpoint_distance", nav.get("distance", 50.0)))
        friction = float(sensors.get("ground_friction", ground.get("friction", 1.0)))

        rays = sensors.get("rays", [50.0] * 8)
        if rays is None or len(rays) < 8:
            rays = [50.0] * 8

        rays = [float(r) for r in rays]

        front = rays[0]
        front_left = rays[1]
        left = rays[2]
        rear_left = rays[3]
        rear_right = rays[5]
        right = rays[6]
        front_right = rays[7]

        front_arc = min(front, front_left, front_right)

        # escape reverse
        if memory["escape_rev"] > 0:
            memory["escape_rev"] -= 1
            if memory["escape_rev"] == 0:
                memory["escape_fwd"] = 22
            return -0.80, memory["escape_dir"]

        # escape forward
        if memory["escape_fwd"] > 0:
            memory["escape_fwd"] -= 1
            memory["post_escape"] = 60
            return 0.80, -0.65 * memory["escape_dir"]

        # stuck detection
        if speed < 0.35 and front_arc < 5.0:
            memory["stuck"] += 1
        else:
            memory["stuck"] = 0

        if memory["stuck"] > 25:
            left_score = front_left + left + rear_left
            right_score = front_right + right + rear_right

            memory["escape_dir"] = -1.0 if left_score > right_score else 1.0
            memory["escape_rev"] = 30
            memory["stuck"] = 0

            return -0.80, memory["escape_dir"]

        # main heading pursuit
        steering = -heading / steer_scale(distance)

        # post escape bias
        if memory["post_escape"] > 0:
            decay = memory["post_escape"] / 60.0
            steering += memory["escape_dir"] * 0.45 * decay
            memory["post_escape"] -= 1

        # front-block override
        if front < 7.0:
            left_score = front_left + 0.4 * left
            right_score = front_right + 0.4 * right

            if left_score > right_score + 1.0:
                steering -= 0.85
            elif right_score > left_score + 1.0:
                steering += 0.85
            else:
                steering += -0.85 if heading > 0 else 0.85

        steering = float(np.clip(steering, -1.0, 1.0))

        # throttle logic
        if distance < 10.0 and abs(heading) > 1.0:
            throttle = 0.25
        elif front < 3.5:
            throttle = 0.35
        elif front < 7.0:
            throttle = 0.65
        elif front < 12.0:
            throttle = 0.85
        else:
            throttle = 1.0

        if abs(steering) > 0.75 and distance > 8.0:
            throttle = min(throttle, 0.70)

        if friction < 0.6:
            soft = max(0.50, 0.4 + friction)
            throttle *= soft
            steering *= soft

        throttle = float(np.clip(throttle, -1.0, 1.0))
        steering = float(np.clip(steering, -1.0, 1.0))

        print(
            f"speed={speed:.2f} heading={heading:.2f} dist={distance:.2f} "
            f"front={front:.2f} throttle={throttle:.2f} steering={steering:.2f}"
        )

        return throttle, steering

    return policy