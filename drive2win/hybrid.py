from drive2win.benchmark import make_mlp_policy
import numpy as np

def make_policy(weights_path: str):
    # Load the proven v11 MLP policy
    base_policy = make_mlp_policy(weights_path)
    
    stuck_frames = 0

    def policy(state: dict) -> tuple[float, float]:
        nonlocal stuck_frames
        
        throttle, steering = base_policy(state)
        
        # --- Smart Recovery Heuristic ---
        # If the car is barely moving, increment stuck counter
        speed = state["sensors"]["speed"]
        if abs(speed) < 0.5:
            stuck_frames += 1
        else:
            stuck_frames = max(0, stuck_frames - 1)
            
        # If stuck for more than 15 frames (0.75 seconds), initiate reverse protocol!
        if stuck_frames > 15:
            # We are completely stuck on a wall!
            # Override neural network: Reverse hard and steer away
            override_throttle = -1.0
            
            # If there's an obstacle on the left (ray 1/2), steer right
            rays = state["sensors"]["rays"]
            left_obstacle = rays[1] + rays[2]
            right_obstacle = rays[6] + rays[7]
            
            if left_obstacle < right_obstacle:
                override_steering = 1.0
            else:
                override_steering = -1.0
                
            return float(override_throttle), float(override_steering)
            
        # Normal driving
        return float(throttle), float(steering)

    return policy
