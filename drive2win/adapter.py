from drive2win.benchmark import make_mlp_policy
import numpy as np

def make_policy(weights_path: str):
    # Load the standard MLP policy that we trained for v11
    base_policy = make_mlp_policy(weights_path)

    def policy(state: dict) -> tuple[float, float]:
        # Get the neural network's original prediction
        throttle, steering = base_policy(state)
        
        # APPLY YOUR V5 CONSTRAINTS!
        # 1. Never reverse, never stop completely (min throttle 0.25)
        throttle = float(np.clip(throttle, 0.25, 1.0))
        
        # 2. Prevent wild spinning (clamp steering)
        steering = float(np.clip(steering, -0.8, 0.8))
        
        return throttle, steering

    return policy
