import sys
sys.path.append('.')
import numpy as np
from drive2win.normalize import normalize_states

for filename in ["data_v2.npz", "data_v3.npz"]:
    d = np.load(filename)
    states = d["states"]
    actions = d["actions"]

    print(f"=== {filename} ===")
    print("Total samples:", len(states))
    print("Positive throttle:", np.sum(actions[:, 0] > 0))
    print("Negative throttle (braking/reversing):", np.sum(actions[:, 0] < 0))
    print("Zero throttle:", np.sum(actions[:, 0] == 0))
    print("Steer left (< 0):", np.sum(actions[:, 1] < 0))
    print("Steer right (> 0):", np.sum(actions[:, 1] > 0))
    print("Steer straight (= 0):", np.sum(actions[:, 1] == 0))

    norm_states = normalize_states(states)
    front_ray = norm_states[:, 3]
    print("Avg front ray:", np.mean(front_ray))
    print("Avg front ray when throttle > 0.5:", np.mean(front_ray[actions[:, 0] > 0.5]))
    print("Avg front ray when throttle <= 0:", np.mean(front_ray[actions[:, 0] <= 0]))
    print()
