from pathlib import Path
import numpy as np


DATA_FILES = [
    "data_v19_clean_s42.npz",
    "data_v18_s42.npz",
    "data_v18_s7.npz",
]


def clean_filter(states, actions):
    speed = states[:, 0]
    heading = states[:, 1]
    front = states[:, 3]

    throttle = actions[:, 0]
    steering = actions[:, 1]

    mask = (
        (speed > 1.0) &
        (throttle > 0.15) &
        (front > 2.0) &
        (np.abs(steering) < 0.98) &
        (np.abs(heading) < 2.5)
    )

    return states[mask], actions[mask]


all_states = []
all_actions = []

for file in DATA_FILES:
    if not Path(file).exists():
        print(f"missing: {file}")
        continue

    d = np.load(file)
    states = d["states"].astype(np.float32)
    actions = d["actions"].astype(np.float32)

    states, actions = clean_filter(states, actions)

    print(f"{file}: kept {len(states)} samples")

    all_states.append(states)
    all_actions.append(actions)

states = np.vstack(all_states)
actions = np.vstack(all_actions)

print("merged states:", states.shape)
print("merged actions:", actions.shape)

np.savez(
    "data_v22_merged_clean.npz",
    states=states.astype(np.float32),
    actions=actions.astype(np.float32),
)

print("Saved data_v22_merged_clean.npz")