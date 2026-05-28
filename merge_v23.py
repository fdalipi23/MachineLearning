from pathlib import Path
import numpy as np


FILES = [
    "data_v19_clean_s42.npz",
    "data_v23_clean_s7.npz",
]


def filter_clean(states, actions):
    speed = states[:, 0]
    heading = states[:, 1]
    front = states[:, 3]

    throttle = actions[:, 0]
    steering = actions[:, 1]

    mask = (
        (throttle > 0.30) &
        (speed > 1.0) &
        (front > 2.0) &
        (np.abs(heading) < 2.8) &
        (np.abs(steering) < 0.98)
    )

    return states[mask], actions[mask]


def main():
    all_states = []
    all_actions = []

    for file in FILES:
        path = Path(file)

        if not path.exists():
            print(f"missing: {file}")
            continue

        data = np.load(path)

        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)

        clean_states, clean_actions = filter_clean(states, actions)

        print(f"{file}: kept {len(clean_states)} / {len(states)} samples")

        all_states.append(clean_states)
        all_actions.append(clean_actions)

    if not all_states:
        raise RuntimeError("No datasets found. Check file names.")

    merged_states = np.vstack(all_states)
    merged_actions = np.vstack(all_actions)

    print("merged states:", merged_states.shape)
    print("merged actions:", merged_actions.shape)

    np.savez(
        "data_v23_clean_merged.npz",
        states=merged_states.astype(np.float32),
        actions=merged_actions.astype(np.float32),
    )

    print("Saved data_v23_clean_merged.npz")


if __name__ == "__main__":
    main()