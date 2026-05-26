import numpy as np

for path in ["data_v2.npz", "data_v3.npz"]:
    print(f"=== {path} ===")
    d = np.load(path, allow_pickle=False)
    print("Files:", d.files)
    states = d["states"]
    actions = d["actions"]
    print("states shape:", states.shape)
    print("actions shape:", actions.shape)
    print("seed:", d.get("seed", "not found"))
    print("actions mean:", np.mean(actions, axis=0))
    print("actions min:", np.min(actions, axis=0))
    print("actions max:", np.max(actions, axis=0))
    # Check if there are any NaN/Inf
    print("states nan count:", np.isnan(states).sum())
    print("actions nan count:", np.isnan(actions).sum())
