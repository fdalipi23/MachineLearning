import numpy as np

d2 = np.load("data_v2.npz")
d3 = np.load("data_v3.npz")

states = np.concatenate([d2["states"], d3["states"]], axis=0)
actions = np.concatenate([d2["actions"], d3["actions"]], axis=0)
# positions are just for plotting, we can concatenate them too
if "positions" in d2.files and "positions" in d3.files:
    positions = np.concatenate([d2["positions"], d3["positions"]], axis=0)
else:
    positions = d2["positions"] if "positions" in d2.files else d3["positions"]

np.savez("data_v2_v3.npz", states=states, actions=actions, positions=positions, seed=d3.get("seed", 42))
print("Merged dataset shape:", states.shape, actions.shape)
