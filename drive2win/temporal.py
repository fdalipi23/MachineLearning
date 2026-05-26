"""Policy adapter for v8+ plain deep MLP (14 inputs -> 2 outputs).

For v7 temporal weights use --module drive2win.temporal_v7 if needed.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from .normalize import sensors_to_input, clip_action

_RUN_GAP_THRESHOLD = 1.5   # seconds — detect new benchmark run


class DeepMLP(nn.Module):
    """14 -> 256 -> 128 -> 64 -> 2  (matches 02_train_v8.py)"""
    def __init__(self, n_in=14, h=(256, 128, 64), n_out=2, dropout=0.1):
        super().__init__()
        layers: list[nn.Module] = []
        sizes = [n_in, *h]
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
        layers += [nn.Linear(sizes[-1], n_out), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_policy(weights_path: str):
    """Load weights and return a (state_dict) -> (throttle, steering) callable."""
    model = DeepMLP()
    try:
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
    except TypeError:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    last_call = [0.0]

    def policy(state):
        now = time.time()
        last_call[0] = now

        x = sensors_to_input(state["sensors"])          # (14,) float32
        
        # Create perfectly mirrored input
        x_mirrored = x.copy()
        x_mirrored[1] = -x[1]  # flip heading_error
        for i, j in [(4, 10), (5, 9), (6, 8)]: # swap symmetric rays
            x_mirrored[i], x_mirrored[j] = x[j], x[i]
        
        # Difference features are negated
        x_mirrored[12] = -x[12]
        x_mirrored[13] = -x[13]

        with torch.no_grad():
            inp = torch.stack([torch.from_numpy(x), torch.from_numpy(x_mirrored)])
            y = model(inp).numpy()
            
        # Average the predictions (but invert steering for the mirrored one)
        throttle = (y[0, 0] + y[1, 0]) / 2.0
        steering = (y[0, 1] - y[1, 1]) / 2.0
        
        return clip_action(np.array([throttle, steering]))

    return policy
