"""Step 2 (Temporal) — Train a Temporal MLP in PyTorch.

Run:  python 02_train_temporal.py --data data_v6.npz --tag v7 --epochs 400
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from drive2win.normalize import normalize_states, FEATURE_NAMES
from drive2win import viz

K = 4  # history window length

def make_windowed(states_norm, K=4):
    """Stack K frames of normalized state vectors."""
    N = len(states_norm)
    pad = np.repeat(states_norm[:1], K - 1, axis=0)
    s = np.concatenate([pad, states_norm], axis=0)
    windows = np.stack([s[i:i+K].reshape(-1) for i in range(N)])
    return windows.astype(np.float32)

def mirror_data(states, actions):
    """Create a mirrored copy of the dataset to balance left/right actions."""
    states_mirrored = states.copy()
    actions_mirrored = actions.copy()
    
    # 1. Flip heading error
    states_mirrored[:, 1] = -states[:, 1]
    
    # 2. Swap rays left/right:
    # ray_1 (index 4) <-> ray_7 (index 10)
    states_mirrored[:, 4] = states[:, 10]
    states_mirrored[:, 10] = states[:, 4]
    
    # ray_2 (index 5) <-> ray_6 (index 9)
    states_mirrored[:, 5] = states[:, 9]
    states_mirrored[:, 9] = states[:, 5]
    
    # ray_3 (index 6) <-> ray_5 (index 8)
    states_mirrored[:, 6] = states[:, 8]
    states_mirrored[:, 8] = states[:, 6]
    
    # 3. Flip steering action
    actions_mirrored[:, 1] = -actions[:, 1]
    
    # Concatenate original and mirrored data
    states_combined = np.concatenate([states, states_mirrored], axis=0)
    actions_combined = np.concatenate([actions, actions_mirrored], axis=0)
    
    return states_combined, actions_combined

class TemporalMLP(nn.Module):
    def __init__(self, n_in=56, h=(128, 64, 32), n_out=2):
        super().__init__()
        layers = []
        sizes = [n_in, *h]
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.LeakyReLU(0.1)]
        layers += [nn.Linear(sizes[-1], n_out), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v6.npz",
                    help="Dataset file from 01_collect.py")
    ap.add_argument("--tag", default="v7",
                    help="Output suffix (nav_<tag>.pt, fig_*_<tag>.png)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    states_loaded, actions_loaded = d["states"], d["actions"]
    
    # Mirror the dataset to balance left/right actions and eliminate sign bias
    states_raw, actions = mirror_data(states_loaded, actions_loaded)
    
    print(f"raw states  : {states_raw.shape}")
    print(f"raw actions : {actions.shape}")

    # Inspect dataset ranges
    print("\nfeature ranges (raw):")
    for i, name in enumerate(FEATURE_NAMES[:states_raw.shape[1]]):
        col = states_raw[:, i]
        print(f"  {name:>20s}: [{col.min():+7.2f}, {col.max():+7.2f}]   "
              f"mean={col.mean():+.2f}  std={col.std():.2f}")

    # Normalize states (shape N, 14)
    states_norm = normalize_states(states_raw)
    
    # Stack history (shape N, 56)
    X_win = make_windowed(states_norm, K=K)
    Y = actions.astype(np.float32)
    print(f"\nWindowed X shape: {X_win.shape}")
    print(f"X range : [{X_win.min():+.2f}, {X_win.max():+.2f}]")
    print(f"Y range : [{Y.min():+.2f}, {Y.max():+.2f}]")

    # Set up PyTorch tensors
    X_t = torch.tensor(X_win, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    # 90/10 split
    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X_t), generator=generator)
    n_val = max(1, len(X_t) // 10)
    Xtr, Ytr = X_t[perm[n_val:]], Y_t[perm[n_val:]]
    Xva, Yva = X_t[perm[:n_val]], Y_t[perm[:n_val]]

    # Initialize model, loss and optimizer
    model = TemporalMLP(n_in=14 * K, h=(128, 64, 32), n_out=2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_weights = None

    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        ep_loss, n_b = 0.0, 0
        for i in range(0, len(Xtr), args.batch):
            b = idx[i:i+args.batch]
            opt.zero_grad()
            pred = model(Xtr[b])
            loss = loss_fn(pred, Ytr[b])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1
        
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva), Yva).item()
        
        train_losses.append(ep_loss / max(1, n_b))
        val_losses.append(v)

        if v < best_val:
            best_val = v
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train={train_losses[-1]:.4f}  val={v:.4f}  best={best_val:.4f}")

    # Save model and plots
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"nav_{args.tag}.pt")
    print(f"\nSaved nav_{args.tag}.pt")

    viz.plot_loss_curves(train_losses, val_losses, out=f"fig_loss_{args.tag}.png")
    viz.plot_action_histograms(actions, out=f"fig_actions_{args.tag}.png")
    viz.plot_heading_vs_steering(states_raw, actions, out=f"fig_heading_{args.tag}.png")

if __name__ == "__main__":
    main()
