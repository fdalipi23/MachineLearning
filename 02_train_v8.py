"""Fast v8 training — plain deep MLP, 14 inputs, steering-weighted loss.

Trains in ~90 seconds on CPU. No temporal windowing (removes the complexity
that was causing the previous model to learn a constant-steering bias).

Run:  python 02_train_v8.py --tag v8
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from drive2win.normalize import normalize_states
from drive2win import viz


# ── Model ──────────────────────────────────────────────────────────────────
class DeepMLP(nn.Module):
    """14-input deep MLP: 14 -> 256 -> 128 -> 64 -> 2"""
    def __init__(self, n_in: int = 14, h=(256, 128, 64), n_out: int = 2, dropout=0.1):
        super().__init__()
        layers: list[nn.Module] = []
        sizes = [n_in, *h]
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
        layers += [nn.Linear(sizes[-1], n_out), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Weighted loss ──────────────────────────────────────────────────────────
class BalancedMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        sq = (pred - target) ** 2
        # target[:, 0] is throttle. If < 0, the human is braking/reversing.
        # That is rare but CRITICAL for not getting stuck.
        throttle_w = torch.where(target[:, 0] < 0.0, 10.0, 1.0)
        
        # We also still care about steering (target[:, 1]), give it weight 2.0
        steering_w = torch.full_like(throttle_w, 2.0)
        
        w = torch.stack([throttle_w, steering_w], dim=1)
        return (sq * w).mean()


# ── Data helpers ───────────────────────────────────────────────────────────
def load_and_mirror(files):
    all_s, all_a = [], []
    for f in files:
        try:
            d = np.load(f, allow_pickle=False)
            all_s.append(d["states"])
            all_a.append(d["actions"])
            print(f"  {f}: {d['states'].shape[0]} samples")
        except FileNotFoundError:
            print(f"  {f}: not found, skipping")

    states = np.concatenate(all_s)
    actions = np.concatenate(all_a)
    print(f"  total before mirror: {len(states)}")

    # Mirror left-right to balance steering
    sm = states.copy(); am = actions.copy()
    sm[:, 1] = -states[:, 1]                          # flip heading_error
    for i, j in [(4, 10), (5, 9), (6, 8)]:            # swap symmetric rays
        sm[:, i], sm[:, j] = states[:, j].copy(), states[:, i].copy()
    am[:, 1] = -actions[:, 1]                          # flip steering

    states = np.concatenate([states, sm])
    actions = np.concatenate([actions, am])
    print(f"  total after  mirror: {len(states)}")

    # Oversample the braking/reversing data 15x so it learns to get unstuck!
    brake_idx = np.where(actions[:, 0] < 0.0)[0]
    brake_states = states[brake_idx]
    brake_actions = actions[brake_idx]
    
    states = np.concatenate([states] + [brake_states]*15)
    actions = np.concatenate([actions] + [brake_actions]*15)
    print(f"  total after braking boost: {len(states)}")

    return states, actions


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",      default="v8")
    ap.add_argument("--epochs",   type=int,   default=400)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--batch",    type=int,   default=256)
    ap.add_argument("--patience", type=int,   default=60)
    args = ap.parse_args()

    print("Loading data …")
    files = ["data_v2.npz", "data_v3.npz", "data_v5.npz", "data_v6.npz"]
    states_raw, actions = load_and_mirror(files)

    # Normalize → (N, 14)
    X = normalize_states(states_raw).astype(np.float32)
    Y = actions.astype(np.float32)
    print(f"\nX: {X.shape}  Y: {Y.shape}")
    print(f"heading range  : [{X[:,1].min():+.3f}, {X[:,1].max():+.3f}]")
    print(f"steering range : [{Y[:,1].min():+.3f}, {Y[:,1].max():+.3f}]")

    # Correlation check
    corr = np.corrcoef(X[:, 1], Y[:, 1])[0, 1]
    print(f"heading->steering corr: {corr:+.4f}  (negative = correct convention)")

    # Train/val split
    Xt = torch.tensor(X); Yt = torch.tensor(Y)
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(Xt), generator=gen)
    nv = max(1, len(Xt) // 10)
    Xtr, Ytr = Xt[perm[nv:]], Yt[perm[nv:]]
    Xva, Yva = Xt[perm[:nv]], Yt[perm[:nv]]
    print(f"train: {len(Xtr)}  val: {len(Xva)}\n")

    model = DeepMLP()
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Model: 14->256->128->64->2  ({n_p:,} params)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=80, T_mult=2, eta_min=1e-5)
    loss_fn = BalancedMSE()

    best_val, best_w, no_imp = float("inf"), None, 0
    train_losses, val_losses = [], []

    print(f"Training {args.epochs} epochs (patience={args.patience}) …\n")
    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        ep_loss, nb = 0.0, 0
        for i in range(0, len(Xtr), args.batch):
            b = idx[i:i+args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva), Yva).item()

        train_losses.append(ep_loss / max(1, nb))
        val_losses.append(v)

        if v < best_val:
            best_val = v
            best_w = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train={train_losses[-1]:.4f}  val={v:.4f}  best={best_val:.4f}  lr={opt.param_groups[0]['lr']:.1e}")

        if no_imp >= args.patience:
            print(f"\nEarly stop at epoch {epoch}")
            break

    # Save
    model.load_state_dict(best_w)
    model.eval()
    torch.save(model.state_dict(), f"nav_{args.tag}.pt")
    print(f"[OK] Saved nav_{args.tag}.pt  (best val={best_val:.4f})")

    # Sanity check
    print("\nSanity check:")
    print("  scenario                    throttle   steering")
    from drive2win.normalize import sensors_to_input, clip_action

    def chk(speed, he, ray=50.0):
        s = {"speed": speed, "heading_error": he,
             "checkpoint_distance": 30.0, "rays": [ray]*8, "ground_friction": 1.0}
        x = torch.tensor(sensors_to_input(s)).unsqueeze(0)
        with torch.no_grad():
            y = model(x)[0].numpy()
        return clip_action(y)

    for name, sp, he, ray in [
        ("aligned (he=0)",       10, 0.0,  50),
        ("soft left he=+0.3",    10, 0.3,  50),
        ("hard left he=+1.0",    10, 1.0,  50),
        ("soft right he=-0.3",   10, -0.3, 50),
        ("hard right he=-1.0",   10, -1.0, 50),
        ("stopped aligned",       0, 0.0,  50),
        ("wall front ray=3m",     5, 0.0,   3),
    ]:
        t, s = chk(sp, he, ray)
        ok = "OK" if (abs(he) < 0.1 and abs(s) < 0.3) or (he > 0.2 and s < 0) or (he < -0.2 and s > 0) else "!!"
        print(f"  {ok} {name:<26} {t:+.3f}      {s:+.3f}")

    # Plots
    viz.plot_loss_curves(train_losses, val_losses, out=f"fig_loss_{args.tag}.png")
    viz.plot_action_histograms(Y, out=f"fig_actions_{args.tag}.png")
    viz.plot_heading_vs_steering(states_raw, Y, out=f"fig_heading_{args.tag}.png")


if __name__ == "__main__":
    main()
