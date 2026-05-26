import torch
import numpy as np
from drive2win import temporal
from drive2win import normalize

def main():
    model = temporal.TemporalMLP()
    model.load_state_dict(torch.load("nav_v7.pt", map_location="cpu"))
    model.eval()

    torch.set_grad_enabled(False)

    print("=== Heading Error Sweep ===")
    print(f"{'Heading Error (rad)':>20} | {'Throttle':>10} | {'Steering':>10}")
    print("-" * 50)
    
    # Sweep heading error from -pi to +pi
    for h in np.linspace(-np.pi, np.pi, 13):
        # speed=0, heading_error=h, ckpt_dist=30, rays=50, friction=1
        raw_state = np.array([0.0, h, 30.0] + [50.0]*8 + [1.0], dtype=np.float32)
        norm_state = normalize.normalize_states(raw_state)
        
        # Stack it 4 times
        x_init = np.concatenate([norm_state, norm_state, norm_state, norm_state]).astype(np.float32)
        
        pred = model(torch.tensor(x_init).unsqueeze(0))[0].numpy()
        print(f"{h:20.4f} | {pred[0]:10.4f} | {pred[1]:10.4f}")

if __name__ == "__main__":
    main()
