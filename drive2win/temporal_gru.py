import torch
import numpy as np
from drive2win.normalize import normalize_states

def make_policy(weights_path: str):
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from train_gru import NavGRU

    model = NavGRU()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    
    from collections import deque
    history = deque(maxlen=16)

    from drive2win.normalize import sensors_to_input

    def policy(state: dict) -> tuple[float, float]:
        # state is {"sensors": {...}, ...}
        x_norm = sensors_to_input(state["sensors"])
        history.append(x_norm)
        
        # If we don't have 16 frames yet, just pad with the first frame
        while len(history) < 16:
            history.append(x_norm)
            
        # Create tensor of shape (1, 16, 14)
        x_seq = np.stack(list(history))
        x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # Pass the 16 frames with h=None, exactly like training!
            out, _ = model(x_tensor, None)
            
        y = out[0].numpy()
        return (
            float(np.clip(y[0], -1.0, 1.0)),
            float(np.clip(y[1], -1.0, 1.0))
        )

    return policy
