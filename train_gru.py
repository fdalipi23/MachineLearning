import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from drive2win.normalize import normalize_states, N_FEATURES, N_ACTIONS

class NavGRU(nn.Module):
    def __init__(self, n_in=N_FEATURES, hidden=128, n_out=N_ACTIONS):
        super().__init__()
        # GRU expects input of shape (batch, seq_len, features)
        self.gru = nn.GRU(n_in, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_out),
            nn.Tanh()
        )

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        # We only care about the prediction at the last time step of the sequence
        return self.fc(out[:, -1, :]), h

class SequenceDataset(Dataset):
    def __init__(self, states, actions, seq_len=16):
        self.seq_len = seq_len
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return max(0, len(self.states) - self.seq_len)

    def __getitem__(self, idx):
        x = self.states[idx:idx + self.seq_len]
        y = self.actions[idx + self.seq_len - 1]
        return x, y

def load_data():
    all_states, all_actions = [], []
    # Load all available data to make up for the lost v12!
    for f in ["data_v4.npz", "data_v5.npz", "data_v6.npz", "data_v11.npz"]:
        try:
            d = np.load(f)
            s = normalize_states(d["states"])
            a = d["actions"]
            all_states.append(s)
            all_actions.append(a)
            print(f"Loaded {f}: {len(s)} samples")
        except:
            pass
    return np.concatenate(all_states), np.concatenate(all_actions)

def main():
    states, actions = load_data()
    
    print(f"Total sequence frames: {len(states)}")

    # 2. Sequence dataset
    dataset = SequenceDataset(states, actions, seq_len=16)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = NavGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    epochs = 50

    print(f"\nTraining GRU for {epochs} epochs...")
    for ep in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred, _ = model(x)
                val_loss += criterion(pred, y).item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "nav_gru.pt")
            
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"Epoch {ep:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val:.4f}")

    print("\nSaved nav_gru.pt")

if __name__ == "__main__":
    main()
