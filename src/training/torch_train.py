import argparse, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ..data.dataset import GraphDataset
from ..models.torch_mpn import TorchMPN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth-graphs.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    ds = GraphDataset(args.data)
    A, X, mask, y = ds.get_arrays()
    A = torch.tensor(A); X = torch.tensor(X); mask = torch.tensor(mask); y = torch.tensor(y).unsqueeze(-1)
    loader = DataLoader(TensorDataset(A, X, mask, y), batch_size=args.batch_size, shuffle=True)

    model = TorchMPN(in_dim=ds.feat_dim, hidden_dim=args.hidden_dim, layers=args.layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs+1):
        total = 0.0
        for A_b, X_b, mask_b, y_b in loader:
            opt.zero_grad()
            pred = model(A_b, X_b, mask_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            opt.step()
            total += loss.item() * A_b.shape[0]
        print(f"epoch {epoch:03d} | train_mse={total/len(ds):.4f}")

if __name__ == "__main__":
    main()
