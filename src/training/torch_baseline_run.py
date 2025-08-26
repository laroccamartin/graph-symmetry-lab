import argparse, os, csv, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from ..data.dataset import GraphDataset
from ..models.torch_mpn import TorchMPN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth-graphs.npz")
    ap.add_argument("--out_dir", type=str, default="results/baseline_torch")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean","sum","max"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    ds = GraphDataset(args.data)
    A, X, mask, y = ds.get_arrays()
    A = torch.tensor(A); X = torch.tensor(X); mask = torch.tensor(mask); y = torch.tensor(y).unsqueeze(-1)

    loader = DataLoader(TensorDataset(A, X, mask, y), batch_size=args.batch_size, shuffle=True)

    # Model/opt
    model = TorchMPN(in_dim=ds.feat_dim, hidden_dim=args.hidden_dim, layers=args.layers, pool=args.pool)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Log file
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse"])

    # Train
    model.train()
    for epoch in range(1, args.epochs+1):
        total = 0.0
        n = 0
        for A_b, X_b, mask_b, y_b in loader:
            opt.zero_grad()
            pred = model(A_b, X_b, mask_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            opt.step()
            total += loss.item() * A_b.shape[0]
            n += A_b.shape[0]
        train_mse = total / max(1, n)
        print(f"epoch {epoch:03d} | train_mse={train_mse:.6f}")
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_mse:.6f}"])

    # Save checkpoint
    ckpt_path = os.path.join(args.out_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)

    # Full-dataset predictions & scatter
    model.eval()
    with torch.no_grad():
        y_pred = model(A, X, mask).squeeze(-1).numpy()
        y_true = y.squeeze(-1).numpy()

    # Save predictions
    preds_path = os.path.join(args.out_dir, "preds.csv")
    with open(preds_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["y_true", "y_pred"])
        for t, p in zip(y_true, y_pred):
            w.writerow([f"{float(t):.6f}", f"{float(p):.6f}"])

    # Compute & save final MSE
    final_mse = float(np.mean((y_pred - y_true)**2))
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"Final MSE on training set: {final_mse:.6f}\n")

    # Scatter plot
    plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    m = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(m, m)  # y=x
    plt.xlabel("True log|Aut(G)|")
    plt.ylabel("Predicted log|Aut(G)|")
    plt.title(f"Torch {args.pool} pool (layers={args.layers}, hidden={args.hidden_dim})\nMSE={final_mse:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scatter.png"), dpi=150)
    print(f"[saved] {metrics_path}\n[saved] {preds_path}\n[saved] {ckpt_path}\n[saved] scatter.png\n[saved] summary.txt")

if __name__ == "__main__":
    main()
