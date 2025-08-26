import argparse, os, csv
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from ..data.dataset import GraphDataset
from ..models.torch_mpn import TorchMPN

def eval_on_dataset(model, path):
    ds = GraphDataset(path)
    A, X, mask, y = ds.get_arrays()
    A = torch.tensor(A); X = torch.tensor(X); mask = torch.tensor(mask); y = torch.tensor(y).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        pred = model(A, X, mask).squeeze(-1).cpu().numpy()
        true = y.squeeze(-1).cpu().numpy()
    mse = float(np.mean((pred - true)**2))
    return true, pred, mse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth-graphs.npz")
    ap.add_argument("--test_data", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="results/baseline_torch")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean","sum","max"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    ds = GraphDataset(args.data)
    A, X, mask, y = ds.get_arrays()
    A = torch.tensor(A); X = torch.tensor(X); mask = torch.tensor(mask); y = torch.tensor(y).unsqueeze(-1)

    loader = DataLoader(TensorDataset(A, X, mask, y), batch_size=args.batch_size, shuffle=True)
    model = TorchMPN(in_dim=ds.feat_dim, hidden_dim=args.hidden_dim, layers=args.layers, pool=args.pool)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # log
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse"])

    # train
    model.train()
    for epoch in range(1, args.epochs+1):
        total, n = 0.0, 0
        for A_b, X_b, mask_b, y_b in loader:
            opt.zero_grad()
            pred = model(A_b, X_b, mask_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            opt.step()
            total += loss.item() * A_b.shape[0]; n += A_b.shape[0]
        train_mse = total / max(1, n)
        print(f"epoch {epoch:03d} | train_mse={train_mse:.6f}")
        with open(os.path.join(args.out_dir, "metrics.csv"), "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_mse:.6f}"])

    # save checkpoint
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

    # evaluate on train
    y_true_tr, y_pred_tr, mse_tr = eval_on_dataset(model, args.data)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"Final MSE on training set: {mse_tr:.6f}\n")

    # scatter train
    plt.figure(); plt.scatter(y_true_tr, y_pred_tr, s=12)
    lo, hi = float(min(y_true_tr.min(), y_pred_tr.min())), float(max(y_true_tr.max(), y_pred_tr.max()))
    plt.plot([lo, hi],[lo, hi]); plt.xlabel("True"); plt.ylabel("Pred")
    plt.title(f"Torch {args.pool} (train) MSE={mse_tr:.4f}")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "scatter_train.png"), dpi=150)

    # evaluate on test if provided
    if args.test_data:
        y_true_te, y_pred_te, mse_te = eval_on_dataset(model, args.test_data)
        with open(os.path.join(args.out_dir, "summary.txt"), "a") as f:
            f.write(f"Final MSE on test set: {mse_te:.6f}\n")
        np.savetxt(os.path.join(args.out_dir, "preds_test.csv"),
                   np.stack([y_true_te, y_pred_te], axis=1), delimiter=",", header="y_true,y_pred", comments="")
        plt.figure(); plt.scatter(y_true_te, y_pred_te, s=12)
        lo, hi = float(min(y_true_te.min(), y_pred_te.min())), float(max(y_true_te.max(), y_pred_te.max()))
        plt.plot([lo, hi],[lo, hi]); plt.xlabel("True"); plt.ylabel("Pred")
        plt.title(f"Torch {args.pool} (test) MSE={mse_te:.4f}")
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "scatter_test.png"), dpi=150)

    print("[done] wrote", args.out_dir)

if __name__ == "__main__":
    main()
