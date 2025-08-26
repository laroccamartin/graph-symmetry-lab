import argparse, os, csv, pickle
import numpy as np
import jax, jax.numpy as jnp
from jax import jit, value_and_grad, random
import optax
import matplotlib.pyplot as plt

from ..data.dataset import GraphDataset
from ..models.jax_mpn import init_params, forward

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth-graphs.npz")
    ap.add_argument("--out_dir", type=str, default="results/baseline_jax")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    ds = GraphDataset(args.data)
    A, X, mask, y = ds.get_arrays()
    A = jnp.array(A); X = jnp.array(X); mask = jnp.array(mask); y = jnp.array(y)[:, None]

    # Model/opt
    key = random.PRNGKey(args.seed)
    params = init_params(key, in_dim=ds.feat_dim, hidden_dim=args.hidden_dim, layers=args.layers)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    @jit
    def mse_loss(p, A_b, X_b, mask_b, y_b):
        pred = forward(p, A_b, X_b, mask_b)
        return jnp.mean((pred - y_b) ** 2)

    @jit
    def update(p, opt_state, A_b, X_b, mask_b, y_b):
        loss, grads = value_and_grad(mse_loss)(p, A_b, X_b, mask_b, y_b)
        updates, opt_state = opt.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    # Logging
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse"])

    # Epoch loop
    num_batches = int(np.ceil(A.shape[0] / args.batch_size))
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for i in range(num_batches):
            sl = slice(i * args.batch_size, (i + 1) * args.batch_size)
            A_b, X_b, mask_b, y_b = A[sl], X[sl], mask[sl], y[sl]
            params, opt_state, loss = update(params, opt_state, A_b, X_b, mask_b, y_b)
            total += float(loss) * A_b.shape[0]
            n += int(A_b.shape[0])
        train_mse = total / max(1, n)
        print(f"epoch {epoch:03d} | train_mse={train_mse:.6f}")
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_mse:.6f}"])

    # Save params (as NumPy arrays inside a pickled pytree)
    params_np = jax.tree_map(lambda x: np.array(x), params)
    with open(os.path.join(args.out_dir, "params.pkl"), "wb") as f:
        pickle.dump(params_np, f)

    # Predictions & scatter
    y_pred = np.array(forward(params, A, X, mask)).squeeze(-1)
    y_true = np.array(y).squeeze(-1)

    preds_path = os.path.join(args.out_dir, "preds.csv")
    with open(preds_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["y_true", "y_pred"])
        for t, p in zip(y_true, y_pred):
            w.writerow([f"{float(t):.6f}", f"{float(p):.6f}"])

    final_mse = float(np.mean((y_pred - y_true) ** 2))
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"Final MSE on training set: {final_mse:.6f}\n")

    plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    lo, hi = float(np.minimum(y_true.min(), y_pred.min())), float(np.maximum(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True log|Aut(G)|"); plt.ylabel("Predicted log|Aut(G)|")
    plt.title(f"JAX baseline (layers={args.layers}, hidden={args.hidden_dim})\nMSE={final_mse:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scatter.png"), dpi=150)
    print(f"[saved] {metrics_path}\n[saved] {preds_path}\n[saved] params.pkl\n[saved] scatter.png\n[saved] summary.txt")

if __name__ == "__main__":
    main()
