import argparse, os, csv, pickle
import numpy as np
import jax, jax.numpy as jnp
from jax import jit, value_and_grad, random
import optax, matplotlib.pyplot as plt
from ..data.dataset import GraphDataset
from ..models.jax_mpn import init_params, forward

def eval_on_dataset(params, path, pool):
    ds = GraphDataset(path)
    A, X, mask, y = ds.get_arrays()
    A = jnp.array(A); X = jnp.array(X); mask = jnp.array(mask); y = jnp.array(y)[:, None]
    pred = np.array(forward(params, A, X, mask, pool=pool)).squeeze(-1)
    true = np.array(y).squeeze(-1)
    mse = float(np.mean((pred - true)**2))
    return true, pred, mse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth-graphs.npz")
    ap.add_argument("--test_data", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="results/baseline_jax")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean","sum","max"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = GraphDataset(args.data)
    A, X, mask, y = ds.get_arrays()
    A = jnp.array(A); X = jnp.array(X); mask = jnp.array(mask); y = jnp.array(y)[:, None]

    key = random.PRNGKey(args.seed)
    params = init_params(key, in_dim=ds.feat_dim, hidden_dim=args.hidden_dim, layers=args.layers)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    @jit
    def mse_loss(p, A_b, X_b, mask_b, y_b):
        pred = forward(p, A_b, X_b, mask_b, pool=args.pool)
        return jnp.mean((pred - y_b)**2)

    @jit
    def update(p, opt_state, A_b, X_b, mask_b, y_b):
        loss, grads = value_and_grad(mse_loss)(p, A_b, X_b, mask_b, y_b)
        updates, opt_state = opt.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    # log
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse"])

    # train
    num_batches = int(np.ceil(A.shape[0] / args.batch_size))
    for epoch in range(1, args.epochs+1):
        total, n = 0.0, 0
        for i in range(num_batches):
            sl = slice(i*args.batch_size, (i+1)*args.batch_size)
            A_b, X_b, mask_b, y_b = A[sl], X[sl], mask[sl], y[sl]
            params, opt_state, loss = update(params, opt_state, A_b, X_b, mask_b, y_b)
            total += float(loss) * A_b.shape[0]; n += int(A_b.shape[0])
        train_mse = total / max(1, n)
        print(f"epoch {epoch:03d} | train_mse={train_mse:.6f}")
        with open(os.path.join(args.out_dir, "metrics.csv"), "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_mse:.6f}"])

    # save params
    with open(os.path.join(args.out_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.tree_map(lambda x: np.array(x), params), f)

    # evaluate train
    y_true_tr, y_pred_tr, mse_tr = eval_on_dataset(params, args.data, args.pool)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"Final MSE on training set: {mse_tr:.6f}\n")

    plt.figure(); plt.scatter(y_true_tr, y_pred_tr, s=12)
    lo, hi = float(min(y_true_tr.min(), y_pred_tr.min())), float(max(y_true_tr.max(), y_pred_tr.max()))
    plt.plot([lo, hi],[lo, hi]); plt.xlabel("True"); plt.ylabel("Pred")
    plt.title(f"JAX {args.pool} (train) MSE={mse_tr:.4f}")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "scatter_train.png"), dpi=150)

    # evaluate test
    if args.test_data:
        y_true_te, y_pred_te, mse_te = eval_on_dataset(params, args.test_data, args.pool)
        with open(os.path.join(args.out_dir, "summary.txt"), "a") as f:
            f.write(f"Final MSE on test set: {mse_te:.6f}\n")
        np.savetxt(os.path.join(args.out_dir, "preds_test.csv"),
                   np.stack([y_true_te, y_pred_te], axis=1), delimiter=",", header="y_true,y_pred", comments="")
        plt.figure(); plt.scatter(y_true_te, y_pred_te, s=12)
        lo, hi = float(min(y_true_te.min(), y_pred_te.min())), float(max(y_true_te.max(), y_pred_te.max()))
        plt.plot([lo, hi],[lo, hi]); plt.xlabel("True"); plt.ylabel("Pred")
        plt.title(f"JAX {args.pool} (test) MSE={mse_te:.4f}")
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "scatter_test.png"), dpi=150)

    print("[done] wrote", args.out_dir)

if __name__ == "__main__":
    main()
