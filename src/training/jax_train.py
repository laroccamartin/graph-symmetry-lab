import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, random
import optax

from ..data.dataset import GraphDataset
from ..models.jax_mpn import init_params, forward

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth-graphs.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = GraphDataset(args.data)
    A, X, mask, y = ds.get_arrays()
    A = jnp.array(A); X = jnp.array(X); mask = jnp.array(mask); y = jnp.array(y)[:, None]

    key = random.PRNGKey(args.seed)
    params = init_params(key, in_dim=ds.feat_dim, hidden_dim=args.hidden_dim, layers=args.layers)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    @jit
    def mse_loss(params, A_b, X_b, mask_b, y_b):
        pred = forward(params, A_b, X_b, mask_b)
        return jnp.mean((pred - y_b)**2)

    @jit
    def update(params, opt_state, A_b, X_b, mask_b, y_b):
        loss, grads = value_and_grad(mse_loss)(params, A_b, X_b, mask_b, y_b)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    num_batches = int(np.ceil(A.shape[0] / args.batch_size))
    for epoch in range(1, args.epochs+1):
        # simple epoch loop without shuffling for clarity
        total = 0.0
        for i in range(num_batches):
            sl = slice(i*args.batch_size, (i+1)*args.batch_size)
            A_b, X_b, mask_b, y_b = A[sl], X[sl], mask[sl], y[sl]
            params, opt_state, loss = update(params, opt_state, A_b, X_b, mask_b, y_b)
            total += float(loss) * A_b.shape[0]
        print(f"epoch {epoch:03d} | train_mse={total/A.shape[0]:.4f}")

if __name__ == "__main__":
    main()
