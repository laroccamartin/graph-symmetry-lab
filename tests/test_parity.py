import numpy as np
import torch
import jax.numpy as jnp
from src.models.torch_mpn import TorchMPN
from src.models.jax_mpn import forward as jax_forward

def test_cross_framework_parity():
    B, N, D, H, L = 2, 5, 6, 8, 2
    rng = np.random.default_rng(0)
    A = rng.random((B,N,N)).astype(np.float32)
    A = (A + A.transpose(0,2,1))/2.0
    np.fill_diagonal(A[0], 0.0); np.fill_diagonal(A[1], 0.0)
    X = rng.random((B,N,D)).astype(np.float32)
    mask = np.ones((B,N), dtype=np.float32)

    # Torch model with fixed params
    tm = TorchMPN(in_dim=D, hidden_dim=H, layers=L)
    # overwrite weights to deterministic constants so both frameworks match
    for i in range(L):
        tm.W_msg[i].data[:] = torch.tensor(rng.standard_normal(tm.W_msg[i].shape).astype(np.float32))*0.01
        tm.W_self[i].data[:] = torch.tensor(rng.standard_normal(tm.W_self[i].shape).astype(np.float32))*0.01
        tm.b[i].data[:] = torch.tensor(np.zeros((H,), dtype=np.float32))
    tm.W_out.data[:] = torch.tensor(rng.standard_normal(tm.W_out.shape).astype(np.float32))*0.01
    tm.b_out.data[:] = torch.tensor(np.zeros((1,), dtype=np.float32))

    with torch.no_grad():
        y_t = tm(torch.tensor(A), torch.tensor(X), torch.tensor(mask)).numpy()

    # Build matching JAX params from Torch weights
    params = {
        "layers": [
            {
                "W_msg": jnp.array(tm.W_msg[i].detach().numpy()),
                "W_self": jnp.array(tm.W_self[i].detach().numpy()),
                "b": jnp.array(tm.b[i].detach().numpy()),
            } for i in range(L)
        ],
        "W_out": jnp.array(tm.W_out.detach().numpy()),
        "b_out": jnp.array(tm.b_out.detach().numpy()),
    }
    y_j = jax_forward(params, jnp.array(A), jnp.array(X), jnp.array(mask))
    np.testing.assert_allclose(y_t, np.array(y_j), rtol=1e-5, atol=1e-5)
