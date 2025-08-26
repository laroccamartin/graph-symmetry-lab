import numpy as np
import torch, jax.numpy as jnp
from src.models.torch_mpn import TorchMPN
from src.models.jax_mpn import init_params, forward
from jax import random

def permute_graph(A, X, P):
    # P is permutation matrix (N,N)
    A2 = P @ A @ P.T
    X2 = P @ X
    return A2, X2

def test_invariance_torch():
    N, D, H, L = 6, 4, 8, 2
    rng = np.random.default_rng(0)
    A = rng.random((N,N)).astype(np.float32); A = (A + A.T)/2; np.fill_diagonal(A, 0.0)
    X = rng.random((N,D)).astype(np.float32)
    mask = np.ones((N,), dtype=np.float32)
    # build permutation matrix
    perm = rng.permutation(N)
    P = np.eye(N, dtype=np.float32)[perm]

    model = TorchMPN(in_dim=D, hidden_dim=H, layers=L)
    with torch.no_grad():
        y1 = model(torch.tensor(A)[None], torch.tensor(X)[None], torch.tensor(mask)[None]).numpy()
        A2, X2 = permute_graph(A, X, P)
        y2 = model(torch.tensor(A2)[None], torch.tensor(X2)[None], torch.tensor(mask)[None]).numpy()
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)

def test_invariance_jax():
    N, D, H, L = 6, 4, 8, 2
    rng = np.random.default_rng(0)
    A = rng.random((N,N)).astype(np.float32); A = (A + A.T)/2; np.fill_diagonal(A, 0.0)
    X = rng.random((N,D)).astype(np.float32)
    mask = np.ones((N,), dtype=np.float32)
    perm = rng.permutation(N)
    P = np.eye(N, dtype=np.float32)[perm]

    key = random.PRNGKey(0)
    params = init_params(key, in_dim=D, hidden_dim=H, layers=L)
    y1 = forward(params, jnp.array(A[None]), jnp.array(X[None]), jnp.array(mask[None]))
    A2, X2 = permute_graph(A, X, P)
    y2 = forward(params, jnp.array(A2[None]), jnp.array(X2[None]), jnp.array(mask[None]))
    np.testing.assert_allclose(np.array(y1), np.array(y2), rtol=1e-5, atol=1e-5)
