from typing import Dict, List
import jax
import jax.numpy as jnp
from jax import random

def init_params(key, in_dim: int, hidden_dim: int = 64, layers: int = 2) -> Dict:
    params = {"layers": []}
    k1, k2 = random.split(key)
    last_dim = in_dim
    for i in range(layers):
        k1, k2 = random.split(k1)
        W_msg = random.normal(k1, (last_dim, hidden_dim)) * 0.1
        W_self = random.normal(k2, (last_dim, hidden_dim)) * 0.1
        b = jnp.zeros((hidden_dim,))
        params["layers"].append({"W_msg": W_msg, "W_self": W_self, "b": b})
        last_dim = hidden_dim
    k1, k2 = random.split(k1)
    params["W_out"] = random.normal(k1, (hidden_dim, 1)) * 0.1
    params["b_out"] = jnp.zeros((1,))
    return params

def forward(params: Dict, A: jnp.ndarray, X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    A: (B, N, N), X: (B, N, D), mask: (B, N)
    returns y: (B, 1)
    """
    H = X
    for layer in params["layers"]:
        AX = jnp.einsum("bij,bjk->bik", A, H)   # (B, N, D/H)
        H = jnp.einsum("bik,kh->bih", AX, layer["W_msg"]) + \
            jnp.einsum("bik,kh->bih", H, layer["W_self"]) + layer["b"]
        H = jax.nn.relu(H)
        H = H * mask[..., None]
    denom = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    pooled = jnp.sum(H * mask[..., None], axis=1) / denom
    y = jnp.einsum("bi,ih->bh", pooled, params["W_out"]) + params["b_out"]
    return y
