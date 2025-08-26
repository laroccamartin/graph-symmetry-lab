import numpy as np
import networkx as nx

def degree_onehot(G: nx.Graph, feat_cap: int) -> np.ndarray:
    n = G.number_of_nodes()
    degs = np.array([d for _, d in G.degree()], dtype=np.int64)
    degs = np.clip(degs, 0, feat_cap - 1)
    X = np.zeros((n, feat_cap), dtype=np.float32)
    X[np.arange(n), degs] = 1.0
    return X

def degree_scalar(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    degs = np.array([d for _, d in G.degree()], dtype=np.float32).reshape(n, 1)
    return degs  # raw degree; simple on purpose

def constant_features(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    return np.ones((n, 1), dtype=np.float32)

def pad_matrix(M: np.ndarray, out_shape: tuple) -> np.ndarray:
    out = np.zeros(out_shape, dtype=M.dtype)
    s0, s1 = M.shape
    out[:s0, :s1] = M
    return out

def pad_features(X: np.ndarray, out_shape: tuple) -> np.ndarray:
    out = np.zeros(out_shape, dtype=X.dtype)
    n, d = X.shape
    out[:n, :d] = X
    return out

def graph_to_arrays(G: nx.Graph, max_n: int, feat_dim: int, feat_mode: str = "onehot"):
    """Convert graph to padded (A, X, mask) given feature mode/dim."""
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, dtype=np.float32)

    if feat_mode == "onehot":
        X = degree_onehot(G, feat_dim).astype(np.float32)
    elif feat_mode == "scalar":
        assert feat_dim == 1, "feat_dim must be 1 for scalar mode"
        X = degree_scalar(G)
    elif feat_mode == "constant":
        assert feat_dim == 1, "feat_dim must be 1 for constant mode"
        X = constant_features(G)
    else:
        raise ValueError(f"Unknown feat_mode: {feat_mode}")

    mask = np.zeros((max_n,), dtype=np.float32)
    mask[:n] = 1.0
    A_pad = pad_matrix(A, (max_n, max_n)).astype(np.float32)
    X_pad = pad_features(X, (max_n, X.shape[1])).astype(np.float32)
    return A_pad, X_pad, mask
