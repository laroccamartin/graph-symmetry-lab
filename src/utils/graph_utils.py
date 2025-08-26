import numpy as np
import networkx as nx

def degree_onehot(G: nx.Graph, feat_dim: int) -> np.ndarray:
    """Return (n_nodes, feat_dim) degree one-hot features (clipped to feat_dim-1)."""
    n = G.number_of_nodes()
    degs = np.array([d for _, d in G.degree()], dtype=np.int64)
    degs = np.clip(degs, 0, feat_dim - 1)
    X = np.zeros((n, feat_dim), dtype=np.float32)
    X[np.arange(n), degs] = 1.0
    return X

def pad_matrix(M: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Pad 2D to out_shape with zeros."""
    out = np.zeros(out_shape, dtype=M.dtype)
    s0, s1 = M.shape
    out[:s0, :s1] = M
    return out

def pad_features(X: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Pad 2D features to (max_n, feat_dim)."""
    out = np.zeros(out_shape, dtype=X.dtype)
    n, d = X.shape
    out[:n, :d] = X
    return out

def graph_to_arrays(G: nx.Graph, max_n: int, feat_dim: int):
    """Convert a NetworkX graph to padded adjacency, feature matrix, and mask."""
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, dtype=np.float32)
    X = degree_onehot(G, feat_dim).astype(np.float32)
    mask = np.zeros((max_n,), dtype=np.float32)
    mask[:n] = 1.0
    A_pad = pad_matrix(A, (max_n, max_n)).astype(np.float32)
    X_pad = pad_features(X, (max_n, feat_dim)).astype(np.float32)
    return A_pad, X_pad, mask
