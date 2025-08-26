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
    return degs

def constant_features(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    return np.ones((n, 1), dtype=np.float32)

def laplacian_positional_encoding(G: nx.Graph, k: int) -> np.ndarray:
    n = G.number_of_nodes()
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=np.float32)
    # Normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
    w, V = np.linalg.eigh(L)  # sorted ascending
    # skip near-zero eigenvalues (components)
    eps = 1e-8
    nz_idx = np.where(w > eps)[0]
    if len(nz_idx) == 0:
        return np.zeros((n, k), dtype=np.float32)
    start = nz_idx[0]
    take = V[:, start:start+k]
    # fix sign ambiguity
    signs = np.sign(take[0, :] + 1e-12); signs[signs == 0] = 1.0
    take = take * signs
    if take.shape[1] < k:
        pad = np.zeros((n, k - take.shape[1]), dtype=take.dtype)
        take = np.concatenate([take, pad], axis=1)
    return take[:, :k].astype(np.float32)

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
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, dtype=np.float32)

    if feat_mode == "onehot":
        X = degree_onehot(G, feat_dim).astype(np.float32)
    elif feat_mode == "scalar":
        assert feat_dim == 1
        X = degree_scalar(G)
    elif feat_mode == "constant":
        assert feat_dim == 1
        X = constant_features(G)
    elif feat_mode == "lap_pe":
        X = laplacian_positional_encoding(G, feat_dim)
    elif feat_mode == "deg_lap_pe":
        # feat_dim = k for the LapPE part; we will concatenate 1D degree
        k = int(feat_dim)
        X = np.concatenate([degree_scalar(G), laplacian_positional_encoding(G, k)], axis=1)
    else:
        raise ValueError(f"Unknown feat_mode: {feat_mode}")

    mask = np.zeros((max_n,), dtype=np.float32); mask[:n] = 1.0
    A_pad = pad_matrix(A, (max_n, max_n)).astype(np.float32)
    X_pad = pad_features(X, (max_n, X.shape[1])).astype(np.float32)
    return A_pad, X_pad, mask
