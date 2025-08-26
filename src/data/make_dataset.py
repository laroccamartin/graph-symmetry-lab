import argparse, json, math, random
import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from .dataset import save_npz
from ..utils.graph_utils import graph_to_arrays

def count_automorphisms(G: nx.Graph) -> int:
    """Count automorphisms via self-isomorphism enumeration (okay for <=7 nodes)."""
    GM = GraphMatcher(G, G)
    return sum(1 for _ in GM.isomorphisms_iter())

def random_graph(n: int) -> nx.Graph:
    """Sample a small graph from a mix of families."""
    choice = random.choice(["er", "cycle", "path", "star", "complete", "regular"])
    if choice == "er":
        p = random.uniform(0.2, 0.8)
        return nx.gnp_random_graph(n, p, seed=random.randint(0, 1_000_000))
    if choice == "cycle" and n >= 3:
        return nx.cycle_graph(n)
    if choice == "path":
        return nx.path_graph(n)
    if choice == "star" and n >= 3:
        return nx.star_graph(n - 1)  # star_graph(k) has k+1 nodes
    if choice == "complete":
        return nx.complete_graph(n)
    if choice == "regular":
        # try a few times to get a valid (d)-regular graph
        for _ in range(10):
            if n <= 2:
                break
            d = random.randint(1, max(1, min(n - 1, 3)))
            if (n * d) % 2 == 0 and d < n:
                try:
                    return nx.random_regular_graph(d, n, seed=random.randint(0, 1_000_000))
                except nx.NetworkXError:
                    continue
        # fallback
        return nx.gnp_random_graph(n, random.uniform(0.2, 0.8), seed=random.randint(0, 1_000_000))
    # fallback generic
    return nx.gnp_random_graph(n, random.uniform(0.2, 0.8), seed=random.randint(0, 1_000_000))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/synth-graphs.npz")
    ap.add_argument("--n_graphs", type=int, default=300)
    ap.add_argument("--min_n", type=int, default=5)
    ap.add_argument("--max_n", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    max_n = args.max_n
    feat_dim = max_n  # degree one-hot up to max degree

    As, Xs, masks, ys, kinds = [], [], [], [], []

    for _ in range(args.n_graphs):
        n = random.randint(args.min_n, args.max_n)
        G = random_graph(n)
        aut = count_automorphisms(G)
        y = float(np.log(max(1, aut)))  # regression target
        A, X, mask = graph_to_arrays(G, max_n=max_n, feat_dim=feat_dim)
        As.append(A); Xs.append(X); masks.append(mask); ys.append(y)
        kinds.append("unknown" if not hasattr(G, "graph") else G.graph.get("kind", "unknown"))

    A = np.stack(As, axis=0)            # (B, N, N)
    X = np.stack(Xs, axis=0)            # (B, N, D)
    mask = np.stack(masks, axis=0)      # (B, N)
    y = np.array(ys, dtype=np.float32)  # (B,)
    meta = {"feat_dim": int(feat_dim), "max_n": int(max_n)}

    save_npz(args.out, A=A, X=X, mask=mask, y=y, meta=json.dumps(meta))

if __name__ == "__main__":
    main()
