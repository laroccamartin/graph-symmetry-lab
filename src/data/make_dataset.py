import argparse, json, math, random
import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from .dataset import save_npz
from ..utils.graph_utils import graph_to_arrays

def count_automorphisms(G: nx.Graph) -> int:
    GM = GraphMatcher(G, G)
    return sum(1 for _ in GM.isomorphisms_iter())

def random_graph(n: int) -> nx.Graph:
    choice = random.choice(["er", "cycle", "path", "star", "complete", "regular"])
    if choice == "er":
        p = random.uniform(0.2, 0.8)
        return nx.gnp_random_graph(n, p, seed=random.randint(0, 1_000_000))
    if choice == "cycle" and n >= 3:
        return nx.cycle_graph(n)
    if choice == "path":
        return nx.path_graph(n)
    if choice == "star" and n >= 3:
        return nx.star_graph(n - 1)
    if choice == "complete":
        return nx.complete_graph(n)
    if choice == "regular":
        for _ in range(10):
            if n <= 2:
                break
            d = random.randint(1, max(1, min(n - 1, 3)))
            if (n * d) % 2 == 0 and d < n:
                try:
                    return nx.random_regular_graph(d, n, seed=random.randint(0, 1_000_000))
                except nx.NetworkXError:
                    continue
        return nx.gnp_random_graph(n, random.uniform(0.2, 0.8), seed=random.randint(0, 1_000_000))
    return nx.gnp_random_graph(n, random.uniform(0.2, 0.8), seed=random.randint(0, 1_000_000))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/synth-graphs.npz")
    ap.add_argument("--n_graphs", type=int, default=300)
    ap.add_argument("--min_n", type=int, default=5)
    ap.add_argument("--max_n", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat_mode", type=str, default="onehot", choices=["onehot","scalar","constant"])
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    max_n = args.max_n
    feat_dim = max_n if args.feat_mode == "onehot" else 1

    As, Xs, masks, ys, kinds = [], [], [], [], []

    for _ in range(args.n_graphs):
        n = random.randint(args.min_n, args.max_n)
        G = random_graph(n)
        aut = count_automorphisms(G)
        y = float(np.log(max(1, aut)))
        A, X, mask = graph_to_arrays(G, max_n=max_n, feat_dim=feat_dim, feat_mode=args.feat_mode)
        As.append(A); Xs.append(X); masks.append(mask); ys.append(y)
        kinds.append("unknown")

    A = np.stack(As, axis=0)
    X = np.stack(Xs, axis=0)
    mask = np.stack(masks, axis=0)
    y = np.array(ys, dtype=np.float32)
    meta = {"feat_dim": int(feat_dim), "max_n": int(max_n), "feat_mode": args.feat_mode}

    save_npz(args.out, A=A, X=X, mask=mask, y=y, meta=json.dumps(meta))

if __name__ == "__main__":
    main()
