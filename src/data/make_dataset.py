import argparse, json, random
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
    if choice == "cycle" and n >= 3: return nx.cycle_graph(n)
    if choice == "path":            return nx.path_graph(n)
    if choice == "star" and n >= 3: return nx.star_graph(n - 1)
    if choice == "complete":        return nx.complete_graph(n)
    if choice == "regular":
        for _ in range(10):
            if n <= 2: break
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
    ap.add_argument("--pad_n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat_mode", type=str, default="onehot",
                    choices=["onehot","scalar","constant","lap_pe","deg_lap_pe"])
    ap.add_argument("--pe_dim", type=int, default=4, help="k eigenvectors for lap_pe/deg_lap_pe")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    pad_n = args.pad_n if args.pad_n is not None else args.max_n

    if args.feat_mode == "onehot":
        feat_dim = pad_n
    elif args.feat_mode in ("scalar","constant"):
        feat_dim = 1
    elif args.feat_mode == "lap_pe":
        feat_dim = int(args.pe_dim)
    elif args.feat_mode == "deg_lap_pe":
        feat_dim = int(args.pe_dim)  # (degree is added inside graph_to_arrays)
    else:
        raise ValueError("unknown feat_mode")

    As, Xs, masks, ys = [], [], [], []
    for _ in range(args.n_graphs):
        n = random.randint(args.min_n, args.max_n)
        G = random_graph(n)
        aut = count_automorphisms(G)
        y = float(np.log(max(1, aut)))
        A, X, mask = graph_to_arrays(G, max_n=pad_n, feat_dim=feat_dim, feat_mode=args.feat_mode)
        As.append(A); Xs.append(X); masks.append(mask); ys.append(y)

    A = np.stack(As, axis=0); X = np.stack(Xs, axis=0)
    mask = np.stack(masks, axis=0); y = np.array(ys, dtype=np.float32)
    meta = {"feat_dim": int(X.shape[-1]), "max_n": int(pad_n), "feat_mode": args.feat_mode, "pe_dim": int(args.pe_dim)}
    save_npz(args.out, A=A, X=X, mask=mask, y=y, meta=json.dumps(meta))

if __name__ == "__main__":
    main()
