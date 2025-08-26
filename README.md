# graph-symmetry-lab

**A hands-on project to learn PyTorch *and* JAX by teaching a neural network to predict graph symmetry.**  
Task: given a graph, predict `log(|Aut(G)|)` — the natural log of the size of the automorphism group.

Why this is a solid "excuse" to do DL work:
- **Graphs are hot** across ML and combinatorics (symmetry, isomorphism, WL tests).
- **Exact labels are computable** for small graphs (via self-isomorphism enumeration), so you can test ideas fast.
- **Dual implementations** (PyTorch & JAX) let you compare ergonomics, speed, and debugging approaches.

## TL;DR quickstart

```bash
# 0) (Recommended) Create a virtualenv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1) Install dependencies
pip install -U pip
pip install -r requirements.txt

# 2) Make a small synthetic dataset (safe defaults)
python -m src.data.make_dataset --out data/synth-graphs.npz --n_graphs 300 --min_n 5 --max_n 7 --seed 0

# 3) Train the PyTorch model (baseline)
python -m src.training.torch_train --data data/synth-graphs.npz --epochs 10 --hidden_dim 64 --layers 2

# 4) Train the JAX model (baseline)
python -m src.training.jax_train --data data/synth-graphs.npz --epochs 10 --hidden_dim 64 --layers 2

# 5) Run tests (includes cross-framework parity sanity check)
pytest -q
```

The dataset is padded to a fixed `max_n` (default = `--max_n`) and exposes a mask so you can try batched message-passing without external GNN libraries.

## What you'll build

- `src/models/torch_mpn.py`: A tiny message-passing network (MPNN) in **PyTorch**.
- `src/models/jax_mpn.py`: The same network in **JAX+Optax**.
- `src/data/make_dataset.py`: Synthetic graphs + exact automorphism counts (for small graphs).
- `src/training/*_train.py`: Minimal training loops (MSE regression on `log(|Aut(G)|)`).

You get a working baseline with tests and CI; extend it into a research-y repo by adding hypotheses, ablations, and a short write-up.

## Research questions to chew on

1. **How far can simple MPNNs go** in predicting symmetry from small graphs? Any inductive bias hacks help (e.g., degree encodings, Laplacian features, positional encodings)?
2. **Generalization across graph families** (ER vs. cycles vs. regular). Where does it break?
3. **Cross-framework ergonomics + perf**: PyTorch vs. JAX (jit, vmap, pmap) for this workload.
4. **(Stretch)** Connect to your quantum/algorithmic interests: can the model predict invariants related to automorphisms to assist classical/quantum pipelines?

## Repo structure

```
graph-symmetry-lab/
├── README.md
├── LICENSE
├── requirements.txt
├── Makefile
├── .gitignore
├── .github/workflows/ci.yml
├── data/                      # created by you (datasets)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── make_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── torch_mpn.py
│   │   └── jax_mpn.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── torch_train.py
│   │   └── jax_train.py
│   └── utils/
│       ├── __init__.py
│       ├── graph_utils.py
│       └── seed.py
└── tests/
    ├── test_dataset.py
    └── test_parity.py
```

## Experiments to try (issue-ify these)

- **Ablations:** degree one-hot vs. constant features; depth vs. width; mean vs. sum pooling.
- **Targets:** classification (`is_symmetric` boolean) vs. regression (`log(|Aut(G)|)`).
- **Data curriculum:** train on 5–7 node graphs, test on 8–9 (careful: labels get expensive to compute!).
- **JAX speedups:** `jit` and `vmap` the forward; try bigger batches.
- **Robustness:** Add minor edge noise and see how predictions shift.

## GitHub setup (exact steps)

1. Create the repo on GitHub named **`graph-symmetry-lab`** (empty — no README).
2. Locally:
   ```bash
   unzip graph-symmetry-lab.zip && cd graph-symmetry-lab
   git init
   git add .
   git commit -m "feat: initial commit (PyTorch+JAX symmetry lab)"
   git branch -M main
   git remote add origin git@github.com:YOUR_USERNAME/graph-symmetry-lab.git   # or https://github.com/YOUR_USERNAME/graph-symmetry-lab.git
   git push -u origin main
   ```
3. On GitHub, the **Actions** tab should show CI running (`pytest -q`).

> Tip: turn each bullet in *Experiments to try* into a GitHub Issue with acceptance criteria. Use a Milestone for each week (e.g., Week 1: baseline + CI; Week 2: ablations; Week 3: JAX jit/vmap; Week 4: short report).

## License

MIT — do research, have fun, cite the repo if useful.

---

If you get stuck: inspect shapes, print masks, and test single-graph forwards before batching. Keep the graphs tiny while debugging.
