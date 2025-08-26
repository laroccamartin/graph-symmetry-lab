# Results

## Torch baseline (2 layers, 64 hidden, 20 epochs)
- Final MSE (train): <paste from results/baseline_torch/summary.txt>
- Artifacts: `results/baseline_torch/` (metrics.csv, preds.csv, model.pt, scatter.png)

## JAX baseline (2 layers, 64 hidden, 20 epochs)
- Final MSE (train): <paste from results/baseline_jax/summary.txt>
- Artifacts: `results/baseline_jax/` (metrics.csv, preds.csv, params.pkl, scatter.png)

Notes:
- Degree one-hot features; masked mean pooling.
- Next: ablate pooling and features; then try JAX `jit`/`vmap`.
