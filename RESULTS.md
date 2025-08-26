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

## Ablations summary

| run | mse |
|---|---|
| abl_feat_const_torch | 2.472943 |
| abl_feat_scalar_torch | 1.833503 |
| abl_pool_max_jax | 0.379394 |
| abl_pool_max_torch | 0.323566 |
| abl_pool_sum_jax | 0.143397 |
| abl_pool_sum_torch | 0.131131 |
| baseline_jax | 0.287016 |
| baseline_torch | 0.268633 |

## Generalization (train 5–7 → test 8)

| run | mse |
|---|---|
| abl_feat_const_torch | 2.472943 |
| abl_feat_scalar_torch | 1.833503 |
| abl_pool_max_jax | 0.379394 |
| abl_pool_max_torch | 0.323566 |
| abl_pool_sum_jax | 0.143397 |
| abl_pool_sum_torch | 0.131131 |
| baseline_jax | 0.287016 |
| baseline_torch | 0.268633 |
| gen_jax_5to7to8 | 0.150344
13.221221 |
| gen_jax_5to7_train8 | 0.144626 |
| gen_torch_5to7to8 | 0.149657
14.210784 |
| gen_torch_5to7_train8 | 0.145806 |

## Generalization (train 5–7 → test 8)

| run | train_mse | test_mse |
|---|---:|---:|
| abl_feat_const_torch | 2.472943 |  |
| abl_feat_scalar_torch | 1.833503 |  |
| abl_pool_max_jax | 0.379394 |  |
| abl_pool_max_torch | 0.323566 |  |
| abl_pool_sum_jax | 0.143397 |  |
| abl_pool_sum_torch | 0.131131 |  |
| baseline_jax | 0.287016 |  |
| baseline_torch | 0.268633 |  |
| gen_jax_5to7to8 | 0.150344 | 13.221221 |
| gen_jax_5to7_train8 | 0.144626 |  |
| gen_torch_5to7to8 | 0.149657 | 14.210784 |
| gen_torch_5to7_train8 | 0.145806 |  |

## Generalization (train 5–7 → test 8)

| run | train_mse | test_mse |
|---|---:|---:|
| abl_feat_const_torch | 2.472943 |  |
| abl_feat_scalar_torch | 1.833503 |  |
| abl_pool_max_jax | 0.379394 |  |
| abl_pool_max_torch | 0.323566 |  |
| abl_pool_sum_jax | 0.143397 |  |
| abl_pool_sum_torch | 0.131131 |  |
| baseline_jax | 0.287016 |  |
| baseline_torch | 0.268633 |  |
| gen_jax_5to7to8 | 0.150344 | 13.221221 |
| gen_jax_5to7_train8 | 0.144626 |  |
| gen_torch_5to7to8 | 0.149657 | 14.210784 |
| gen_torch_5to7_train8 | 0.145806 |  |

## JAX timing

steady_state_epoch_s_jit=0.085
steady_state_epoch_s_nojit=0.418

## Generalization (train 5–7 → test 8)

| run | train_mse | test_mse |
|---|---:|---:|
| abl_feat_const_torch | 2.472943 |  |
| abl_feat_scalar_torch | 1.833503 |  |
| abl_pool_max_jax | 0.379394 |  |
| abl_pool_max_torch | 0.323566 |  |
| abl_pool_sum_jax | 0.143397 |  |
| abl_pool_sum_torch | 0.131131 |  |
| baseline_jax | 0.287016 |  |
| baseline_torch | 0.268633 |  |
| gen_jax_5to7to8 | 0.150344 | 13.221221 |
| gen_jax_5to7_train8 | 0.144626 |  |
| gen_torch_5to7to8 | 0.149657 | 14.210784 |
| gen_torch_5to7_train8 | 0.145806 |  |
| timing_jax_jit | 0.202951 |  |
| timing_jax_nojit | 0.202951 |  |

## JAX timing

steady_state_epoch_s_jit=0.087
steady_state_epoch_s_nojit=0.347
