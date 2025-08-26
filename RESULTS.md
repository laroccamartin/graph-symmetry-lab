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

## Generalization ablations

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
| gen_jax_5to7to8_pool_sum | 0.129626 | 21.291138 |
| gen_jax_5to7to8_scalar | 1.117377 | 5.598892 |
| gen_jax_5to7_train8 | 0.144626 |  |
| gen_torch_5to7to8 | 0.149657 | 14.210784 |
| gen_torch_5to7to8_pool_sum | 0.118059 | 28.667725 |
| gen_torch_5to7to8_scalar | 1.228734 | 4.342133 |
| gen_torch_5to7_train8 | 0.145806 |  |
| timing_jax_jit | 0.202951 |  |
| timing_jax_nojit | 0.202951 |  |

## Test-set metrics (R², MAE)

### gen_jax_5to7to8
R2=0.1769
MAE=2.2579
N=200

### gen_jax_5to7to8_pool_sum
R2=-0.3254
MAE=2.5843
N=200

### gen_jax_5to7to8_scalar
R2=0.6515
MAE=1.9431
N=200

### gen_torch_5to7to8
R2=0.1153
MAE=2.3849
N=200

### gen_torch_5to7to8_pool_sum
R2=-0.7846
MAE=3.0199
N=200

### gen_torch_5to7to8_scalar
R2=0.7297
MAE=1.7171
N=200


## Test-set metrics (R², MAE)

### gen_jax_5to7to8
R2=0.1769
MAE=2.2579
N=200

### gen_jax_5to7to8_pool_sum
R2=-0.3254
MAE=2.5843
N=200

### gen_jax_5to7to8_scalar
R2=0.6515
MAE=1.9431
N=200

### gen_torch_5to7to8
R2=0.1153
MAE=2.3849
N=200

### gen_torch_5to7to8_pool_sum
R2=-0.7846
MAE=3.0199
N=200

### gen_torch_5to7to8_scalar
R2=0.7297
MAE=1.7171
N=200


## LapPE (k=4) generalization

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
| gen_jax_5to7to8_lappe4_mean | 0.216037 | 6.162695 |
| gen_jax_5to7to8_lappe4_sum | 0.078688 | 1.495054 |
| gen_jax_5to7to8_pool_sum | 0.129626 | 21.291138 |
| gen_jax_5to7to8_scalar | 1.117377 | 5.598892 |
| gen_jax_5to7_train8 | 0.144626 |  |
| gen_torch_5to7to8 | 0.149657 | 14.210784 |
| gen_torch_5to7to8_lappe4_mean | 0.258424 | 6.702781 |
| gen_torch_5to7to8_lappe4_sum | 0.065613 | 1.285148 |
| gen_torch_5to7to8_pool_sum | 0.118059 | 28.667725 |
| gen_torch_5to7to8_scalar | 1.228734 | 4.342133 |
| gen_torch_5to7_train8 | 0.145806 |  |
| timing_jax_jit | 0.202951 |  |
| timing_jax_nojit | 0.202951 |  |

## LapPE test-set metrics (R², MAE)

### gen_jax_5to7to8_lappe4_mean
R2=0.6164
MAE=1.9434
N=200

### gen_jax_5to7to8_lappe4_sum
R2=0.9069
MAE=1.0903
N=200

### gen_torch_5to7to8_lappe4_mean
R2=0.5827
MAE=1.9575
N=200

### gen_torch_5to7to8_lappe4_sum
R2=0.9200
MAE=1.0085
N=200


## Deg + LapPE (k=4) generalization

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
| gen_jax_5to7to8_deglap4_mean | 0.303067 | 1.711538 |
| gen_jax_5to7to8_deglap4_sum | 0.121660 | 1.955511 |
| gen_jax_5to7to8_lappe4_mean | 0.216037 | 6.162695 |
| gen_jax_5to7to8_lappe4_sum | 0.078688 | 1.495054 |
| gen_jax_5to7to8_pool_sum | 0.129626 | 21.291138 |
| gen_jax_5to7to8_scalar | 1.117377 | 5.598892 |
| gen_jax_5to7_train8 | 0.144626 |  |
| gen_torch_5to7to8 | 0.149657 | 14.210784 |
| gen_torch_5to7to8_deglap4_mean | 0.233839 | 1.625502 |
| gen_torch_5to7to8_deglap4_sum | 0.113168 | 3.095674 |
| gen_torch_5to7to8_lappe4_mean | 0.258424 | 6.702781 |
| gen_torch_5to7to8_lappe4_sum | 0.065613 | 1.285148 |
| gen_torch_5to7to8_pool_sum | 0.118059 | 28.667725 |
| gen_torch_5to7to8_scalar | 1.228734 | 4.342133 |
| gen_torch_5to7_train8 | 0.145806 |  |
| timing_jax_jit | 0.202951 |  |
| timing_jax_nojit | 0.202951 |  |

## Deg + LapPE test-set metrics (R², MAE)

### gen_jax_5to7to8_deglap4_mean
R2=0.8935
MAE=0.9897
N=200

### gen_jax_5to7to8_deglap4_sum
R2=0.8783
MAE=1.0120
N=200

### gen_torch_5to7to8_deglap4_mean
R2=0.8988
MAE=1.0794
N=200

### gen_torch_5to7to8_deglap4_sum
R2=0.8073
MAE=1.1421
N=200


## Sweep (small): pool × feature × framework (L=2,H=64)


| framework | pool | feature | layers | hidden | seed | train_mse | test_mse |
|---|---|---|---:|---:|---:|---:|---:|
| torch | mean | onehot | 2 | 64 | 0 | 0.149657 | 14.210784 |
| torch | mean | deg_lap_pe | 2 | 64 | 0 | 0.233839 | 1.625502 |
| torch | sum | onehot | 2 | 64 | 0 | 0.118059 | 28.667725 |
| torch | sum | deg_lap_pe | 2 | 64 | 0 | 0.113168 | 3.095674 |
| jax | mean | onehot | 2 | 64 | 0 | 0.150344 | 13.221221 |
| jax | mean | deg_lap_pe | 2 | 64 | 0 | 0.303067 | 1.711538 |
| jax | sum | onehot | 2 | 64 | 0 | 0.129626 | 21.291138 |
| jax | sum | deg_lap_pe | 2 | 64 | 0 | 0.121660 | 1.955511 |

## Auto Report — 2025-08-26 00:40

### Generalization table

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
| gen_jax_5to7to8_deglap4_mean | 0.303067 | 1.711538 |
| gen_jax_5to7to8_deglap4_sum | 0.121660 | 1.955511 |
| gen_jax_5to7to8_lappe4_mean | 0.216037 | 6.162695 |
| gen_jax_5to7to8_lappe4_sum | 0.078688 | 1.495054 |
| gen_jax_5to7to8_pool_sum | 0.129626 | 21.291138 |
| gen_jax_5to7to8_scalar | 1.117377 | 5.598892 |
| gen_jax_5to7_train8 | 0.144626 |  |
| gen_torch_5to7to8 | 0.149657 | 14.210784 |
| gen_torch_5to7to8_deglap4_mean | 0.233839 | 1.625502 |
| gen_torch_5to7to8_deglap4_sum | 0.113168 | 3.095674 |
| gen_torch_5to7to8_lappe4_mean | 0.258424 | 6.702781 |
| gen_torch_5to7to8_lappe4_sum | 0.065613 | 1.285148 |
| gen_torch_5to7to8_pool_sum | 0.118059 | 28.667725 |
| gen_torch_5to7to8_scalar | 1.228734 | 4.342133 |
| gen_torch_5to7_train8 | 0.145806 |  |
| timing_jax_jit | 0.202951 |  |
| timing_jax_nojit | 0.202951 |  |

### Test-set metrics (R², MAE)

| run | R^2 | MAE | N |
|---|---:|---:|---:|
| gen_jax_5to7to8 | 0.1769 | 2.2579 | 200 |
| gen_jax_5to7to8_deglap4_mean | 0.8935 | 0.9897 | 200 |
| gen_jax_5to7to8_deglap4_sum | 0.8783 | 1.0120 | 200 |
| gen_jax_5to7to8_lappe4_mean | 0.6164 | 1.9434 | 200 |
| gen_jax_5to7to8_lappe4_sum | 0.9069 | 1.0903 | 200 |
| gen_jax_5to7to8_pool_sum | -0.3254 | 2.5843 | 200 |
| gen_jax_5to7to8_scalar | 0.6515 | 1.9431 | 200 |
| gen_torch_5to7to8 | 0.1153 | 2.3849 | 200 |
| gen_torch_5to7to8_deglap4_mean | 0.8988 | 1.0794 | 200 |
| gen_torch_5to7to8_deglap4_sum | 0.8073 | 1.1421 | 200 |
| gen_torch_5to7to8_lappe4_mean | 0.5827 | 1.9575 | 200 |
| gen_torch_5to7to8_lappe4_sum | 0.9200 | 1.0085 | 200 |
| gen_torch_5to7to8_pool_sum | -0.7846 | 3.0199 | 200 |
| gen_torch_5to7to8_scalar | 0.7297 | 1.7171 | 200 |

### JAX timing (steady-state)

```
steady_state_epoch_s_jit=0.087
steady_state_epoch_s_nojit=0.347
```

### Sweep winners

**Overall best (lowest test MSE):**

| framework | pool | feature | layers | hidden | seed | train_mse | test_mse | out_dir |
|---|---|---|---:|---:|---:|---:|---:|---|
| torch | mean | deg_lap_pe | 2 | 64 | 0 | 0.233839 | 1.625502 | results/sweep_small/torch_deg_lap_pe_L2_H64_mean_s0 |

**Per-framework best:**

| framework | pool | feature | layers | hidden | seed | train_mse | test_mse | out_dir |
|---|---|---|---:|---:|---:|---:|---:|---|
| jax | mean | deg_lap_pe | 2 | 64 | 0 | 0.303067 | 1.711538 | results/sweep_small/jax_deg_lap_pe_L2_H64_mean_s0 |
| torch | mean | deg_lap_pe | 2 | 64 | 0 | 0.233839 | 1.625502 | results/sweep_small/torch_deg_lap_pe_L2_H64_mean_s0 |
