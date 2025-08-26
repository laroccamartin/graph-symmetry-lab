import argparse, itertools, os, subprocess, csv, json, time, pathlib

def run_cmd(cmd, cwd=None):
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True)

def read_summary(out_dir):
    p = pathlib.Path(out_dir) / "summary.txt"
    train = test = ""
    if p.exists():
        for line in p.read_text().splitlines():
            if "training set:" in line: train = line.strip().split()[-1]
            if "test set:" in line:     test  = line.strip().split()[-1]
    return train, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frameworks", default="torch,jax")
    ap.add_argument("--pools", default="mean,sum")
    ap.add_argument("--features", default="onehot,scalar,lap_pe,deg_lap_pe")
    ap.add_argument("--layers", default="2")
    ap.add_argument("--hiddens", default="64")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--pad_n", type=int, default=8)
    ap.add_argument("--pe_dim", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--train_range", default="5,7")  # "min,max"
    ap.add_argument("--test_n", type=int, default=8)
    ap.add_argument("--out_root", default="results/sweep")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    frs   = args.frameworks.split(",")
    pools = args.pools.split(",")
    feats = args.features.split(",")
    lays  = [int(x) for x in args.layers.split(",")]
    hids  = [int(x) for x in args.hiddens.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    tmin, tmax = [int(x) for x in args.train_range.split(",")]

    # prepare datasets per feature mode
    ds_cache = {}
    for feat in feats:
        train_path = f"data/train_{tmin}to{tmax}_pad{args.pad_n}_{feat}{args.pe_dim if feat!='onehot' and feat!='scalar' and feat!='constant' else ''}.npz"
        test_path  = f"data/test_{args.test_n}_pad{args.pad_n}_{feat}{args.pe_dim if feat!='onehot' and feat!='scalar' and feat!='constant' else ''}.npz"
        if not os.path.exists(train_path):
            run_cmd([
                "python","-m","src.data.make_dataset","--out",train_path,
                "--n_graphs",str(args.n_train),"--min_n",str(tmin),"--max_n",str(tmax),
                "--pad_n",str(args.pad_n),"--seed","0","--feat_mode",feat,"--pe_dim",str(args.pe_dim)
            ])
        if not os.path.exists(test_path):
            run_cmd([
                "python","-m","src.data.make_dataset","--out",test_path,
                "--n_graphs",str(args.n_test),"--min_n",str(args.test_n),"--max_n",str(args.test_n),
                "--pad_n",str(args.pad_n),"--seed","1","--feat_mode",feat,"--pe_dim",str(args.pe_dim)
            ])
        ds_cache[feat] = (train_path, test_path)

    out_csv = os.path.join(args.out_root, "sweep_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["framework","pool","feature","layers","hidden","seed","train_mse","test_mse","out_dir"])

    for fw, pool, feat, L, H, sd in itertools.product(frs, pools, feats, lays, hids, seeds):
        train_path, test_path = ds_cache[feat]
        tag = f"{fw}_{feat}_L{L}_H{H}_{pool}_s{sd}"
        out_dir = os.path.join(args.out_root, tag)
        os.makedirs(out_dir, exist_ok=True)
        if fw == "torch":
            cmd = [
                "python","-m","src.training.torch_baseline_run",
                "--data",train_path,"--test_data",test_path,
                "--out_dir",out_dir,"--epochs","30","--hidden_dim",str(H),
                "--layers",str(L),"--pool",pool,"--seed",str(sd)
            ]
        else:
            cmd = [
                "python","-m","src.training.jax_baseline_run",
                "--data",train_path,"--test_data",test_path,
                "--out_dir",out_dir,"--epochs","30","--hidden_dim",str(H),
                "--layers",str(L),"--pool",pool,"--seed",str(sd)
            ]
        run_cmd(cmd)
        tr, te = read_summary(out_dir)
        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow([fw,pool,feat,L,H,sd,tr,te,out_dir])

    print("[done] wrote", out_csv)

if __name__ == "__main__":
    main()
