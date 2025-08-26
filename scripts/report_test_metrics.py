import argparse, os, csv
import numpy as np
import matplotlib.pyplot as plt

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    err = y_pred - y_true
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean())**2) if len(y_true)>1 else np.nan
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan
    mae = float(np.mean(np.abs(err)))
    return r2, mae, err

def binned_calibration(y_true, y_pred, bins=10):
    idx = np.argsort(y_true)
    y_true_sorted = y_true[idx]; y_pred_sorted = y_pred[idx]
    bsz = max(1, len(y_true)//bins)
    xs, ys = [], []
    for i in range(0, len(y_true), bsz):
        sl = slice(i, min(i+bsz, len(y_true)))
        xs.append(float(np.mean(y_true_sorted[sl])))
        ys.append(float(np.mean(y_pred_sorted[sl])))
    return np.array(xs), np.array(ys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+", help="results/* dirs with preds_test.csv")
    args = ap.parse_args()

    for d in args.result_dirs:
        p = os.path.join(d, "preds_test.csv")
        if not os.path.exists(p):
            print(f"[skip] {d} (no preds_test.csv)")
            continue
        data = np.genfromtxt(p, delimiter=",", names=True)
        y_true, y_pred = data["y_true"], data["y_pred"]
        r2, mae, err = metrics(y_true, y_pred)
        with open(os.path.join(d, "test_metrics.txt"), "w") as f:
            f.write(f"R2={r2:.4f}\nMAE={mae:.4f}\nN={len(y_true)}\n")
        # plots
        # 1) calibration curve
        xs, ys = binned_calibration(y_true, y_pred, bins=10)
        plt.figure()
        lo, hi = float(min(xs.min(), ys.min())), float(max(xs.max(), ys.max()))
        plt.plot([lo, hi], [lo, hi])
        plt.plot(xs, ys, marker="o")
        plt.xlabel("True (bin mean)"); plt.ylabel("Predicted (bin mean)")
        plt.title("Calibration (test)")
        plt.tight_layout(); plt.savefig(os.path.join(d, "calibration_test.png"), dpi=150)
        # 2) residual histogram
        plt.figure()
        plt.hist(err, bins=20)
        plt.xlabel("Prediction error (y_pred - y_true)"); plt.ylabel("count")
        plt.title("Residuals (test)")
        plt.tight_layout(); plt.savefig(os.path.join(d, "residuals_test.png"), dpi=150)
        print(f"[ok] {d}: R2={r2:.4f}, MAE={mae:.4f}")

if __name__ == "__main__":
    main()
