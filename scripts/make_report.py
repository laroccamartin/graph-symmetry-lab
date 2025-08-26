import os, sys, csv, glob, subprocess, datetime as dt, statistics as st, re, pathlib

RESULTS_MD = "RESULTS.md"
RESULTS_ROOT = "results"

def ensure_summary_table():
    """Regenerate results/summary_table.md using summarize_results.sh if present; otherwise build it here."""
    out_md = os.path.join(RESULTS_ROOT, "summary_table.md")
    summ = "scripts/summarize_results.sh"
    if os.path.exists(summ) and os.access(summ, os.X_OK):
        subprocess.run([summ], check=True, stdout=subprocess.PIPE)  # script itself writes the file
        return out_md if os.path.exists(out_md) else None

    # Fallback: build table by scanning summary.txt files
    rows = []
    for d in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*"))):
        s = os.path.join(d, "summary.txt")
        if not os.path.isfile(s): continue
        run = os.path.basename(d)
        train = test = ""
        with open(s) as f:
            for line in f:
                if "Final MSE on training set:" in line: train = line.strip().split()[-1]
                if "Final MSE on test set:"     in line: test  = line.strip().split()[-1]
        rows.append((run, train, test))
    rows.sort()
    with open(out_md, "w") as f:
        f.write("| run | train_mse | test_mse |\n|---|---:|---:|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} |\n")
    return out_md

def load_text(path):
    return open(path).read().strip() if os.path.exists(path) else ""

def parse_timing():
    """Prefer results/timing_summary.txt; else compute from timing_jax_* metrics.csv."""
    ts_path = os.path.join(RESULTS_ROOT, "timing_summary.txt")
    if os.path.exists(ts_path):
        return load_text(ts_path)

    def avg_epoch(path_csv):
        if not os.path.exists(path_csv): return None
        with open(path_csv) as f:
            r = list(csv.DictReader(f))
        times = [float(row["epoch_s"]) for row in r if row.get("epoch") != "1" and row.get("epoch_s")]
        return st.mean(times) if times else None

    jit_csv   = os.path.join(RESULTS_ROOT, "timing_jax_jit",   "metrics.csv")
    nojit_csv = os.path.join(RESULTS_ROOT, "timing_jax_nojit", "metrics.csv")
    jit   = avg_epoch(jit_csv)
    nojit = avg_epoch(nojit_csv)
    lines = []
    if jit is not None:   lines.append(f"steady_state_epoch_s_jit={jit:.3f}")
    if nojit is not None: lines.append(f"steady_state_epoch_s_nojit={nojit:.3f}")
    return "\n".join(lines)

def best_from_sweeps():
    """Scan any results/sweep*/sweep_results.csv; return overall best and per-framework best tables (Markdown)."""
    csv_paths = glob.glob(os.path.join(RESULTS_ROOT, "sweep*", "sweep_results.csv"))
    rows = []
    for p in csv_paths:
        with open(p) as f:
            for i, row in enumerate(csv.DictReader(f)):
                try:
                    te = float(row["test_mse"]) if row["test_mse"] not in ("", "nan", "NaN") else float("inf")
                except Exception:
                    te = float("inf")
                rows.append((row, te))
    if not rows:
        return "", ""

    # overall best
    rows_valid = [r for r in rows if r[1] != float("inf")]
    rows_valid.sort(key=lambda x: x[1])
    def row_md(row):
        return f"| {row['framework']} | {row['pool']} | {row['feature']} | {row['layers']} | {row['hidden']} | {row['seed']} | {row['train_mse']} | {row['test_mse']} | {row['out_dir']} |"
    overall = []
    overall.append("| framework | pool | feature | layers | hidden | seed | train_mse | test_mse | out_dir |")
    overall.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    if rows_valid:
        best = rows_valid[0][0]
        overall.append(row_md(best))

    # per-framework best
    perfw = []
    perfw.append("| framework | pool | feature | layers | hidden | seed | train_mse | test_mse | out_dir |")
    perfw.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    by_fw = {}
    for r, te in rows_valid:
        fw = r["framework"]
        if fw not in by_fw or float(r["test_mse"]) < float(by_fw[fw]["test_mse"]):
            by_fw[fw] = r
    for fw, r in sorted(by_fw.items()):
        perfw.append(row_md(r))
    return "\n".join(overall), "\n".join(perfw)

def test_metrics_table():
    """Aggregate test R2/MAE from any results/*/test_metrics.txt."""
    paths = glob.glob(os.path.join(RESULTS_ROOT, "*", "test_metrics.txt"))
    if not paths: return ""
    rows = []
    for p in sorted(paths):
        run = pathlib.Path(p).parent.name
        kv = {}
        with open(p) as f:
            for line in f:
                if "=" in line:
                    k,v = line.strip().split("=",1); kv[k]=v
        rows.append((run, kv.get("R2",""), kv.get("MAE",""), kv.get("N","")))
    out = []
    out.append("| run | R^2 | MAE | N |")
    out.append("|---|---:|---:|---:|")
    for r in rows:
        out.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    return "\n".join(out)

def main():
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    # 1) Generalization table
    table_path = ensure_summary_table()
    table_md = load_text(table_path) if table_path else ""
    # 2) Timing
    timing_md = parse_timing()
    # 3) Sweep winners
    overall_md, perfw_md = best_from_sweeps()
    # 4) Test metrics (R2/MAE)
    metrics_md = test_metrics_table()

    lines = [f"\n## Auto Report — {stamp}\n"]
    if table_md:
        lines += ["### Generalization table\n", table_md, ""]
    if metrics_md:
        lines += ["### Test-set metrics (R², MAE)\n", metrics_md, ""]
    if timing_md:
        lines += ["### JAX timing (steady-state)\n", "```\n"+timing_md+"\n```", ""]
    if overall_md or perfw_md:
        lines += ["### Sweep winners\n"]
        if overall_md:
            lines += ["**Overall best (lowest test MSE):**\n", overall_md, ""]
        if perfw_md:
            lines += ["**Per-framework best:**\n", perfw_md, ""]
    report = "\n".join(lines)

    with open(RESULTS_MD, "a") as f:
        f.write(report)
    print("[ok] Appended Auto Report to", RESULTS_MD)

if __name__ == "__main__":
    main()
