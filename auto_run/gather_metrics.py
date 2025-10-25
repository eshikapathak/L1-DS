#!/usr/bin/env python3
import json, csv
from pathlib import Path
import numpy as np

ROOT = Path("auto_run/outputs/expts_least_effort")

def dtw_distance(P, Q):
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean as _euclid
        d, _ = fastdtw(P.tolist(), Q.tolist(), dist=_euclid)
        return float(d)
    except Exception:
        n, m = len(P), len(Q)
        D = np.full((n+1, m+1), np.inf); D[0,0]=0.0
        for i in range(1, n+1):
            for j in range(1, m+1):
                c = np.linalg.norm(P[i-1]-Q[j-1])
                D[i,j] = c + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        return float(D[n,m])

rows_global = []
for shape_dir in sorted(ROOT.iterdir()):
    if not shape_dir.is_dir():
        continue
    # find the avg demo file (saved by run_experiments)
    avg_demo_files = list(shape_dir.glob("**/*_avg_demo.npz"))
    if not avg_demo_files:
        print(f"[WARN] No avg demo in {shape_dir}")
        continue
    avg_demo = np.load(avg_demo_files[0])
    demo_avg_pos = avg_demo["pos"]

    per_shape_rows = []
    for npz in sorted(shape_dir.glob("**/*.npz")):
        if npz.name.endswith("_avg_demo.npz") or npz.name.endswith("_node_ref_from_avg_init.npz"):
            continue
        dat = np.load(npz)
        z = dat["z"]
        d = dtw_distance(z, demo_avg_pos)
        # infer condition & controller from path/name
        rel = npz.relative_to(shape_dir).as_posix()
        per_shape_rows.append([shape_dir.name, rel, f"{d:.6f}"])
        rows_global.append([shape_dir.name, rel, f"{d:.6f}"])

    # per-shape CSV
    out_csv = shape_dir / f"{shape_dir.name}_dtw.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "relpath", "dtw_vs_avg_demo"])
        w.writerows(per_shape_rows)
    print(f"[OK] {out_csv}")

# global CSV
out_csv = ROOT / "_all_dtw.csv"
with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["shape", "relpath", "dtw_vs_avg_demo"])
    w.writerows(rows_global)
print(f"[OK] {out_csv}")
