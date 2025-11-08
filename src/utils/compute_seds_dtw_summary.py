# src/utils/compute_seds_dtw_summary.py
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import numpy as np

# ---------- Optional fast DTW backend ----------
_DTWFN = None
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean as _euclid
    def _dtw_fast(X: np.ndarray, Y: np.ndarray) -> float:
        dist, _ = fastdtw(X.tolist(), Y.tolist(), dist=_euclid)
        return float(dist)
    _DTWFN = _dtw_fast
except Exception:
    _DTWFN = None

def _dtw_exact(X: np.ndarray, Y: np.ndarray) -> float:
    N, M = len(X), len(Y)
    C = np.full((N+1, M+1), np.inf, dtype=float)
    C[0, 0] = 0.0
    for i in range(1, N+1):
        xi = X[i-1]
        for j in range(1, M+1):
            yj = Y[j-1]
            cost = float(np.linalg.norm(xi - yj))
            C[i, j] = cost + min(C[i-1, j], C[i, j-1], C[i-1, j-1])
    return float(C[N, M])

def dtw_distance(X: np.ndarray, Y: np.ndarray) -> float:
    if _DTWFN is not None:
        try:
            return _DTWFN(X, Y)
        except Exception:
            pass
    return _dtw_exact(X, Y)

# ---------- IO helpers ----------
def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}

def pick_xy(d: dict) -> np.ndarray:
    for k in ["z", "traj", "xy", "pos", "Z", "ref", "target", "ref_pos"]:
        if k in d:
            arr = np.asarray(d[k])
            if arr.ndim >= 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    raise KeyError("No (N,2)-like trajectory key in: " + ", ".join(d.keys()))

def find_ref_npz(expt_dir: Path) -> Path | None:
    # Prefer SEDS reference, else fall back to avg demo
    patterns = ["*seds_ref*.npz", "*avg_demo*.npz"]
    for pat in patterns:
        hits = list(expt_dir.glob(pat))
        if hits:
            return hits[0]
    # Sometimes refs live in sibling expt folders for the same shape
    for sib in expt_dir.parent.iterdir():
        if sib.is_dir() and sib != expt_dir:
            for pat in patterns:
                hits = list(sib.glob(pat))
                if hits:
                    return hits[0]
    return None

def is_disturbance_dir(p: Path) -> bool:
    return p.is_dir() and not p.name.endswith("_le")  # ignore least-effort runs

def iter_shape_experiment_dirs(root: Path):
    """
    Yield (shape_name, expt_dir) where expt_dir contains subfolders for disturbances.
    Handles both:
      root/<Shape>/experiments_new_dist/<disturbance>/
      root/<Shape>/experiments/<disturbance>/        (fallback)
      or if root itself is an experiments* dir for a single shape.
    """
    # Case A: root/*/experiments_new_dist or experiments
    found = False
    for shape_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for expt_name in ["experiments_new_dist"]: #, "experiments"]:
            expts = shape_dir / expt_name
            if expts.is_dir():
                found = True
                yield (shape_dir.name, expts)
                break
    if found:
        return
    # Case B: root itself is an experiments* dir (assume shape = parent name)
    # if root.name.startswith("experiments"):
    #     yield (root.parent.name, root)

def readable_dist(dirname: str) -> str:
    # Keep folder name as the disturbance label
    return dirname

# ---------- main compute ----------
def compute_seds(expts_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []  # shape, dist, d_seds, d_clf, d_l1, n_seds, n_clf, n_l1

    for shape, expts_dir in iter_shape_experiment_dirs(expts_root):
        for dist_dir in sorted([p for p in expts_dir.iterdir() if is_disturbance_dir(p)]):
            dist = readable_dist(dist_dir.name)

            # Find logs
            seds_hits = sorted(dist_dir.glob("*_SEDS.npz"))
            clf_hits  = sorted(dist_dir.glob("*_SEDS_CLF.npz"))
            l1_hits   = sorted(dist_dir.glob("*_SEDS_CLF_L1.npz"))
            if not (seds_hits and clf_hits and l1_hits):
                print(f"[WARN] Missing logs for {shape}/{dist}; skipping.")
                continue
            seds_npz = seds_hits[-1]; clf_npz = clf_hits[-1]; l1_npz = l1_hits[-1]

            # Reference
            ref_npz = find_ref_npz(dist_dir)
            if ref_npz is None:
                print(f"[WARN] No SEDS reference/avg demo found for {shape}/{dist}; skipping.")
                continue

            try:
                ref_xy  = pick_xy(load_npz(ref_npz))
                seds_xy = pick_xy(load_npz(seds_npz))
                clf_xy  = pick_xy(load_npz(clf_npz))
                l1_xy   = pick_xy(load_npz(l1_npz))
            except Exception as e:
                print(f"[WARN] Load error {shape}/{dist}: {e}")
                continue

            d_seds = dtw_distance(ref_xy, seds_xy)
            d_clf  = dtw_distance(ref_xy, clf_xy)
            d_l1   = dtw_distance(ref_xy, l1_xy)

            denom = d_seds if d_seds > 0 else 1.0
            n_seds = d_seds / denom
            n_clf  = d_clf  / denom
            n_l1   = d_l1   / denom

            rows.append([shape, dist, d_seds, d_clf, d_l1, n_seds, n_clf, n_l1])

    if not rows:
        print("[INFO] No rows computed. Check --expts_root.")
        return

    # Per-shape raw CSV
    raw_csv = out_dir / "raw_per_shape.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape", "disturbance",
            "dtw_seds", "dtw_seds_clf", "dtw_l1_seds",
            "norm_seds", "norm_seds_clf", "norm_l1_seds"
        ])
        w.writerows(rows)
    print(f"[OK] wrote {raw_csv}")

    # Aggregate by disturbance
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r[1]].append(r)

    summary = []  # dist, mean/var for normalized SEDS, SEDS+CLF, L1-SEDS, count
    for dist, grp in sorted(groups.items()):
        arr = np.array([[r[5], r[6], r[7]] for r in grp], dtype=float)
        means = np.mean(arr, axis=0)
        vars_ = np.var(arr, axis=0, ddof=1) if arr.shape[0] >= 2 else np.zeros(3)
        summary.append([dist, means[0], vars_[0], means[1], vars_[1], means[2], vars_[2], len(grp)])

    summ_csv = out_dir / "summary_per_dist.csv"
    with summ_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "disturbance",
            "seds_mean", "seds_var",
            "seds+clf_mean", "seds+clf_var",
            "l1-seds_mean", "l1-seds_var",
            "num_shapes"
        ])
        w.writerows(summary)
    print(f"[OK] wrote {summ_csv}")

    md = out_dir / "summary_per_dist.md"
    with md.open("w") as f:
        f.write("| Disturbance | SEDS μ±σ² | SEDS+CLF μ±σ² | L1-SEDS μ±σ² | #Shapes |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for dist, m_s, v_s, m_c, v_c, m_l, v_l, n in summary:
            f.write(f"| {dist} | {m_s:.3f} ± {v_s:.3f} | {m_c:.3f} ± {v_c:.3f} | {m_l:.3f} ± {v_l:.3f} | {n} |\n")
    print(f"[OK] wrote {md}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--expts_root",
        type=str,
        default="auto_run_seds",
        help="Root that contains per-shape folders with experiments_new_dist/ subfolders."
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="auto_run_seds/dtw_reports",
        help="Output directory for CSV/MD reports."
    )
    args = ap.parse_args()
    compute_seds(Path(args.expts_root), Path(args.out_dir))

if __name__ == "__main__":
    main()
