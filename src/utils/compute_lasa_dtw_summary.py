# src/utils/compute_lasa_dtw_summary.py
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import numpy as np

# ---------- tiny helpers ----------

def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}

def _pick_traj(d: dict) -> np.ndarray:
    """Return an (N,2) trajectory from various possible keys."""
    for k in ["z", "traj", "xy", "pos", "Z", "ref", "target", "ref_pos"]:
        if k in d:
            arr = np.asarray(d[k])
            if arr.ndim >= 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    raise KeyError("No trajectory-like array found in npz keys: " + ", ".join(d.keys()))

def dtw_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Plain DTW with L2 point cost. Shapes: (N,2) and (M,2)."""
    N, M = len(X), len(Y)
    C = np.full((N+1, M+1), np.inf, dtype=float)
    C[0,0] = 0.0
    for i in range(1, N+1):
        xi = X[i-1]
        for j in range(1, M+1):
            yj = Y[j-1]
            cost = np.linalg.norm(xi - yj)
            C[i,j] = cost + min(C[i-1,j], C[i,j-1], C[i-1,j-1])
    return float(C[N,M])

def find_ref_npz(expt_dir: Path) -> Path | None:
    """
    Try to find a 'node_ref_from_avg_init.npz' for this experiment:
      1) In this expt directory.
      2) In any sibling expt directory for this shape.
    """
    # 1) Here
    here = list(expt_dir.glob("*node_ref_from_avg_init*.npz"))
    if here:
        return here[0]
    # 2) Any sibling under shape
    for sib in expt_dir.parent.iterdir():
        if sib.is_dir():
            cand = list(sib.glob("*node_ref_from_avg_init*.npz"))
            if cand:
                return cand[0]
    return None

def readable_dist_name(dirname: str) -> str:
    # Keep folder name as disturbance “type” for grouping
    # e.g., "with_llc_unmatched_pulse", "with_llc_matched_multisine_unmatched_pulse", "no_llc_pulses"
    return dirname

# ---------- main ----------

def compute(expts_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect rows: one per (shape, disturbance)
    raw_rows = []  # shape, dist, node_raw, clf_raw, l1_raw, node_norm, clf_norm, l1_norm

    # Shapes = top-level dirs
    for shape_dir in sorted([p for p in expts_root.iterdir() if p.is_dir()]):
        shape = shape_dir.name
        # Disturbance subfolders inside shape directory
        for expt_dir in sorted([p for p in shape_dir.iterdir() if p.is_dir()]):
            dist = readable_dist_name(expt_dir.name)

            # Find the three logs
            node_npz = sorted(expt_dir.glob("*_NODE.npz"))
            clf_npz  = sorted(expt_dir.glob("*_NODE_CLF.npz"))
            l1_npz   = sorted(expt_dir.glob("*_NODE_CLF_L1.npz"))

            if not (node_npz and clf_npz and l1_npz):
                # Skip if any is missing (quietly or print a note)
                print(f"[WARN] Missing logs for {shape}/{dist}; skipping.")
                continue

            node_npz = node_npz[-1]
            clf_npz  = clf_npz[-1]
            l1_npz   = l1_npz[-1]

            # Find reference trajectory
            ref_npz = find_ref_npz(expt_dir)
            if ref_npz is None:
                print(f"[WARN] No node_ref_from_avg_init.npz found for {shape}/{dist}; skipping.")
                continue

            try:
                ref_xy  = _pick_traj(load_npz(ref_npz))
                node_xy = _pick_traj(load_npz(node_npz))
                clf_xy  = _pick_traj(load_npz(clf_npz))
                l1_xy   = _pick_traj(load_npz(l1_npz))
            except Exception as e:
                print(f"[WARN] Failed to load trajectories for {shape}/{dist}: {e}")
                continue

            # Raw DTWs vs reference
            d_node = dtw_distance(ref_xy, node_xy)
            d_clf  = dtw_distance(ref_xy, clf_xy)
            d_l1   = dtw_distance(ref_xy, l1_xy)

            # Normalize by NODE DTW (guard div-by-zero)
            denom = d_node if d_node > 0 else 1.0
            n_node = d_node / denom
            n_clf  = d_clf  / denom
            n_l1   = d_l1   / denom

            raw_rows.append([shape, dist, d_node, d_clf, d_l1, n_node, n_clf, n_l1])

    if not raw_rows:
        print("[INFO] No rows computed. Check paths and file names.")
        return

    # Write raw per-shape csv
    raw_csv = out_dir / "raw_per_shape.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "disturbance",
                    "dtw_node", "dtw_node_clf", "dtw_l1_node",
                    "norm_node", "norm_node_clf", "norm_l1_node"])
        w.writerows(raw_rows)
    print(f"[OK] wrote {raw_csv}")

    # Aggregate (mean & variance across shapes) by disturbance
    # Group rows by 'disturbance'
    from collections import defaultdict
    groups = defaultdict(list)
    for r in raw_rows:
        groups[r[1]].append(r)

    summary_rows = []  # dist, mean/var for node, clf, l1 (normalized)
    for dist, rows in sorted(groups.items()):
        arr = np.array([[r[5], r[6], r[7]] for r in rows], dtype=float)  # norm columns
        # means & (sample) variances
        means = np.mean(arr, axis=0)
        # Use sample variance (ddof=1) if at least two samples, else 0
        vars_ = np.var(arr, axis=0, ddof=1) if arr.shape[0] >= 2 else np.zeros(3, dtype=float)
        summary_rows.append([
            dist,
            means[0], vars_[0],
            means[1], vars_[1],
            means[2], vars_[2],
            len(rows)
        ])

    # Write summary csv
    summ_csv = out_dir / "summary_per_dist.csv"
    with summ_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "disturbance",
            "node_mean", "node_var",
            "node+clf_mean", "node+clf_var",
            "l1-node_mean", "l1-node_var",
            "num_shapes"
        ])
        w.writerows(summary_rows)
    print(f"[OK] wrote {summ_csv}")

    # Also write a simple Markdown table (nice for pasting)
    md = out_dir / "summary_per_dist.md"
    with md.open("w") as f:
        f.write("| Disturbance | NODE μ±σ² | NODE+CLF μ±σ² | L1-NODE μ±σ² | #Shapes |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for dist, m_node, v_node, m_clf, v_clf, m_l1, v_l1, n in summary_rows:
            f.write(f"| {dist} | {m_node:.3f} ± {v_node:.3f} | {m_clf:.3f} ± {v_clf:.3f} | {m_l1:.3f} ± {v_l1:.3f} | {n} |\n")
    print(f"[OK] wrote {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expts_root", type=str, default="auto_run/outputs_newdist/expts",
                    help="Root folder containing per-shape experiment folders.")
    ap.add_argument("--out_dir", type=str, default="auto_run/outputs_newdist/dtw_reports",
                    help="Where to write CSV/MD reports.")
    args = ap.parse_args()

    compute(Path(args.expts_root), Path(args.out_dir))


if __name__ == "__main__":
    main()

# python src/utils/compute_lasa_dtw_summary.py \
#   --expts_root auto_run/outputs_newdist/expts \
#   --out_dir auto_run/outputs_newdist/dtw_reports
