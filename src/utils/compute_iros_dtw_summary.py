# src/utils/compute_iros_dtw_summary.py
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

# def pick_xy(d: dict) -> np.ndarray:
#     """Return (N,2) trajectory from common keys."""
#     for k in ["z", "traj", "xy", "pos", "Z", "ref", "target", "ref_pos", "pos_true"]:
#         if k in d:
#             arr = np.asarray(d[k])
#             if arr.ndim >= 2 and arr.shape[1] >= 2:
#                 return arr[:, :2]
#     raise KeyError("No (N,2)-like trajectory key in: " + ", ".join(d.keys()))

# --- replace the two helpers below in compute_iros_dtw_summary.py ---

def pick_xy(d: dict) -> np.ndarray:
    """(Logs) Return (N,2) from common controller files."""
    for k in ["z", "traj", "xy", "pos", "Z", "ref", "target", "ref_pos", "pos_true"]:
        if k in d:
            arr = np.asarray(d[k])
            if arr.ndim >= 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    raise KeyError("No (N,2)-like trajectory key in: " + ", ".join(d.keys()))

def pick_xy_ref(d: dict) -> np.ndarray:
    """
    (IROS reference NPZ) Prefer the rollout curve; otherwise fall back to
    an average over demos if present.
    Expected keys in these files: 'rollout', 'demos', 't'.
    """
    # 1) Direct rollout (usual case)
    if "rollout" in d:
        arr = np.asarray(d["rollout"])
        if arr.ndim >= 2 and arr.shape[-1] >= 2:
            return arr[:, :2]

    # 2) Average demo (shape ~ (#demos, T, 2))
    if "demos" in d:
        demos = np.asarray(d["demos"])
        if demos.ndim == 3 and demos.shape[-1] >= 2:
            return np.mean(demos, axis=0)[:, :2]

    # 3) Fallback to any other standard key names just in case
    for k in ["ref", "target", "ref_pos", "pos_true"]:
        if k in d:
            arr = np.asarray(d[k])
            if arr.ndim >= 2 and arr.shape[-1] >= 2:
                return arr[:, :2]

    raise KeyError("No (N,2)-like trajectory in reference NPZ; keys: " + ", ".join(d.keys()))


def iros_reference_npz(iros_root: Path, shape: str) -> Path | None:
    cand = iros_root / "rollout_plots" / shape / f"{shape}_rollout_vs_training.npz"
    return cand if cand.exists() else None

def is_dist_dir(p: Path) -> bool:
    return p.is_dir() and not p.name.endswith("_le")

# ---------- main compute ----------
def compute_iros(iros_root: Path, out_dir: Path) -> None:
    """
    iros_root points at: auto_run_iros/iros_outputs_auto_run_2
    """
    expts_root = iros_root / "experiments_2"
    if not expts_root.is_dir():
        raise FileNotFoundError(f"Not found: {expts_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []  # shape, dist, d_node, d_clf, d_l1, n_node, n_clf, n_l1

    # shapes are top-level dirs under experiments_2
    for shape_dir in sorted([p for p in expts_root.iterdir() if p.is_dir()]):
        shape = shape_dir.name
        ref_npz = iros_reference_npz(iros_root, shape)
        if ref_npz is None:
            print(f"[WARN] No reference rollout for shape '{shape}'; skipping shape.")
            continue

        try:
            # ref_xy = pick_xy(load_npz(ref_npz))
            ref_xy = pick_xy_ref(load_npz(ref_npz))
        except Exception as e:
            print(f"[WARN] Failed loading reference for {shape}: {e}")
            continue

        # disturbances = subfolders inside shape directory
        for dist_dir in sorted([p for p in shape_dir.iterdir() if is_dist_dir(p)]):
            dist = dist_dir.name

            node_hits = sorted(dist_dir.glob("*_NODE.npz"))
            clf_hits  = sorted(dist_dir.glob("*_NODE_CLF.npz"))
            l1_hits   = sorted(dist_dir.glob("*_NODE_CLF_L1.npz"))
            if not (node_hits and clf_hits and l1_hits):
                print(f"[WARN] Missing logs for {shape}/{dist}; skipping.")
                continue

            node_npz = node_hits[-1]; clf_npz = clf_hits[-1]; l1_npz = l1_hits[-1]

            try:
                node_xy = pick_xy(load_npz(node_npz))
                clf_xy  = pick_xy(load_npz(clf_npz))
                l1_xy   = pick_xy(load_npz(l1_npz))
            except Exception as e:
                print(f"[WARN] Failed loading trajectories for {shape}/{dist}: {e}")
                continue

            d_node = dtw_distance(ref_xy, node_xy)
            d_clf  = dtw_distance(ref_xy, clf_xy)
            d_l1   = dtw_distance(ref_xy, l1_xy)

            denom = d_node if d_node > 0 else 1.0
            n_node = d_node / denom
            n_clf  = d_clf  / denom
            n_l1   = d_l1   / denom

            rows.append([shape, dist, d_node, d_clf, d_l1, n_node, n_clf, n_l1])

    if not rows:
        print("[INFO] No rows computed. Check paths and filenames.")
        return

    # Per-shape raw CSV
    raw_csv = out_dir / "raw_per_shape.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape", "disturbance",
            "dtw_node", "dtw_node_clf", "dtw_l1_node",
            "norm_node", "norm_node_clf", "norm_l1_node"
        ])
        w.writerows(rows)
    print(f"[OK] wrote {raw_csv}")

    # Aggregate by disturbance
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r[1]].append(r)

    summary = []  # dist, means/vars (normalized), count
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
            "node_mean", "node_var",
            "node+clf_mean", "node+clf_var",
            "l1-node_mean", "l1-node_var",
            "num_shapes"
        ])
        w.writerows(summary)
    print(f"[OK] wrote {summ_csv}")

    md = out_dir / "summary_per_dist.md"
    with md.open("w") as f:
        f.write("| Disturbance | NODE μ±σ² | NODE+CLF μ±σ² | L1-NODE μ±σ² | #Shapes |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for dist, m_n, v_n, m_c, v_c, m_l, v_l, n in summary:
            f.write(f"| {dist} | {m_n:.3f} ± {v_n:.3f} | {m_c:.3f} ± {v_c:.3f} | {m_l:.3f} ± {v_l:.3f} | {n} |\n")
    print(f"[OK] wrote {md}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--iros_root",
        type=str,
        default="auto_run_iros/iros_outputs_auto_run_2",
        help="Root containing experiments_2/ and rollout_plots/."
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="auto_run_iros/iros_outputs_auto_run_2/dtw_reports",
        help="Where to write CSV/MD reports."
    )
    args = ap.parse_args()
    compute_iros(Path(args.iros_root), Path(args.out_dir))

if __name__ == "__main__":
    main()

# python src/utils/compute_iros_dtw_summary.py \
#   --iros_root auto_run_iros/iros_outputs_auto_run_2 \
#   --out_dir   auto_run_iros/iros_outputs_auto_run_2/dtw_reports

# python src/utils/compute_iros_dtw_summary.py \
#   --iros_root auto_run_iros/iros_outputs_auto_run_2 \
#   --out_dir   auto_run_iros/iros_outputs_auto_run_2/dtw_reports
