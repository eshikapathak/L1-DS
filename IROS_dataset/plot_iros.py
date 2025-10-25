#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _safe_load(path: Path) -> Any:
    if path.suffix.lower() == ".npz":
        return np.load(path, allow_pickle=True)
    return np.load(path, allow_pickle=True)


def _try_as_dict(obj: Any) -> Optional[Dict[str, Any]]:
    try:
        if isinstance(obj, np.lib.npyio.NpzFile):
            return {k: obj[k] for k in obj.files}
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
            maybe = obj.item()
            if isinstance(maybe, dict):
                return maybe
    except Exception:
        pass
    return None


def _title_from_name(p: Path) -> str:
    return p.stem.replace("_", " ")


def _plot_xy(ax: plt.Axes, xy: np.ndarray, **kw):
    ax.plot(xy[:, 0], xy[:, 1], **kw)


def _plot_traj_stack(ax: plt.Axes, xyz: np.ndarray, mean_label="mean", cloud_label="trajectories"):
    """
    xyz: (K,N,>=2) — plot all K paths (light) + mean (thick).
    Returns the mean path (N,2).
    """
    K, N, D = xyz.shape
    xy = xyz[..., :2]

    # light cloud of individual demos
    for k in range(K):
        _plot_xy(ax, xy[k], lw=1.1, alpha=0.35, color="#1f77b4")

    # mean path
    mean_xy = np.mean(xy, axis=0)
    _plot_xy(ax, mean_xy, lw=2.4, color="#d62728", label=mean_label)

    # one legend entry for the cloud
    ax.plot([], [], lw=1.1, alpha=0.35, color="#1f77b4", label=cloud_label)
    return mean_xy


def _overlay_all(trajs: List[Tuple[str, np.ndarray]], out_path: Path):
    if not trajs:
        return
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for i, (name, t) in enumerate(trajs):
        ax.plot(t[:, 0], t[:, 1], lw=2.0, label=name, color=palette[i % len(palette)])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Overlay of trajectories")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_logs_dict(path: Path, logs: Dict[str, Any], out_dir: Path) -> Optional[np.ndarray]:
    """Plot dict/npz logs (supports 'z', 'demo', 'node_ref', 't_norm', 'd_direct')."""
    z = None
    for k in ["z", "pos", "trajectory"]:
        if k in logs:
            z = np.asarray(logs[k]); break

    demo = None
    for k in ["demo", "demo_avg", "avg_demo", "demo_pos"]:
        if k in logs:
            demo = np.asarray(logs[k]); break

    node_ref = None
    for k in ["node_ref", "node_ref_pos"]:
        if k in logs:
            node_ref = np.asarray(logs[k]); break

    t_norm = logs.get("t_norm", None)
    d_direct = logs.get("d_direct", None)
    has_dist = (
        t_norm is not None and d_direct is not None
        and len(np.asarray(t_norm).reshape(-1)) == len(np.asarray(d_direct))
    )

    # try to infer z if missing
    if z is None:
        for v in logs.values():
            arr = np.asarray(v)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                z = arr; break
            if arr.ndim == 2 and arr.shape[1] == 4:
                z = arr[:, :2]; break

    # figure layout
    if has_dist:
        fig = plt.figure(figsize=(6.6, 8.0))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
    else:
        fig, ax1 = plt.subplots(figsize=(6.6, 6.0))
        ax2 = None

    # top: paths
    if demo is not None and demo.ndim == 2 and demo.shape[1] >= 2:
        ax1.plot(demo[:, 0], demo[:, 1], color="red", lw=1.2, alpha=0.9, label="demo(avg)")

    if node_ref is not None and node_ref.ndim == 2 and node_ref.shape[1] >= 2:
        ax1.plot(node_ref[:, 0], node_ref[:, 1], color="#888", lw=1.2, alpha=0.9, label="NODE ref")

    if z is not None and z.ndim == 2 and z.shape[1] >= 2:
        _plot_xy(ax1, z, lw=2.3, label="trajectory")

    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(_title_from_name(path))
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(loc="best", frameon=True, framealpha=0.9)

    # bottom: disturbance magnitude
    if has_dist:
        t = np.asarray(t_norm).reshape(-1)
        d = np.asarray(d_direct)
        mag = np.linalg.norm(d, axis=1)
        ax2.plot(t, mag, lw=2.0)
        ax2.set_xlim(0.0, 1.0)
        ax2.set_ylabel(r"$\|d(t)\|$")
        ax2.set_xlabel("normalized time")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Applied disturbance magnitude")

    fig.tight_layout()
    out_path = out_dir / f"{path.stem}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return z if (z is not None and z.ndim == 2 and z.shape[1] >= 2) else None


def _plot_file(path: Path, out_dir: Path) -> Optional[np.ndarray]:
    data = _safe_load(path)
    logs = _try_as_dict(data)
    if logs is not None:
        return _plot_logs_dict(path, logs, out_dir)

    arr = np.asarray(data)

    # --- 3D stack: (K, N, >=2) ---
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        K, N, D = arr.shape
        fig, ax = plt.subplots(figsize=(6.6, 6.0))
        mean_xy = _plot_traj_stack(ax, arr, mean_label="mean path", cloud_label=f"{K} trajectories")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(_title_from_name(path))
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{path.stem}.png", dpi=220)
        plt.close(fig)
        return mean_xy

    # --- 2D: (N, >=2) ---
    if arr.ndim == 2 and arr.shape[1] >= 2:
        xy = arr[:, :2]
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        _plot_xy(ax, xy, lw=2.3, label="trajectory")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(_title_from_name(path))
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{path.stem}.png", dpi=220)
        plt.close(fig)
        return xy

    print(f"[WARN] Skipping {path.name}: unsupported array shape {arr.shape}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="iros_dataset",
                    help="Directory containing .npy/.npz files.")
    ap.add_argument("--out_dir", type=str, default="iros_dataset/plots",
                    help="Where to save generated PNGs.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([*in_dir.glob("*.npy"), *in_dir.glob("*.npz")])
    if not files:
        print(f"[INFO] No .npy/.npz files found in {in_dir}")
        return

    overlay_trajs: List[Tuple[str, np.ndarray]] = []

    for p in files:
        try:
            rep = _plot_file(p, out_dir)
            if rep is not None and rep.ndim == 2 and rep.shape[1] == 2:
                # For 3D inputs, rep is the mean path; for 2D inputs, rep is the path itself
                overlay_trajs.append((p.stem, rep))
            print(f"[OK] Plotted {p.name}")
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")

    if len(overlay_trajs) >= 2:
        _overlay_all(overlay_trajs, out_dir / "_overlay_all.png")
        print(f"[OK] Wrote overlay: {out_dir / '_overlay_all.png'}")


if __name__ == "__main__":
    main()
