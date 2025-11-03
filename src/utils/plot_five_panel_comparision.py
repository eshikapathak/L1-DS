# src/utils/plot_five_panel_comparison.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import diffrax

# ===== Roots (edit if needed) =====
LASA_MODELS_ROOT = Path("auto_run/models")
LASA_EXPTS_ROOT  = Path("auto_run/outputs/expts")

IROS_ROOT        = Path("auto_run_iros/iros_outputs_auto_run_2")  # (as in your tree)
IROS_MODELS_ROOT = IROS_ROOT / "models_2"
IROS_EXPTS_ROOT  = IROS_ROOT / "experiments_2"
IROS_ROLLOUTS    = IROS_ROOT / "rollout_plots"

# ===== Minimal NODE (must match training arch) =====
class Func(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, data_size, width_size, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=data_size, out_size=data_size,
            width_size=width_size, depth=depth,
            activation=jnn.tanh, key=key
        )
    def __call__(self, t, y, args):
        return self.mlp(y)

class NeuralODE(eqx.Module):
    func: Func
    def __init__(self, data_size, width_size, depth, *, key):
        self.func = Func(data_size, width_size, depth, key=key)
    def __call__(self, ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=float(ts[0]),
            t1=float(ts[-1]),
            dt0=float(ts[1] - ts[0]) if len(ts) > 1 else 1e-3,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

# ===== Helpers =====
def _parse_wd_from_text(text: str) -> Tuple[int, int]:
    mw = re.search(r"_w(\d+)", text)
    md = re.search(r"_d(\d+)", text)
    if not (mw and md):
        raise ValueError(f"Could not parse width/depth from: {text}")
    return int(mw.group(1)), int(md.group(1))

def _newest_eqx_under(root: Path, shape: str) -> Path:
    """Find newest *.eqx anywhere under root/shape/**."""
    globs = list((root / shape).rglob("*.eqx"))
    if not globs:
        raise FileNotFoundError(f"No .eqx files under {root/shape}")
    globs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return globs[0]

def load_node_model(models_root: Path, shape: str) -> Tuple[NeuralODE, Path]:
    """Find newest .eqx and parse w,d from filename (fallback: parent dir)."""
    eqx_path = _newest_eqx_under(models_root, shape)
    try:
        w, d = _parse_wd_from_text(eqx_path.name)
    except Exception:
        w, d = _parse_wd_from_text(eqx_path.parent.name)  # fallback
    key = jax.random.PRNGKey(0)
    template = NeuralODE(data_size=2, width_size=w, depth=d, key=key)
    model = eqx.tree_deserialise_leaves(str(eqx_path), template)
    return model, eqx_path

def find_npz(expt_dir: Path, suffix: str) -> Path:
    hits = list(expt_dir.glob(f"*{suffix}"))
    if not hits:
        hits = list(expt_dir.glob(f"*{suffix.lower()}")) + list(expt_dir.glob(f"*{suffix.upper()}"))
    if not hits:
        raise FileNotFoundError(f"No file ending {suffix} in {expt_dir}")
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]

def load_rollout_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}

def compute_bounds(points: List[np.ndarray], pad_ratio: float = 0.12):
    Z = np.concatenate(points, axis=0) if len(points) > 1 else points[0]
    xmin, xmax = float(np.min(Z[:,0])), float(np.max(Z[:,0]))
    ymin, ymax = float(np.min(Z[:,1])), float(np.max(Z[:,1]))
    rng = max(xmax - xmin, ymax - ymin, 1e-6)
    pad = pad_ratio * rng
    return (xmin - pad, xmax + pad), (ymin - pad, ymax + pad)

def plot_vector_field(ax, model: NeuralODE, bounds):
    (xmin,xmax),(ymin,ymax) = bounds
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 28), np.linspace(ymin, ymax, 28))
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = jnp.asarray([X[i,j], Y[i,j]])
            uv = np.array(model.func(0.0, p, None))
            U[i,j], V[i,j] = uv[0], uv[1]
    ax.streamplot(X, Y, U, V, density=1.1, color="#cfcfcf", linewidth=0.7, arrowsize=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

def rollout_ref_from_start(model: NeuralODE, start_xy: np.ndarray, T: int) -> np.ndarray:
    ts = jnp.linspace(0.0, 1.0, max(T, 2))
    return np.array(model(ts, jnp.asarray(start_xy, dtype=jnp.float32)))

def robust_get_target_from_npz(path: Path) -> Optional[np.ndarray]:
    d = load_rollout_npz(path)
    for k in ["target","ref","ref_pos","pos_true","target_pos","demo","ref_xy"]:
        if k in d:
            arr = np.asarray(d[k])
            if arr.ndim==2 and arr.shape[1]>=2:
                return arr[:, :2]
    if "pos_true" in d:
        return np.asarray(d["pos_true"])[:, :2]
    return None

def load_three_logs(expt_dir: Path) -> Dict[str, np.ndarray]:
    logs = {}
    node_p = find_npz(expt_dir, "_NODE.npz")
    clf_p  = find_npz(expt_dir, "_NODE_CLF.npz")
    l1_p   = find_npz(expt_dir, "_NODE_CLF_L1.npz")
    for label, p in [("NODE", node_p), ("NODE+CLF", clf_p), ("L1-NODE", l1_p)]:
        d = load_rollout_npz(p)
        for k in ["z","traj","xy","pos","Z"]:
            if k in d:
                logs[label] = np.asarray(d[k])[:, :2]
                break
        else:
            raise KeyError(f"No trajectory key in {p}")
    return logs

def add_panel_letter(ax, letter: str):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=12, fontweight="bold")

def main():
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), constrained_layout=False)
    letters = ["a","b","c","d","e"]

    # === (a) LASA: CShape / no_llc_pulse ===
    shape = "CShape"
    expt_dir = LASA_EXPTS_ROOT / shape / "no_llc_pulse"
    model, _ = load_node_model(LASA_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    start_xy = np.asarray(logs["NODE"][0], dtype=float)
    ref_xy = rollout_ref_from_start(model, start_xy, T=logs["NODE"].shape[0])
    bounds = compute_bounds([logs["NODE"], logs["NODE+CLF"], logs["L1-NODE"], ref_xy])
    ax = axes[0]; plot_vector_field(ax, model, bounds)
    ax.plot(ref_xy[:,0], ref_xy[:,1], lw=2.0, label="Target trajectory")
    ax.plot(logs["NODE"][:,0],     logs["NODE"][:,1],     lw=2.0, label="NODE")
    ax.plot(logs["NODE+CLF"][:,0], logs["NODE+CLF"][:,1], lw=2.0, label="NODE+CLF")
    ax.plot(logs["L1-NODE"][:,0],  logs["L1-NODE"][:,1],  lw=2.0, label="L1-NODE")
    ax.set_xticks([]); ax.set_yticks([]); add_panel_letter(ax, letters[0])

    # === (b) LASA: Snake / with_llc_unmatched_sine ===
    shape = "Snake"
    expt_dir = LASA_EXPTS_ROOT / shape / "with_llc_unmatched_sine"
    model, _ = load_node_model(LASA_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    ref_xy = rollout_ref_from_start(model, np.asarray(logs["NODE"][0]), T=logs["NODE"].shape[0])
    bounds = compute_bounds([logs["NODE"], logs["NODE+CLF"], logs["L1-NODE"], ref_xy])
    ax = axes[1]; plot_vector_field(ax, model, bounds)
    ax.plot(ref_xy[:,0], ref_xy[:,1], lw=2.0)
    ax.plot(logs["NODE"][:,0],     logs["NODE"][:,1],     lw=2.0)
    ax.plot(logs["NODE+CLF"][:,0], logs["NODE+CLF"][:,1], lw=2.0)
    ax.plot(logs["L1-NODE"][:,0],  logs["L1-NODE"][:,1],  lw=2.0)
    ax.set_xticks([]); ax.set_yticks([]); add_panel_letter(ax, letters[1])

    # === (c) LASA: Worm / with_llc_unmatched_const ===
    shape = "Worm"
    expt_dir = LASA_EXPTS_ROOT / shape / "with_llc_unmatched_const"
    model, _ = load_node_model(LASA_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    ref_xy = rollout_ref_from_start(model, np.asarray(logs["NODE"][0]), T=logs["NODE"].shape[0])
    bounds = compute_bounds([logs["NODE"], logs["NODE+CLF"], logs["L1-NODE"], ref_xy])
    ax = axes[2]; plot_vector_field(ax, model, bounds)
    ax.plot(ref_xy[:,0], ref_xy[:,1], lw=2.0)
    ax.plot(logs["NODE"][:,0],     logs["NODE"][:,1],     lw=2.0)
    ax.plot(logs["NODE+CLF"][:,0], logs["NODE+CLF"][:,1], lw=2.0)
    ax.plot(logs["L1-NODE"][:,0],  logs["L1-NODE"][:,1],  lw=2.0)
    ax.set_xticks([]); ax.set_yticks([]); add_panel_letter(ax, letters[2])

    # === (d) IROS: OShape / with_llc_unmatched_pulse ===
    shape = "OShape"
    expt_dir = IROS_EXPTS_ROOT / shape / "with_llc_unmatched_pulse"
    model, _ = load_node_model(IROS_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    tgt_npz = IROS_ROLLOUTS / shape / f"{shape}_rollout_vs_training.npz"
    target_xy = robust_get_target_from_npz(tgt_npz) or logs["NODE"]
    bounds = compute_bounds([logs["NODE"], logs["NODE+CLF"], logs["L1-NODE"], target_xy])
    ax = axes[3]; plot_vector_field(ax, model, bounds)
    ax.plot(target_xy[:,0], target_xy[:,1], lw=2.0)
    ax.plot(logs["NODE"][:,0],     logs["NODE"][:,1],     lw=2.0)
    ax.plot(logs["NODE+CLF"][:,0], logs["NODE+CLF"][:,1], lw=2.0)
    ax.plot(logs["L1-NODE"][:,0],  logs["L1-NODE"][:,1],  lw=2.0)
    ax.set_xticks([]); ax.set_yticks([]); add_panel_letter(ax, letters[3])

    # === (e) IROS: SShape / with_llc_matched_multisine_unmatched_pulse ===
    shape = "SShape"
    expt_dir = IROS_EXPTS_ROOT / shape / "with_llc_matched_multisine_unmatched_pulse"
    model, _ = load_node_model(IROS_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    tgt_npz = IROS_ROLLOUTS / shape / f"{shape}_rollout_vs_training.npz"
    target_xy = robust_get_target_from_npz(tgt_npz) or logs["NODE"]
    bounds = compute_bounds([logs["NODE"], logs["NODE+CLF"], logs["L1-NODE"], target_xy])
    ax = axes[4]; plot_vector_field(ax, model, bounds)
    ax.plot(target_xy[:,0], target_xy[:,1], lw=2.0)
    ax.plot(logs["NODE"][:,0],     logs["NODE"][:,1],     lw=2.0)
    ax.plot(logs["NODE+CLF"][:,0], logs["NODE+CLF"][:,1], lw=2.0)
    ax.plot(logs["L1-NODE"][:,0],  logs["L1-NODE"][:,1],  lw=2.0)
    ax.set_xticks([]); ax.set_yticks([]); add_panel_letter(ax, letters[4])

    # Shared legend (top center)
    labels = ["Target trajectory", "NODE", "NODE+CLF", "L1-NODE"]
    handles = []
    for lbl in labels:
        (h,) = axes[0].plot([], [], lw=2.0, label=lbl)
        handles.append(h)
    fig = axes[0].figure
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.88, wspace=0.08)

    out = Path("five_panel_lasa_iros.png")
    fig.savefig(out, dpi=300)
    print(f"[OK] wrote {out.resolve()}")

if __name__ == "__main__":
    main()
