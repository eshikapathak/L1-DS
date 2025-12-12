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
import matplotlib

# ===== Setup Matplotlib for LaTeX and Professional Look =====
plt.style.use('default')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    # Font sizes
    "font.size": 24,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ===== Style =====
LINE_STYLES = {
    "L1-NODE": "-",     # Solid line (Best performance)
    "NODE+CLF": "-.",   # Dash-dot line (Medium performance)
    "NODE": ":",        # Dotted line (Worst performance)
    "TARGET": "--",     # Dashed line (Target)
}
COLORS = {
    "TARGET": "#000000",    # Black/Darkest
    "L1-NODE": "#0072B2",   # Blue
    "NODE+CLF": "#D55E00",  # Orange
    "NODE": "#666666",      # Darker gray
}
TARGET_LS = LINE_STYLES["TARGET"]

# Line widths
LW_TARGET = 3.0
LW_L1_NODE = 3.5  # Thickest for best performance
LW_NODE_CLF = 2.5
LW_NODE = 2.2

# Goal marker
GOAL_MARKER = "*"
GOAL_MS = 14
GOAL_COLOR = COLORS["TARGET"]

# ===== Roots (edit if needed) =====
LASA_MODELS_ROOT = Path("auto_run/models")
LASA_EXPTS_ROOT  = Path("auto_run/outputs_newdist/expts")

IROS_ROOT        = Path("auto_run_iros/iros_outputs_auto_run_2")
IROS_MODELS_ROOT = IROS_ROOT / "models_2"
IROS_EXPTS_ROOT  = IROS_ROOT / "experiments_2"
IROS_ROLLOUTS    = IROS_ROOT / "rollout_plots"

# ===== Minimal NODE (must match training arch) and Helpers =====
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

def _parse_wd_from_text(text: str) -> Tuple[int, int]:
    mw = re.search(r"_w(\d+)", text)
    md = re.search(r"_d(\d+)", text)
    if not (mw and md):
        raise ValueError(f"Could not parse width/depth from: {text}")
    return int(mw.group(1)), int(md.group(1))

def _newest_eqx_under(root: Path, shape: str) -> Path:
    globs = list((root / shape).rglob("*.eqx"))
    if not globs:
        raise FileNotFoundError(f"No .eqx files under {root/shape}")
    globs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return globs[0]

def load_node_model(models_root: Path, shape: str) -> Tuple[NeuralODE, Path]:
    eqx_path = _newest_eqx_under(models_root, shape)
    try:
        w, d = _parse_wd_from_text(eqx_path.name)
    except Exception:
        w, d = _parse_wd_from_text(eqx_path.parent.name)
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
    cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
    half = 0.5*max(xmax-xmin, ymax-ymin, 1e-6)
    half *= (1.0 + pad_ratio)
    return (cx-half, cx+half), (cy-half, cy+half)

def plot_vector_field(ax, model: NeuralODE, bounds):
    (xmin,xmax),(ymin,ymax) = bounds
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 18), np.linspace(ymin, ymax, 18))
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = jnp.asarray([X[i,j], Y[i,j]])
            uv = np.array(model.func(0.0, p, None))
            U[i,j], V[i,j] = uv[0], uv[1]
    ax.streamplot(X, Y, U, V, density=1.0, color="#d9d9d9", linewidth=0.7, arrowsize=0.8)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

def rollout_ref_from_start(model: NeuralODE, start_xy: np.ndarray, T: int) -> np.ndarray:
    ts = jnp.linspace(0.0, 1.0, max(T, 2))
    return np.array(model(ts, jnp.asarray(start_xy, dtype=jnp.float32)))

def robust_get_target_from_npz(path: Path) -> Optional[np.ndarray]:
    d = load_rollout_npz(path)
    for k in ["rollout"]:
        if k in d:
            arr = np.asarray(d[k])
            if arr.ndim == 2 and arr.shape[1] >= 2:
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
        for k in ["z", "traj", "xy", "pos", "Z"]:
            if k in d:
                logs[label] = np.asarray(d[k])[:, :2]
                break
        else:
            raise KeyError(f"No trajectory key in {p}")
    return logs

# ===== Single-figure plotter =====
def make_single_figure(title: str,
                       outpath: Path,
                       model: NeuralODE,
                       traj_logs: Dict[str, np.ndarray],
                       target_xy: np.ndarray,
                       show_goal_marker: bool = True):
    ref_xy = target_xy
    bounds = compute_bounds([traj_logs["NODE"], traj_logs["NODE+CLF"], traj_logs["L1-NODE"], ref_xy])

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.0))
    plot_vector_field(ax, model, bounds)

    # Target
    ax.plot(ref_xy[:, 0], ref_xy[:, 1],
            lw=LW_TARGET, ls=TARGET_LS, color=COLORS["TARGET"], label="Target trajectory")
    if show_goal_marker:
        ax.plot(ref_xy[-1, 0], ref_xy[-1, 1],
                marker=GOAL_MARKER, ms=GOAL_MS, color=GOAL_COLOR, zorder=5, clip_on=False, label="Goal")

    # Methods
    ax.plot(traj_logs["NODE"][:, 0], traj_logs["NODE"][:, 1],
            lw=LW_NODE, ls=LINE_STYLES["NODE"], color=COLORS["NODE"], label="NODE")
    ax.plot(traj_logs["NODE+CLF"][:, 0], traj_logs["NODE+CLF"][:, 1],
            lw=LW_NODE_CLF, ls=LINE_STYLES["NODE+CLF"], color=COLORS["NODE+CLF"], label="NODE+CLF")
    ax.plot(traj_logs["L1-NODE"][:, 0], traj_logs["L1-NODE"][:, 1],
            lw=LW_L1_NODE, ls=LINE_STYLES["L1-NODE"], color=COLORS["L1-NODE"], label="L1-NODE")

    ax.set_title(title, fontsize=24)
    ax.legend(loc="best", frameon=True, ncol=1)
    fig.tight_layout(pad=0.2)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] wrote {outpath.resolve()}")

# ===== Main =====
def main():
    # --- (1) LASA: Angle / no_llc_pulses ---
    shape = "Angle"
    expt_dir = LASA_EXPTS_ROOT / shape / "no_llc_pulses"
    model, _ = load_node_model(LASA_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    start_xy = np.asarray(logs["NODE"][0], dtype=float)
    ref_xy = rollout_ref_from_start(model, start_xy, T=logs["NODE"].shape[0])
    make_single_figure(
        title=f"LASA: {shape}",
        outpath=Path("fig_angle_no_llc_pulses.png"),
        model=model,
        traj_logs=logs,
        target_xy=ref_xy,
        show_goal_marker=True,
    )

    # --- (2) LASA: GShape / with_llc_unmatched_multisine ---
    shape = "GShape"
    expt_dir = LASA_EXPTS_ROOT / shape / "with_llc_unmatched_multisine"
    model, _ = load_node_model(LASA_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    ref_xy = rollout_ref_from_start(model, np.asarray(logs["NODE"][0]), T=logs["NODE"].shape[0])
    make_single_figure(
        title=f"LASA: {shape}",
        outpath=Path("fig_gshape_with_llc_unmatched_multisine.png"),
        model=model,
        traj_logs=logs,
        target_xy=ref_xy,
        show_goal_marker=True,
    )

    # --- (3) LASA: DoubleBendedLine / with_llc_matched_multisine ---
    shape = "DoubleBendedLine"
    expt_dir = LASA_EXPTS_ROOT / shape / "with_llc_matched_multisine"
    model, _ = load_node_model(LASA_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    ref_xy = rollout_ref_from_start(model, np.asarray(logs["NODE"][0]), T=logs["NODE"].shape[0])
    make_single_figure(
        title=f"LASA: {shape}",
        outpath=Path("fig_doublebendedline_with_llc_matched_multisine.png"),
        model=model,
        traj_logs=logs,
        target_xy=ref_xy,
        show_goal_marker=True,
    )

    # --- (4) IROS: RShape / with_llc_unmatched_const ---
    shape = "RShape"
    expt_dir = IROS_EXPTS_ROOT / shape / "with_llc_unmatched_const"
    model, _ = load_node_model(IROS_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    tgt_npz = IROS_ROLLOUTS / shape / f"{shape}_rollout_vs_training.npz"
    target_xy = robust_get_target_from_npz(tgt_npz)
    make_single_figure(
        title=f"IROS: {shape}",
        outpath=Path("fig_rshape_with_llc_unmatched_const.png"),
        model=model,
        traj_logs=logs,
        target_xy=target_xy,
        show_goal_marker=False,  # was commented out in your original for IROS panels
    )

    # --- (5) IROS: IShape / with_llc_matched_multisine_unmatched_pulse ---
    shape = "IShape"
    expt_dir = IROS_EXPTS_ROOT / shape / "with_llc_matched_multisine_unmatched_pulse"
    model, _ = load_node_model(IROS_MODELS_ROOT, shape)
    logs = load_three_logs(expt_dir)
    tgt_npz = IROS_ROLLOUTS / shape / f"{shape}_rollout_vs_training.npz"
    target_xy = robust_get_target_from_npz(tgt_npz)
    make_single_figure(
        title=f"IROS: {shape}",
        outpath=Path("fig_ishape_with_llc_matched_multisine_unmatched_pulse.png"),
        model=model,
        traj_logs=logs,
        target_xy=target_xy,
        show_goal_marker=False,  # was commented out in your original for IROS panels
    )

if __name__ == "__main__":
    main()
