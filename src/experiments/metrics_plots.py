from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple

# =======================
#        Metrics
# =======================

def dtw_distance(P, Q):
    """DTW distance between two 2D sequences P, Q."""
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean as _euclid
        d, _ = fastdtw(P.tolist(), Q.tolist(), dist=_euclid)
        return float(d)
    except Exception:
        n, m = len(P), len(Q)
        D = np.full((n+1, m+1), np.inf); D[0, 0] = 0.0
        for i in range(1, n+1):
            for j in range(1, m+1):
                c = np.linalg.norm(P[i-1] - Q[j-1])
                D[i, j] = c + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        return float(D[n, m])

def time_in_tube(P, Q, radius: float):
    """Fraction of time P stays within a radius of Q (pointwise; min length used)."""
    L = min(len(P), len(Q))
    d = np.linalg.norm(P[:L] - Q[:L], axis=1)
    return float(np.mean(d <= radius))

def phase_drift_deg(P, Q):
    """Crude phase drift (deg) between centered P and Q via mean wrapped angle diff."""
    p0 = P - P.mean(0); q0 = Q - Q.mean(0)
    ang_p = np.unwrap(np.arctan2(p0[:, 1], p0[:, 0]))
    ang_q = np.unwrap(np.arctan2(q0[:, 1], q0[:, 0]))
    L = min(len(ang_p), len(ang_q))
    return float(np.degrees(ang_p[:L].mean() - ang_q[:L].mean()))

# =======================
#         Plots
# =======================

def _compute_field_grid(field_fn, bounds, grid=25):
    (xmin, xmax), (ymin, ymax) = bounds
    X, Y = np.meshgrid(np.linspace(xmin, xmax, grid), np.linspace(ymin, ymax, grid))
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u = field_fn(np.array([X[i, j], Y[i, j]]))  # expected shape (2,)
            U[i, j], V[i, j] = u[0], u[1]
    return X, Y, U, V

def plot_rollouts_with_field(rows_labels, cols_cells, titles, outpath: Path,
                             field_fn=None, field_bounds=None, demo=None, subtitle=None):
    """
    rows_labels: list[str] (controllers)
    cols_cells: list[list[(z, label)]] per column (grid of plots)
    titles: list[str] per column
    field_fn: callable(p)->vel for streamplot background (optional)
    field_bounds: ((xmin,xmax),(ymin,ymax)) for streamplot window
    demo: optional (N,2) curve to overlay
    """
    R = len(rows_labels); C = len(cols_cells)
    fig, axs = plt.subplots(R, C, figsize=(4.6*C, 4.6*R), squeeze=False)

    if subtitle:
        fig.suptitle(subtitle, fontsize=11, y=0.995)

    # precompute field once
    field = None
    if field_fn is not None and field_bounds is not None:
        field = _compute_field_grid(field_fn, field_bounds, grid=28)

    for i in range(R):
        for j in range(C):
            ax = axs[i, j]
            if field is not None:
                X, Y, U, V = field
                ax.streamplot(X, Y, U, V, density=1.0, color="#cccccc",
                              linewidth=0.7, arrowsize=0.7)
            if demo is not None:
                ax.plot(demo[:, 0], demo[:, 1], c="0.6", lw=2, label="demo")
            z, lbl = cols_cells[j][i]
            ax.plot(z[:, 0], z[:, 1], lw=2.0, label=lbl)
            if i == 0:
                ax.set_title(titles[j])
            ax.axis("equal"); ax.grid(True, alpha=0.3)
            if j == 0: ax.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outpath, dpi=240)
    plt.close()

def bar_with_ci(names, vals, ci, ylabel, title, outpath):
    """Bar chart with CI and numeric labels on top of bars."""
    x = np.arange(len(names))
    w = 0.65
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(x, vals, yerr=ci, width=w, capsize=5, alpha=0.9)

    # annotate values on bars
    for rect, v in zip(bars, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(),
            f"{float(v):.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x, names, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_all_together(rollouts, demo, field_fn, field_bounds, subtitle, outpath):
    """
    Single panel with vector field, thin red demo, and multiple controller rollouts.
    rollouts: [(z_array (N,2), label), ...]
    """
    (xmin, xmax), (ymin, ymax) = field_bounds
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # --- light vector field background
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 22),
                         np.linspace(ymin, ymax, 22))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    vv = np.stack([field_fn(p) for p in pts], axis=0)  # (M,2)
    U = vv[:, 0].reshape(gx.shape)
    V = vv[:, 1].reshape(gy.shape)
    ax.streamplot(gx, gy, U, V, density=1.2, linewidth=0.6, arrowsize=0.7,
                  color="#cccccc", zorder=0)

    # --- demo (thin red)
    ax.plot(demo[:, 0], demo[:, 1], color="red", lw=1.1, alpha=0.9,
            label="demo", zorder=2)

    # --- rollouts
    palette = ["#1f77b4", "#2ca02c", "#9467bd"]  # blue, green, purple
    for k, (z, lbl) in enumerate(rollouts):
        color = palette[k % len(palette)]
        ax.plot(z[:, 0], z[:, 1], lw=2.3, label=lbl, color=color, zorder=3)

    # cosmetics
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25)
    ax.set_title(subtitle, fontsize=12)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_all_together_with_dist(
    rollouts: List[Tuple[np.ndarray, str]],
    demo: Optional[np.ndarray],
    field_fn,
    field_bounds,
    subtitle: str,
    outpath: Path,
    t_norm: Optional[np.ndarray] = None,
    d_direct: Optional[np.ndarray] = None,
    ref_curves: Optional[List[Tuple[str, np.ndarray]]] = None,
    d_matched: Optional[np.ndarray] = None,     # <-- NEW (LLC: acceleration-channel disturbance)
    d_unmatched: Optional[np.ndarray] = None,   # <-- NEW (LLC: position-rate disturbance)
):
    """
    Top: vector field + reference curves + controller rollouts.
    Bottom: disturbance plot.
      - no_llc: use d_direct (vector) -> plot ||d_direct||
      - with_llc: use d_matched and/or d_unmatched (vectors) -> plot their norms
    """
    (xmin, xmax), (ymin, ymax) = field_bounds
    fig = plt.figure(figsize=(10, 8.2))

    # --- top panel
    ax1 = fig.add_subplot(2, 1, 1)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 22),
                         np.linspace(ymin, ymax, 22))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    vv = np.stack([field_fn(p) for p in pts], axis=0)
    U = vv[:, 0].reshape(gx.shape); V = vv[:, 1].reshape(gy.shape)
    ax1.streamplot(gx, gy, U, V, density=1.2, linewidth=0.6, arrowsize=0.7,
                   color="#cccccc", zorder=0)

    if ref_curves is not None:
        for lbl, curve in ref_curves:
            if "avg" in lbl.lower() and "demo" in lbl.lower():
                ax1.plot(curve[:, 0], curve[:, 1], label=lbl,
                         linewidth=1.4, alpha=0.95, color="red", zorder=1.5)
            else:
                ax1.plot(curve[:, 0], curve[:, 1], label=lbl,
                         linewidth=1.3, alpha=0.95, linestyle="--", color="gray", zorder=1.5)
    elif demo is not None:
        ax1.plot(demo[:, 0], demo[:, 1], color="red", lw=1.1, alpha=0.9,
                 label="Avg training demo", zorder=1.5)

    palette = ["#1f77b4", "#2ca02c", "#9467bd"]
    for k, (z, lbl) in enumerate(rollouts):
        ax1.plot(z[:, 0], z[:, 1], lw=2.3, label=lbl,
                 color=palette[k % len(palette)], zorder=3)

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.25)
    ax1.set_title(subtitle, fontsize=12)
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    # ax1.legend(loc="best", frameon=True, framealpha=0.9)
    ax1.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),  # (x center relative to axis, y below axis)
    frameon=True,
    framealpha=0.9
    )


    # --- bottom panel (disturbance)
    ax2 = fig.add_subplot(2, 1, 2)

    plotted = False
    if (t_norm is not None):
        # with_llc: matched / unmatched channels
        if (d_matched is not None) and len(d_matched) == len(t_norm):
            ax2.plot(t_norm, np.linalg.norm(d_matched, axis=1), lw=2.0, label="‖matched‖")
            plotted = True
        if (d_unmatched is not None) and len(d_unmatched) == len(t_norm):
            ax2.plot(t_norm, np.linalg.norm(d_unmatched, axis=1), lw=2.0, label="‖unmatched‖")
            plotted = True
        # no_llc: direct disturbance
        if (d_direct is not None) and len(d_direct) == len(t_norm):
            ax2.plot(t_norm, np.linalg.norm(d_direct, axis=1), lw=2.0, label="‖direct‖")
            plotted = True

    if plotted:
        ax2.set_xlim(0.0, 1.0)
        ax2.set_ylabel(r"disturbance magnitude")
        ax2.set_xlabel("normalized time")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Applied disturbance(s)", fontsize=11)
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "No disturbance logged", ha="center", va="center")
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
