from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- Metrics --------

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

def time_in_tube(P, Q, radius: float):
    d = np.linalg.norm(P - Q, axis=1)
    return np.mean(d <= radius)

def phase_drift_deg(P, Q):
    p0 = P - P.mean(0); q0 = Q - Q.mean(0)
    ang_p = np.unwrap(np.arctan2(p0[:,1], p0[:,0]))
    ang_q = np.unwrap(np.arctan2(q0[:,1], q0[:,0]))
    L = min(len(ang_p), len(ang_q))
    return np.degrees(ang_p[:L].mean() - ang_q[:L].mean())

# -------- Plots --------

def _compute_field_grid(field_fn, bounds, grid=25):
    (xmin, xmax), (ymin, ymax) = bounds
    X, Y = np.meshgrid(np.linspace(xmin, xmax, grid), np.linspace(ymin, ymax, grid))
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u = field_fn(np.array([X[i, j], Y[i, j]]))  # returns (2,)
            U[i, j], V[i, j] = u[0], u[1]
    return X, Y, U, V

def plot_rollouts_with_field(rows_labels, cols_cells, titles, outpath: Path,
                             field_fn=None, field_bounds=None, demo=None, subtitle=None):
    """
    rows_labels: list[str] (controllers)
    cols_cells: list[list[(z, label)]] per column
    titles: list[str]
    field_fn: callable(p)->vel (for streamplot background), optional
    field_bounds: ((xmin,xmax),(ymin,ymax)) for field plot window
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
                sp = ax.streamplot(X, Y, U, V, density=1.0, color="#cccccc", linewidth=0.7, arrowsize=0.7)
            if demo is not None:
                ax.plot(demo[:,0], demo[:,1], c="0.6", lw=2, label="demo")
            z, lbl = cols_cells[j][i]
            ax.plot(z[:,0], z[:,1], lw=2.0, label=lbl)
            if i == 0:
                ax.set_title(titles[j])
            ax.axis("equal"); ax.grid(True, alpha=0.3)
            if j == 0: ax.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outpath, dpi=240); plt.close()

def bar_with_ci(names, vals, ci, ylabel, title, outpath):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(names))
    w = 0.65
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(x, vals, yerr=ci, width=w, capsize=5, alpha=0.9)

    # annotate values
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
    rollouts: list of tuples [(z_array, label), ...]
      - z_array: (N, 2) executed trajectory
    demo: (N, 2) demo trajectory
    field_fn: function R^2 -> R^2 giving the learned drift at a point
    field_bounds: ((xmin,xmax),(ymin,ymax))
    """
    import numpy as np
    import matplotlib.pyplot as plt

    (xmin, xmax), (ymin, ymax) = field_bounds
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # --- light vector field background
    gx, gy = np.meshgrid(
        np.linspace(xmin, xmax, 22),
        np.linspace(ymin, ymax, 22),
    )
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    vv = np.stack([field_fn(p) for p in pts], axis=0)  # (M,2)
    U = vv[:, 0].reshape(gx.shape)
    V = vv[:, 1].reshape(gy.shape)
    ax.streamplot(gx, gy, U, V, density=1.2, linewidth=0.6, arrowsize=0.7, color="#cccccc", zorder=0)

    # --- demo (thin red)
    ax.plot(demo[:, 0], demo[:, 1], color="red", lw=1.1, alpha=0.9, label="demo", zorder=2)

    # --- rollouts
    palette = ["#1f77b4", "#2ca02c", "#9467bd"]  # blue, green, purple
    for k, (z, lbl) in enumerate(rollouts):
        color = palette[k % len(palette)]
        ax.plot(z[:, 0], z[:, 1], lw=2.3, label=lbl, color=color, zorder=3)

    # cosmetics
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Qualitative rollout\n{subtitle}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_all_together_with_dist(rollouts, demo, field_fn, field_bounds,
                                subtitle, outpath, t_norm=None, d_direct=None):
    """
    rollouts: [(z_array, label), ...]
    demo: (N,2)  -- here, the average training demo
    field_fn: R^2 -> R^2
    field_bounds: ((xmin,xmax),(ymin,ymax))
    t_norm: (N,) normalized time 0..1
    d_direct: (N,2) disturbance samples to plot magnitude over time
    """
    import numpy as np
    import matplotlib.pyplot as plt

    (xmin, xmax), (ymin, ymax) = field_bounds
    fig = plt.figure(figsize=(6.6, 8.2))

    # --- top: trajectories with field
    ax1 = fig.add_subplot(2, 1, 1)
    gx, gy = np.meshgrid(
        np.linspace(xmin, xmax, 22),
        np.linspace(ymin, ymax, 22),
    )
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    vv = np.stack([field_fn(p) for p in pts], axis=0)
    U = vv[:, 0].reshape(gx.shape)
    V = vv[:, 1].reshape(gy.shape)
    ax1.streamplot(gx, gy, U, V, density=1.2, linewidth=0.6, arrowsize=0.7, color="#cccccc", zorder=0)

    # demo average (thin red)
    ax1.plot(demo[:, 0], demo[:, 1], color="red", lw=1.1, alpha=0.9, label="training avg demo", zorder=2)

    palette = ["#1f77b4", "#2ca02c", "#9467bd"]
    for k, (z, lbl) in enumerate(rollouts):
        ax1.plot(z[:, 0], z[:, 1], lw=2.3, label=lbl, color=palette[k % len(palette)], zorder=3)

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.25)
    ax1.set_title(subtitle, fontsize=12)   # paper-friendly, no “qualitative” wording
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(loc="best", frameon=True, framealpha=0.9)

    # --- bottom: disturbance magnitude over time
    ax2 = fig.add_subplot(2, 1, 2)
    if (t_norm is not None) and (d_direct is not None) and len(t_norm) == len(d_direct):
        mag = np.linalg.norm(d_direct, axis=1)
        ax2.plot(t_norm, mag, lw=2.0)
        ax2.set_xlim(0.0, 1.0)
        ax2.set_ylabel(r"$\|d(t)\|$")
        ax2.set_xlabel("normalized time")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Applied lower-level disturbance", fontsize=11)
    else:
        ax2.text(0.5, 0.5, "No disturbance logged", ha="center", va="center")
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

