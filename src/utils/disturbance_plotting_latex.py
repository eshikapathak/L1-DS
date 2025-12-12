# scripts/plot_lasa_disturbances_gallery.py
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    # Font sizes
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 16,
})
import matplotlib.pyplot as plt
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2] if (HERE.name == "disturbance_plotting_latex.py") else HERE
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.disturbances_periodic import (
    two_mid_pulses,
    make_matched_fn,
    make_unmatched_fn,
)

def main():
    # 1000 samples over normalized time [0, 1]
    N = 1000
    t_norm = np.linspace(0.0, 1.0, N)  # used by two_mid_pulses(...) (normalized episode time)
    t_phys = t_norm.copy()             # "physical" time interpreted by matched/unmatched funcs

    # --- Disturbances (LASA-style), tuned for multiple visible cycles where applicable ---
    pulses_fn = two_mid_pulses(
        center1=0.30, width1=0.20, mag1=40.0, ax_gain1=(1.0, -1.0),
        center2=0.80, width2=0.50, mag2=50.0, ax_gain2=(1.0,  1.0),
    )
    pulses = np.stack([pulses_fn(t) for t in t_norm], axis=0)  # (N,2)

    # Concentrate multisine components (band chosen to show multiple cycles)
    M_MULTI_KW = dict(MAG=10.0, fmin=2.0, fmax=9.0)
    U_MULTI_KW = dict(MAG=10.0, fmin=2.0, fmax=9.0)

    # Matched multisine (d_m)
    m_multi_fn = make_matched_fn(on=True, kind="multisine", **M_MULTI_KW)
    m_multi = np.stack([m_multi_fn(t) for t in t_phys], axis=0)  # (N,2)

    # Unmatched constant (d_um) and unmatched multisine (d_um)
    u_const_fn = make_unmatched_fn(on=True, kind="const", val=(8.0, 6.0))
    u_const = np.stack([u_const_fn(t) for t in t_phys], axis=0)  # (N,2)

    u_multi_fn = make_unmatched_fn(on=True, kind="multisine", **U_MULTI_KW)
    u_multi = np.stack([u_multi_fn(t) for t in t_phys], axis=0)  # (N,2)

    # For panel (e): matched multisine and unmatched pulses (no summation; plot all four components)
    u_pulses_fn = make_unmatched_fn(on=True, kind="pulse", MAG=10.0, FREQ=10.0)
    u_pulses = np.stack([u_pulses_fn(t) for t in t_phys], axis=0)  # (N,2)

    titles = [
        "(a) Steps",
        "(b) M. multisine",
        "(c) U. constant",
        "(d) U. multisine",
        "(e) M. multisine + U. pulses",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=False)

    # (a) Direct task-space steps: show per-axis components
    ax = axes[0]
    l1, = ax.plot(t_norm, pulses[:, 0], lw=1.8)
    l2, = ax.plot(t_norm, pulses[:, 1], lw=1.8, ls="--")
    ax.legend([l1, l2], [r"$\sigma(x_1)$", r"$\sigma(x_2)$"], loc="upper right", frameon=True)
    ax.set_xlim(0.0, 1.0)
    ymin, ymax = pulses.min(), pulses.max()
    pad = 0.5 * max(abs(ymin), abs(ymax), 1e-9)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, alpha=0.3)
    ax.set_title(titles[0])
    ax.set_xlabel(r"Normalized time $t \in [0,1]$")
    ax.set_ylabel("Disturbance magnitude", labelpad=-5)
    ax.tick_params(axis="y", which="both")

    # (b) Matched multisine: plot d_{m,1}, d_{m,2}
    ax = axes[1]
    l1, = ax.plot(t_phys, m_multi[:, 0], lw=1.8)
    l2, = ax.plot(t_phys, m_multi[:, 1], lw=1.8, ls="--")
    ax.legend([l1, l2], [r"$d_{m,1}$", r"$d_{m,2}$"], loc="upper right", frameon=True)
    ax.set_xlim(0.0, 1.0)
    ymin, ymax = m_multi.min(), m_multi.max()
    pad = 0.5 * max(abs(ymin), abs(ymax), 1e-9)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, alpha=0.3)
    ax.set_title(titles[1])
    ax.set_xlabel(r"Normalized time $t \in [0,1]$")
    ax.tick_params(axis="y", which="both")

    # (c) Unmatched constant: plot d_{um,1}, d_{um,2}
    ax = axes[2]
    l1, = ax.plot(t_phys, u_const[:, 0], lw=1.8)
    l2, = ax.plot(t_phys, u_const[:, 1], lw=1.8, ls="--")
    ax.legend([l1, l2], [r"$d_{um,1}$", r"$d_{um,2}$"], loc="upper right", frameon=True)
    ax.set_xlim(0.0, 1.0)
    ymin, ymax = u_const.min(), u_const.max()
    pad = 0.5 * max(abs(ymin), abs(ymax), 1e-9)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, alpha=0.3)
    ax.set_title(titles[2])
    ax.set_xlabel(r"Normalized time $t \in [0,1]$")
    ax.tick_params(axis="y", which="both")

    # (d) Unmatched multisine: plot d_{um,1}, d_{um,2}
    ax = axes[3]
    l1, = ax.plot(t_phys, u_multi[:, 0], lw=1.8)
    l2, = ax.plot(t_phys, u_multi[:, 1], lw=1.8, ls="--")
    ax.legend([l1, l2], [r"$d_{um,1}$", r"$d_{um,2}$"], loc="upper right", frameon=True)
    ax.set_xlim(0.0, 1.0)
    ymin, ymax = u_multi.min(), u_multi.max()
    pad = 0.5 * max(abs(ymin), abs(ymax), 1e-9)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, alpha=0.3)
    ax.set_title(titles[3])
    ax.set_xlabel(r"Normalized time $t \in [0,1]$")
    ax.tick_params(axis="y", which="both")

    # (e) Matched multisine + Unmatched pulses: plot four lines, no summation
    ax = axes[4]
    l1, = ax.plot(t_phys, m_multi[:, 0], lw=1.8)
    l2, = ax.plot(t_phys, m_multi[:, 1], lw=1.8, ls="--")
    l3, = ax.plot(t_phys, u_pulses[:, 0], lw=1.2)
    l4, = ax.plot(t_phys, u_pulses[:, 1], lw=1.2, ls="--")
    ax.legend(
        [l1, l2, l3, l4],
        [r"$d_{m,1}$", r"$d_{m,2}$", r"$d_{um,1}$", r"$d_{um,2}$"],
        loc="upper right", frameon=True
    )
    ax.set_xlim(0.0, 1.0)
    ymin, ymax = np.min([m_multi.min(), u_pulses.min()]), np.max([m_multi.max(), u_pulses.max()])
    pad = 0.5 * max(abs(ymin), abs(ymax), 1e-9)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, alpha=0.3)
    ax.set_title(titles[4])
    ax.set_xlabel(r"Normalized time $t \in [0,1]$")
    ax.tick_params(axis="y", which="both")

    plt.subplots_adjust(bottom=0.15, left=0.03, right=0.99, top=0.90, wspace=0.15)
    out = "lasa_disturbances_panels.png"
    fig.savefig(out, dpi=300)
    print(f"[OK] saved {out}")

if __name__ == "__main__":
    main()
