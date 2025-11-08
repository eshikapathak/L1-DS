# scripts/plot_lasa_disturbances_gallery.py
from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
import sys 
from pathlib import Path 

# HERE = Path(__file__).resolve() 
# REPO_ROOT = HERE.parents[2] if (HERE.name == "disturbance_plotting.py") else HERE 
# sys.path.insert(0, str(REPO_ROOT))

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

    # --- Disturbances (LASA-style), tuned for ~10 visible cycles where applicable ---
    pulses_fn = two_mid_pulses(
        center1=0.30, width1=0.20, mag1=40.0, ax_gain1=(1.0, -1.0),
        center2=0.80, width2=0.50, mag2=50.0, ax_gain2=(1.0,  1.0),
    )
    pulses = np.stack([pulses_fn(t) for t in t_norm], axis=0)

    # Concentrate multisine components (adjusted band to show multiple cycles)
    M_MULTI_KW = dict(MAG=10.0, fmin=2.0, fmax=9.0)
    U_MULTI_KW = dict(MAG=10.0, fmin=2.0, fmax=9.0)

    m_multi_fn = make_matched_fn(on=True, kind="multisine", **M_MULTI_KW)
    m_multi = np.stack([m_multi_fn(t) for t in t_phys], axis=0)

    u_const_fn = make_unmatched_fn(on=True, kind="const", val=(8.0, 6.0))
    u_const = np.stack([u_const_fn(t) for t in t_phys], axis=0)

    u_multi_fn = make_unmatched_fn(on=True, kind="multisine", **U_MULTI_KW)
    u_multi = np.stack([u_multi_fn(t) for t in t_phys], axis=0)

    u_pulses_fn = make_unmatched_fn(on=True, kind="pulse", MAG=10.0, FREQ=10.0)
    m_multi_u_pulses = np.stack([m_multi_fn(t) + u_pulses_fn(t) for t in t_phys], axis=0)

    titles = [
        "(a) Steps",
        "(b) M. multisine",
        "(c) U. constant",
        "(d) U. multisine",
        "(e) M. multisine + U. pulses",
    ]
    time_axes = [t_norm, t_phys, t_phys, t_phys, t_phys]
    series = [pulses, m_multi, u_const, u_multi, m_multi_u_pulses]

    # --- Plot (no shared y-limits; keep each panel's natural scale) ---
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=False)

    for idx, (ax, tt, vec, title) in enumerate(zip(axes, time_axes, series, titles), start=1):
        line1, = ax.plot(tt, vec[:, 0], lw=1.8)            # x1 / matched
        line2, = ax.plot(tt, vec[:, 1], lw=1.8, ls="--")   # x2 / unmatched

        if idx == 1:
            ax.legend([line1, line2], [r"$\sigma(x_1)$", r"$\sigma(x_2)$"],
                      loc="upper right", frameon=True)
        else:
            ax.legend([line1, line2], [r"$d_m$", r"$d_um"],
                      loc="upper right", frameon=True)

        ax.set_xlim(0.0, 1.0)
        ymin, ymax = vec.min(), vec.max()
        pad = 0.5 * max(abs(ymin), abs(ymax), 1e-9)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(r"Normalized time $t \in [0,1]$")
        ax.tick_params(axis="y", which="both")

    axes[0].set_ylabel("Disturbance magnitude", labelpad=-5)

    plt.subplots_adjust(bottom=0.15, left=0.03, right=0.99, top=0.90, wspace=0.15)
    out = "lasa_disturbances_panels.png"
    fig.savefig(out, dpi=300)
    print(f"[OK] saved {out}")

if __name__ == "__main__":
    main()
