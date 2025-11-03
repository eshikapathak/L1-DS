# src/utils/plot_all_disturbances.py
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- make 'src' importable even if run from repo root or elsewhere ---
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2] if (HERE.name == "plot_all_disturbances.py") else HERE
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.disturbances_periodic import (
    make_matched_fn, make_unmatched_fn, make_direct_fn,
    big_mid_pulse, two_mid_pulses
)

OUT_DIR = REPO_ROOT / "iros_outputs_auto_run" / "disturbance_previews"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- common time grids ----
T_world = 8.0
N = 2000
t = np.linspace(0.0, T_world, N, dtype=float)
t_norm = np.linspace(0.0, 1.0, N, dtype=float)

def _plot_vec(name: str, f, tvec, fname: Path, title: str):
    y = np.stack([f(tt) for tt in tvec], axis=0)  # (N,2)
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
    ax[0].plot(tvec, y[:,0], lw=1.4); ax[0].set_ylabel("x-comp")
    ax[1].plot(tvec, y[:,1], lw=1.4); ax[1].set_ylabel("y-comp")
    ax[1].set_xlabel("t")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=220); plt.close(fig)

def plot_all():
    # ---- matched channel ----
    matched_kinds = ["const","sine","square","pulse","multisine","chirp","noise"]
    for k in matched_kinds:
        f = make_matched_fn(on=True, kind=k)  # defaults
        _plot_vec(f"matched_{k}", f, t, OUT_DIR / f"matched_{k}.png",
                  f"Matched: {k} (defaults)")

    # ---- unmatched channel ----
    unmatched_kinds = ["const","sine","square","pulse","multisine","chirp","noise"]
    for k in unmatched_kinds:
        f = make_unmatched_fn(on=True, kind=k)  # defaults
        _plot_vec(f"unmatched_{k}", f, t, OUT_DIR / f"unmatched_{k}.png",
                  f"Unmatched: {k} (defaults)")

    # ---- direct channel (no_llc) ----
    direct_kinds = ["const","sine","square","pulse","multisine","chirp","noise"]
    for k in direct_kinds:
        f = make_direct_fn(on=True, kind=k)  # defaults
        _plot_vec(f"direct_{k}", f, t, OUT_DIR / f"direct_{k}.png",
                  f"Direct/no_llc: {k} (defaults)")

    # ---- normalized window pulses (for reference) ----
    f_big = big_mid_pulse()          # t in [0,1]
    f_two = two_mid_pulses()
    _plot_vec("big_mid_pulse", f_big, t_norm, OUT_DIR / "big_mid_pulse.png",
              "Windowed big_mid_pulse on t∈[0,1]")
    _plot_vec("two_mid_pulses", f_two, t_norm, OUT_DIR / "two_mid_pulses.png",
              "Windowed two_mid_pulses on t∈[0,1]")

    print(f"[OK] Wrote previews to: {OUT_DIR}")

if __name__ == "__main__":
    plot_all()
