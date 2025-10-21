# src/utils/plotting.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_xy_overlay(pos_ref: np.ndarray, pos_model: np.ndarray, savepath: Path, title: str):
    plt.figure(figsize=(5.2, 5))
    plt.plot(pos_ref[:,0], pos_ref[:,1], label="demo", lw=2, alpha=0.7, c="gray")
    plt.plot(pos_model[:,0], pos_model[:,1], label="model", lw=2, c="crimson")
    plt.scatter(pos_ref[0,0], pos_ref[0,1], s=40, c="k", marker="o", label="start")
    plt.axis("equal"); plt.grid(True); plt.legend()
    plt.title(title)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(savepath, dpi=200); plt.close()

def plot_loss_curve(history, savepath: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,3.3))
    plt.plot(history, lw=2)
    plt.xlabel("step"); plt.ylabel("MSE loss")
    plt.grid(True); plt.tight_layout()
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, dpi=200); plt.close()
