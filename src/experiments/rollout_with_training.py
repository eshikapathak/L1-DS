# src/experiments/rollout_with_training.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
import equinox as eqx

# Use your project modules
from models.neural_ode import NeuralODE
from src.train.train_node_periodic import load_shape, resample

def latest_model_path(models_root: Path, shape: str) -> Path:
    shape_dir = models_root / shape
    if not shape_dir.exists():
        raise FileNotFoundError(f"No directory for shape: {shape_dir}")
    # find latest run dir and model file
    run_dirs = sorted(shape_dir.glob("segcur_w*_d*_ntr*_ns*_lr*_seed*_*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        run_dirs = sorted(shape_dir.glob("segcur_w*_d*_ntr*_ns*_lr*_seed*_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders in {shape_dir}")
    run_dir = run_dirs[0]
    model_path = run_dir / f"{shape}_NODE_segcur.eqx"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path

def read_meta_for_dims(model_path: Path) -> tuple[int,int,int,int]:
    """Returns (width, depth, nsamples, ntrain) if meta.json present; else sensible defaults."""
    meta_p = model_path.parent / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        width = int(meta.get("width", 128))
        depth = int(meta.get("depth", 3))
        ns = int(meta.get("nsamples", 10000))
        ntr = int(meta.get("ntrain", 3))
        return width, depth, ns, ntr
    return 128, 3, 10000, 3

def rollout(model: NeuralODE, y0_xy: np.ndarray, T: int) -> np.ndarray:
    ts = jnp.linspace(0.0, 10.0, T)
    ys = np.array(model(ts, jnp.asarray(y0_xy)))
    return ys  # (T,2)

def plot_overlay(demos: list[np.ndarray], rollout_xy: np.ndarray, outpath: Path, title: str):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,6))
    # training demos
    for k, P in enumerate(demos):
        ax.plot(P[:,0], P[:,1], color="#bbbbbb", lw=1.0, alpha=0.9 if k==0 else 0.5,
                label="training demos" if k==0 else None)
    # rollout
    ax.plot(rollout_xy[:,0], rollout_xy[:,1], lw=2.2, label="NODE rollout")
    ax.scatter([rollout_xy[0,0]], [rollout_xy[0,1]], c="k", s=24, zorder=5, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.95, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser("Overlay NODE rollout with training demos.")
    ap.add_argument("--shape", required=True, type=str, help="e.g., IShape, RShape, Oshape, Sshape")
    ap.add_argument("--models_root", type=str, default="iros_outputs_auto_run/models")
    ap.add_argument("--model", type=str, default=None, help="Optional explicit model .eqx path")
    ap.add_argument("--data_root", type=str, default="IROS_dataset")
    ap.add_argument("--out_dir", type=str, default="iros_outputs_auto_run/rollout_plots")
    ap.add_argument("--nsamples", type=int, default=None, help="Override resample size (else use meta)")
    ap.add_argument("--ntrain", type=int, default=None, help="Override number of demos used (else use meta)")
    ap.add_argument("--width", type=int, default=None, help="Override width if meta missing")
    ap.add_argument("--depth", type=int, default=None, help="Override depth if meta missing")
    args = ap.parse_args()

    models_root = Path(args.models_root)
    model_path = Path(args.model) if args.model else latest_model_path(models_root, args.shape)
    width_m, depth_m, ns_m, ntr_m = read_meta_for_dims(model_path)

    width  = args.width or width_m
    depth  = args.depth or depth_m
    ns     = args.nsamples or ns_m
    ntrain = args.ntrain or ntr_m

    # load model with a template matching saved shapes
    key = jax.random.PRNGKey(0)
    template = NeuralODE(data_size=2, width_size=width, depth=depth, key=key)
    model = eqx.tree_deserialise_leaves(str(model_path), template)

    # load & resample data
    data = load_shape(args.shape, root=args.data_root)
    pos_rs, _, t_rs = resample(data, nsamples=ns)
    ntrain = min(ntrain, len(pos_rs))
    demos = [pos_rs[i] for i in range(ntrain)]

    # rollout from first point of demo 0
    y0 = demos[0][0]
    rollout_xy = rollout(model, y0, T=len(t_rs))

    out_dir = Path(args.out_dir) / args.shape
    out_path = out_dir / f"{args.shape}_rollout_vs_training.png"
    title = f"{args.shape} — NODE rollout vs training demos"
    plot_overlay(demos, rollout_xy, out_path, title)

    # also save raw arrays for later
    np.savez_compressed(out_dir / f"{args.shape}_rollout_vs_training.npz",
                        rollout=rollout_xy, demos=np.array(demos, dtype=float), t=t_rs)
    print(f"[OK] Wrote: {out_path}")

if __name__ == "__main__":
    main()
