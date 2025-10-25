# src/periodic/train_node_periodic.py
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from decimal import Decimal, getcontext
from typing import Tuple, Iterator

import numpy as np
import matplotlib.pyplot as plt

import jax, jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from models.neural_ode import NeuralODE
from src.periodic.iros import load_shape, resample

# ---------------------------
# small helpers
# ---------------------------
def _fmt_lr_decimal(x) -> str:
    getcontext().prec = 24
    s = format(Decimal(str(x)), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s

def dataloader(arrays: Tuple[jnp.ndarray, ...], batch_size: int, key) -> Iterator[Tuple[jnp.ndarray, ...]]:
    (X,) = arrays
    N = X.shape[0]
    idxs = jnp.arange(N)
    k = key
    while True:
        perm = jr.permutation(k, idxs)
        k, = jr.split(k, 1)
        for i in range(0, N, batch_size):
            sl = perm[i:i + batch_size]
            if sl.shape[0] == batch_size:
                yield (X[sl],)

def _plot_xy_overlay(ref_xy: np.ndarray, pred_xy: np.ndarray, outpath: Path, title: str):
    """Thin, fast overlay like LASA trainer."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], color="red", lw=1.2, label="training demo")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#1f77b4", lw=2.2, label="NODE rollout")
    ax.scatter([ref_xy[0, 0]], [ref_xy[0, 1]], c="k", s=28, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=210)
    plt.close(fig)

def _plot_loss_curve(loss_hist, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 3.6))
    plt.plot(loss_hist, lw=1.8)
    plt.xlabel("step"); plt.ylabel("train loss (MSE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ---------------------------
# trainer
# ---------------------------
def train_one(
    shape: str = "IShape",
    data_root: str = "iros_dataset",
    nsamples: int = 1000,
    ntrain: int = 4,
    seed: int = 1385,
    width: int = 100,
    depth: int = 3,
    order: int = 1,                 # 1st-order NODE
    steps: int = 20000,
    batch_size: int = 2,
    base_lr: float = 5e-4,
    curriculum_steps: int = 200,
    print_every: int = 100,
    save_every: int = 1000,
    out_root: str = "outputs_periodic",
    models_root: str = "outputs_periodic/models",
):
    # --- data
    data = load_shape(shape, root=data_root)
    pos_rs, vel_rs, t_rs = resample(data, nsamples=nsamples)  # lists
    K = len(pos_rs)
    if ntrain > K:
        ntrain = K
    train_idxs = list(range(ntrain))

    # training tensor (B, T, 2)
    ys = np.stack([pos_rs[i] for i in train_idxs], axis=0)
    ts = np.asarray(t_rs, dtype=float)
    T = ts.shape[0]

    # --- output dirs
    out_root = Path(out_root)
    models_root = Path(models_root)
    run_id = (
        f"order{order}_w{width}_d{depth}_ntr{ntrain}_ns{nsamples}"
        f"_curr{curriculum_steps}_lr{_fmt_lr_decimal(base_lr)}_seed{seed}"
    )
    out_dir = out_root / shape / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    # --- model & opt
    key = jr.PRNGKey(seed)
    model_key, loader_key = jr.split(key, 2)
    model = NeuralODE(data_size=2, width_size=width, depth=depth, key=model_key)

    sched = optax.cosine_decay_schedule(init_value=base_lr, decay_steps=steps, alpha=0.95)
    optim = optax.adabelief(learning_rate=sched)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    Xtrain = jnp.asarray(ys)  # (B,T,2)
    t_full = jnp.asarray(ts)
    t_short = t_full[: int(0.30 * T)]  # curriculum like before

    @eqx.filter_value_and_grad
    def loss_fn(m, ti, Yi):
        Y0 = Yi[:, 0]                                  # (B,2)
        pred = jax.vmap(m, in_axes=(None, 0))(ti, Y0)  # (B,len(ti),2)
        return jnp.mean((pred - Yi[:, : ti.shape[0], :]) ** 2)

    @eqx.filter_jit
    def train_step(m, opt_state, ti, Yi):
        l, g = loss_fn(m, ti, Yi)
        updates, opt_state = optim.update(g, opt_state)
        m = eqx.apply_updates(m, updates)
        return l, m, opt_state

    dl = dataloader((Xtrain,), batch_size=batch_size, key=loader_key)
    loss_hist = []

    # pick a fixed demo for intermediate preview (first training demo)
    demo_ref = np.array(pos_rs[train_idxs[0]], dtype=float)  # (T,2)

    t_start = time.time()
    for step in range(1, steps + 1):
        (Yi,) = next(dl)
        ti = t_short if step <= curriculum_steps else t_full
        loss, model, opt_state = train_step(model, opt_state, ti, Yi)
        loss_hist.append(float(loss))

        if (step % print_every) == 0 or step == steps:
            print(f"[{shape}] step {step:06d}/{steps} | loss {float(loss):.6f}")

        # quick overlay snapshot like LASA trainer
        if (step % save_every) == 0 or step == steps:
            rollout = np.array(model(t_full, jnp.asarray(demo_ref[0])))
            _plot_xy_overlay(
                demo_ref,
                rollout,
                out_dir / f"train_overlay_step{step}.png",
                f"{shape} — step {step} (order=1)",
            )

    dur = time.time() - t_start
    print(f"[{shape}] Done in {dur:.1f}s | final loss={loss_hist[-1]:.6e}")

    # --- save model + meta + curves
    model_path = out_dir / f"{shape}_NODE_w{width}_d{depth}_ntr{ntrain}_ns{nsamples}.eqx"
    eqx.tree_serialise_leaves(str(model_path), model)
    _plot_loss_curve(loss_hist, out_dir / "loss_curve.png")

    # also drop a final preview with vector field + arrays for convenience
    rollout = np.array(model(t_full, jnp.asarray(demo_ref[0])))
    np.savez_compressed(out_dir / "train_preview.npz", t=np.asarray(t_full), pos_true=demo_ref, pos_pred=rollout)

    meta = dict(
        shape=shape, nsamples=nsamples, ntrain=ntrain, seed=seed,
        width=width, depth=depth, order=order,
        steps=steps, batch_size=batch_size, base_lr=base_lr,
        curriculum_steps=curriculum_steps, seconds=float(dur),
        files=dict(
            model=str(model_path),
            loss_curve=str(out_dir / "loss_curve.png"),
            preview=str(out_dir / "train_overlay_step{step}.png"),
            arrays=str(out_dir / "train_preview.npz"),
        ),
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved model to: {model_path}")
    return str(model_path)

# ---------------------------
# CLI
# ---------------------------
def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=str, default="configs_periodic/IShape/node_train.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.yaml)
    train_one(**cfg)

if __name__ == "__main__":
    main()
