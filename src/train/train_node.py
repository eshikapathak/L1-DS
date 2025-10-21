# src/train/train_node.py
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from models.neural_ode import NeuralODE               # first-order model (2->2)
from models.neural_ode2 import NeuralODE2nd           # second-order model (4->4)
from src.data.lasa import get_shape_names, load_shape, resample, train_test_split
from src.utils.plotting import plot_loss_curve, plot_xy_overlay
from src.utils.seed import make_seeds


# ---------------------------
# CLI / config
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Neural ODE on LASA shapes (by name).")
    p.add_argument("--config", type=str, default=None, help="Optional YAML with defaults.")

    # Core data/model params
    p.add_argument("--shape", type=str, default=None, help="One of: " + ", ".join(get_shape_names()))
    p.add_argument("--nsamples", type=int, default=None)
    p.add_argument("--ntrain", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--base_lr", type=float, default=None)
    p.add_argument("--curriculum_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_root", type=str, default=None)

    # Logging cadence
    p.add_argument("--print_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)

    # 1st vs 2nd order
    p.add_argument("--order", type=int, default=None, choices=[1, 2],
                   help="1: pos->vel (2D state). 2: [pos,vel]->[vel,acc] (4D state).")
    p.add_argument("--lambda_vel", type=float, default=None,
                   help="weight on velocity MSE for 2nd-order training")

    return p.parse_args()


# ---------------------------
# Data utils
# ---------------------------
def dataloader(arrays: Tuple[jnp.ndarray, ...], batch_size: int, key) -> Iterator[Tuple[jnp.ndarray, ...]]:
    """Infinite sampler over first dimension (demo index). Yields full sequences."""
    X, = arrays
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


# ---------------------------
# Trainer
# ---------------------------
def train_one(
    *,
    shape: str | None = None,
    shape_name: str | None = None,
    nsamples: int = 1000,
    ntrain: int = 4,
    width: int = 64,
    depth: int = 3,
    steps: int = 5000,
    batch_size: int = 2,
    base_lr: float = 3e-4,
    curriculum_steps: int = 1000,
    seed: int = 1000,
    out_root: str = "outputs",
    print_every: int = 100,
    save_every: int = 1000,
    order: int = 1,          # 1 or 2
    lambda_vel: float = 1.0, # only used if order == 2
    **_ignore,               # ignore unknown keys from YAML/CLI
):
    # Resolve shape name
    shape_name = shape or shape_name
    assert shape_name is not None, "Provide a LASA shape via --shape (or shape_name)."

    # ---------- Output dir ----------
    out_root = Path(out_root)
    run_id = f"order{order}_w{width}_d{depth}_ntr{ntrain}_ns{nsamples}_curr{curriculum_steps}_lr{base_lr}_seed{seed}"
    out_dir = out_root / f"{shape_name}/{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Data ----------
    ds = load_shape(shape_name)
    pos_rs, vel_rs, t_rs = resample(ds, nsamples=nsamples)
    # t_rs = ds.t[:nsamples]
    (pos_tr, vel_tr), (pos_te, vel_te) = train_test_split(pos_rs, vel_rs, ntrain=ntrain)

    # ---------- Model ----------
    model_key, loader_key = make_seeds(seed, 2)
    if order == 1:
        model = NeuralODE(data_size=2, width_size=width, depth=depth, key=model_key)
        Xtrain = jnp.array(pos_tr)  # (N,T,2)
    else:
        model = NeuralODE2nd(width_size=width, depth=depth, key=model_key)
        Xtrain = jnp.concatenate([pos_tr, vel_tr], axis=-1)  # (N,T,4) = [pos,vel]

    # ---------- Optim ----------
    sched = optax.cosine_decay_schedule(init_value=base_lr, decay_steps=steps, alpha=0.95)
    optim = optax.adabelief(learning_rate=sched)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # ---------- Loss ----------
    if order == 1:
        @eqx.filter_value_and_grad
        def loss_fn(m, ti, Yi):
            # Yi: (B,T,2) positions
            Y0 = Yi[:, 0]                                 # (B,2)
            pred = jax.vmap(m, in_axes=(None, 0))(ti, Y0) # (B,T,2)
            return jnp.mean((pred - Yi) ** 2)
    else:
        @eqx.filter_value_and_grad
        def loss_fn(m: NeuralODE2nd, ti, Yi):
            # Yi: (B,T,4) = [pos, vel]
            Y0 = Yi[:, 0]                                   # (B,4)
            pred = jax.vmap(m, in_axes=(None, 0))(ti, Y0)   # (B,T,4)
            pos_pred, vel_pred = pred[:, :, :2], pred[:, :, 2:]
            pos_true, vel_true = Yi[:, :, :2], Yi[:, :, 2:]
            pos_mse = jnp.mean((pos_pred - pos_true) ** 2)
            vel_mse = jnp.mean((vel_pred - vel_true) ** 2)
            return pos_mse + lambda_vel * vel_mse

    @eqx.filter_jit
    def train_step(m, opt_state, ti, Yi):
        l, g = loss_fn(m, ti, Yi)
        updates, opt_state = optim.update(g, opt_state)
        m = eqx.apply_updates(m, updates)
        return l, m, opt_state

    # ---------- Loop ----------
    dl = dataloader((Xtrain,), batch_size=batch_size, key=loader_key)
    hist = []
    tvec_full = jnp.array(t_rs)
    tvec_short = tvec_full[: int(0.25 * len(tvec_full))]  # curriculum

    t0 = time.time()
    for step in range(1, steps + 1):
        (Yi,) = next(dl)  # (B,T,2) or (B,T,4)
        ti = tvec_short if step <= curriculum_steps else tvec_full
        Yi = Yi[:, : ti.shape[0], :]

        loss, model, opt_state = train_step(model, opt_state, ti, Yi)
        hist.append(float(loss))

        if (step % print_every) == 0 or step == steps:
            print(f"[{shape_name}] order={order} step {step:05d}/{steps} loss={float(loss):.6f}")

        if (step % save_every) == 0 or step == steps:
            # quick qualitative overlay (positions)
            if order == 1:
                Yref = pos_tr[0]
                Yhat = np.array(model(tvec_full, Yref[0]))               # (T,2)
            else:
                Yref = pos_tr[0]
                Y0 = np.concatenate([pos_tr[0, 0], vel_tr[0, 0]], axis=-1)  # (4,)
                Yroll = np.array(model(tvec_full, Y0))                   # (T,4)
                Yhat = Yroll[:, :2]
            plot_xy_overlay(Yref, Yhat, out_dir / f"train_overlay_step{step}.png",
                            f"{shape_name} — step {step} (order={order})")

    t1 = time.time()
    print(f"Done in {(t1 - t0):.1f}s. Final loss={hist[-1]:.6e}")

    # ---------- Save artifacts ----------
    suffix = f"{'NODE' if order==1 else 'NODE2nd'}_w{width}_d{depth}_ntr{ntrain}_ns{nsamples}"
    model_path = out_dir / f"{shape_name}_{suffix}.eqx"
    eqx.tree_serialise_leaves(model_path.as_posix(), model)

    plot_loss_curve(hist, out_dir / "loss_curve.png")

    # Extra velocity overlay for order==2 (held-out demo if available)
    if order == 2 and pos_te.shape[0] > 0:
        import matplotlib.pyplot as plt
        Y0_te = np.concatenate([pos_te[0, 0], vel_te[0, 0]], axis=-1)
        Yroll = np.array(model(tvec_full, Y0_te))   # (T,4)
        vel_hat = Yroll[:, 2:]
        plt.figure(figsize=(6, 3.3))
        plt.plot(tvec_full, vel_te[0, :, 0], label="vx demo")
        plt.plot(tvec_full, vel_hat[:, 0], label="vx model")
        plt.plot(tvec_full, vel_te[0, :, 1], label="vy demo")
        plt.plot(tvec_full, vel_hat[:, 1], label="vy model")
        plt.legend(); plt.grid(True); plt.tight_layout()
        (out_dir / "vel_overlay.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "vel_overlay.png", dpi=200); plt.close()

    # Metadata
    meta = dict(
        shape=shape_name,
        nsamples=nsamples,
        ntrain=ntrain,
        width=width,
        depth=depth,
        steps=steps,
        batch_size=batch_size,
        base_lr=base_lr,
        curriculum_steps=curriculum_steps,
        seed=seed,
        order=order,
        lambda_vel=lambda_vel,
        t_len=len(t_rs),
        dt_nominal=ds.dt,
        files=dict(
            model=str(model_path),
            loss=str(out_dir / "loss_curve.png"),
            overlay_train=str(out_dir / f"train_overlay_step{steps}.png"),
            vel_overlay=str(out_dir / "vel_overlay.png") if (order == 2 and pos_te.shape[0] > 0) else None,
        ),
    )
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return model_path, out_dir


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    args = parse_args()

    # Load YAML if provided (lazy import so pyyaml is optional unless used)
    merged = {}
    if getattr(args, "config", None):
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "PyYAML is not installed. Run `pip install pyyaml` or remove the --config flag."
            ) from e
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
        merged = deepcopy(cfg); merged.update(cli)
    else:
        merged = {k: v for k, v in vars(args).items() if v is not None and k != "config"}

    # Defaults if not provided (good standalone behavior without YAML)
    merged.setdefault("shape", "Worm")
    merged.setdefault("nsamples", 1000)
    merged.setdefault("ntrain", 4)
    merged.setdefault("width", 64)
    merged.setdefault("depth", 3)
    merged.setdefault("steps", 5000)
    merged.setdefault("batch_size", 2)
    merged.setdefault("base_lr", 3e-4)
    merged.setdefault("curriculum_steps", 1000)
    merged.setdefault("seed", 1385)
    merged.setdefault("out_root", "outputs")
    merged.setdefault("print_every", 100)
    merged.setdefault("save_every", 1000)
    merged.setdefault("order", 1)
    merged.setdefault("lambda_vel", 1.0)

    # train_one accepts both 'shape' and 'shape_name'
    train_one(**merged)
