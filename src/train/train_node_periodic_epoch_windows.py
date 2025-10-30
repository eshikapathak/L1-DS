# src/periodic/train_node_periodic.py

from __future__ import annotations

import argparse, json, time, os
from pathlib import Path
from typing import List, Tuple, Iterator
from decimal import Decimal, getcontext

import diffrax
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
import optax


# ---------------------------
# Minimal Neural ODE
# ---------------------------
class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        initializer = jnn.initializers.orthogonal()
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )
        # Orthogonal init for each layer weight (bias left default)
        key_weights = jr.split(key, depth + 1)
        for i in range(depth + 1):
            where = lambda m: m.layers[i].weight
            shape = self.mlp.layers[i].weight.shape
            self.mlp = eqx.tree_at(
                where,
                self.mlp,
                replace=initializer(key_weights[i], shape, dtype=jnp.float32),
            )

    @eqx.filter_jit
    def __call__(self, t, y, args):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys  # (len(ts), dim)


# ---------------------------
# Defaults (local paths)
# ---------------------------
DEFAULT_DATA_ROOT = "IROS_dataset"
DEFAULT_OUT_ROOT  = "outputs_periodic/new_training"


# =========================
# IROS dataset utilities
# =========================
def get_shape_files(root: str | Path):
    root = Path(root)
    return sorted(root.glob("*.npy"))

def get_shape_names(root: str | Path):
    return [p.stem for p in get_shape_files(root)]

def load_shape(shape: str, root: str | Path = "IROS_dataset") -> dict:
    """
    Returns:
      dict with keys:
        - demos: list of dict(pos(N,2), vel(N,2), t(N,))
        - name: shape name
    Accepts files that are (K,N,2) or (N,2). Time normalized to [0,1].
    """
    root = Path(root)
    path = root / f"{shape}.npy"

    if not path.exists():
        avail = ", ".join(get_shape_names(root))
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"data_root = {root}\n"
            f"Available shapes here: [{avail}]"
        )

    arr = np.asarray(np.load(path, allow_pickle=True))
    demos = []

    def _finite_diff(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
        dt = np.diff(t)
        dt = np.concatenate([dt[:1], dt, dt[-1:]])
        v = np.zeros_like(pos)
        v[1:-1] = (pos[2:] - pos[:-2]) / (t[2:] - t[:-2]).reshape(-1, 1)
        v[0] = (pos[1] - pos[0]) / dt[0]
        v[-1] = (pos[-1] - pos[-2]) / dt[-1]
        return v

    if arr.ndim == 3 and arr.shape[-1] >= 2:
        K, N, _ = arr.shape
        t = np.linspace(0.0, 1.0, N, dtype=float)
        for k in range(K):
            pos = np.asarray(arr[k, :, :2], dtype=float)
            vel = _finite_diff(pos, t)
            demos.append(dict(pos=pos, vel=vel, t=t.copy()))
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        N = arr.shape[0]
        t = np.linspace(0.0, 1.0, N, dtype=float)
        pos = np.asarray(arr[:, :2], dtype=float)
        vel = _finite_diff(pos, t)
        demos.append(dict(pos=pos, vel=vel, t=t.copy()))
    else:
        raise ValueError(f"Unsupported IROS array shape: {arr.shape}")

    return dict(name=shape, demos=demos)

def resample(data: dict, nsamples: int = 2000) -> Tuple[list, list, np.ndarray]:
    """Uniform resample to (nsamples,). Returns lists pos_rs[k], vel_rs[k], and shared t(nsamples,)."""
    t_new = np.linspace(0.0, 1.0, nsamples, dtype=float)
    pos_rs, vel_rs = [], []
    from scipy import interpolate
    for d in data["demos"]:
        t, P, V = d["t"], d["pos"], d["vel"]
        fx = interpolate.interp1d(t, P[:, 0], kind="linear")
        fy = interpolate.interp1d(t, P[:, 1], kind="linear")
        fvx = interpolate.interp1d(t, V[:, 0], kind="linear")
        fvy = interpolate.interp1d(t, V[:, 1], kind="linear")
        pos_rs.append(np.stack([fx(t_new),  fy(t_new)],  axis=1))
        vel_rs.append(np.stack([fvx(t_new), fvy(t_new)], axis=1))
    return pos_rs, vel_rs, t_new


# =========================
# Plot helpers
# =========================
def _plot_xy_overlay(ref_xy: np.ndarray, pred_xy: np.ndarray, outpath: Path, title: str):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], color="red", lw=1.2, label="reference segment")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#1f77b4", lw=2.0, label="NODE rollout")
    ax.scatter([ref_xy[0, 0]], [ref_xy[0, 1]], c="k", s=18, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=210); plt.close(fig)

def _plot_loss_curve(loss_hist, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 3.2))
    plt.plot(loss_hist, lw=1.8)
    plt.xlabel("step"); plt.ylabel("train loss (MSE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

def _plot_vector_field(model, bounds, outpath: Path, title="Vector field"):
    (xmin, xmax), (ymin, ymax) = bounds
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 28), np.linspace(ymin, ymax, 28))
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = jnp.asarray([X[i, j], Y[i, j]])
            u = np.array(model.func(0.0, p, None))
            U[i, j], V[i, j] = u[0], u[1]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.streamplot(X, Y, U, V, density=1.1, color="#bfbfbf", linewidth=0.7, arrowsize=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(outpath, dpi=220); plt.close(fig)


# =========================
# Segment curriculum helpers
# =========================
def _fmt_lr_decimal(x) -> str:
    getcontext().prec = 24
    s = format(Decimal(str(x)), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s

def _make_bounds_from_trajs(trajs: np.ndarray, pad_ratio: float = 0.12):
    # trajs: (B, T, 2)
    xmin = float(np.min(trajs[..., 0])); xmax = float(np.max(trajs[..., 0]))
    ymin = float(np.min(trajs[..., 1])); ymax = float(np.max(trajs[..., 1]))
    xr = xmax - xmin; yr = ymax - ymin
    pad = pad_ratio * max(xr, yr if (xr or yr) else 1.0)
    return (xmin - pad, xmax + pad), (ymin - pad, ymax + pad)

def compute_starts(T: int, seg_len: int, stride: int) -> List[int]:
    """Enumerate window start indices with given stride."""
    if seg_len > T:
        seg_len = T
    return list(range(0, max(1, T - seg_len + 1), max(1, stride)))

def count_windows(T: int, seg_len: int, stride: int) -> int:
    if seg_len > T: seg_len = T
    return max(1, (T - seg_len) // max(1, stride) + 1)


# =========================
# Loaders
# =========================
def mixed_window_loader(
    Xfull: jnp.ndarray,          # (B, T, 2)
    seg_len: int,
    batch_size: int,
    key,
    hist_starts: List[int],      # accumulated from earlier stages
    curr_starts: List[int],      # current stage's starts
    alpha: float                 # prob of sampling from history
) -> Iterator[jnp.ndarray]:
    """
    Yields (Bsz, seg_len, 2), mixing historical and current starts.
    Historical starts are filtered to be valid for the current seg_len.
    """
    B, T, _ = Xfull.shape
    valid_hist = [s for s in hist_starts if s + seg_len <= T]
    valid_curr = [s for s in curr_starts if s + seg_len <= T]
    if not valid_curr:
        valid_curr = compute_starts(T, seg_len, stride=max(1, seg_len // 5))
    k = key
    while True:
        use_hist = jr.bernoulli(k, p=jnp.clip(alpha, 0.0, 1.0), shape=(batch_size,))
        k, = jr.split(k, 1)
        batch = []
        for uh in np.array(use_hist):
            pool = valid_hist if (bool(uh) and len(valid_hist) > 0) else valid_curr
            bi = int(jr.randint(k, (), 0, B)); k, = jr.split(k, 1)
            si = int(pool[int(jr.randint(k, (), 0, len(pool)))]); k, = jr.split(k, 1)
            seg = Xfull[bi, si:si + seg_len, :]
            batch.append(seg)
        yield jnp.asarray(np.stack(batch, axis=0))  # (Bsz, seg_len, 2)

def sequential_window_loader(
    Xfull: np.ndarray,           # (B, T, 2)
    seg_len: int,
    batch_size: int,
    stride: int
) -> Iterator[jnp.ndarray]:
    """
    Deterministic sliding windows: left->right with step 'stride'.
    Batches are built by consecutive windows in order across demos.
    """
    B, T, _ = Xfull.shape
    starts = compute_starts(T, seg_len, stride=stride)
    if not starts:
        starts = [0]
    pairs = []
    for s in starts:
        for b in range(B):
            pairs.append((b, s))
    idx = 0
    while True:
        batch = []
        for _ in range(batch_size):
            b, s = pairs[idx % len(pairs)]
            idx += 1
            batch.append(Xfull[b, s:s + seg_len, :])
        yield jnp.asarray(np.stack(batch, axis=0))  # (Bsz, seg_len, 2)


# =========================
# Training with epochs-over-windows (mixed or sequential)
# =========================
def train_one(
    shape: str = "IShape",
    data_root: str = DEFAULT_DATA_ROOT,
    out_root: str  = DEFAULT_OUT_ROOT,
    nsamples: int = 10000,
    ntrain: int = 4,
    seed: int = 1234,
    width: int = 128,
    depth: int = 3,
    # global safety cap (optional). If None, no global cap.
    steps: int | None = 3000000,
    base_lr: float = 5e-4,
    batch_size: int = 16,
    curriculum_fracs: str = "0.20,0.35,0.55,0.75,1.00",
    # Mixed (random) mode overlap for current stage: stride ≈ overlap_frac * seg_len
    overlap_frac: float = 0.2,
    # Replay mix schedule per stage (only for mixed mode)
    alpha_start: float = 0.4,
    alpha_end: float   = 0.1,
    # Windowing mode
    window_mode: str = "mixed",    # "mixed" (random+replay) or "sequential"
    stride_frac: float = 0.10,     # sequential: stride as fraction of T_full
    epochs_per_stage: float = 10.0,# epochs over windows per stage (both modes)
    print_every: int = 200,
    save_every: int = 2000,
):
    # --- data ---
    data = load_shape(shape, root=data_root)
    pos_rs, _, t_rs = resample(data, nsamples=nsamples)
    K = len(pos_rs)
    if ntrain > K: ntrain = K
    train_idxs = list(range(ntrain))

    Y_full = np.stack([pos_rs[i] for i in train_idxs], axis=0)  # (B, T, 2)
    T_full = Y_full.shape[1]
    # t_full = np.asarray(t_rs, dtype=float)  # not used directly below

    # --- model/opt ---
    key = jr.PRNGKey(seed)
    model_key, loader_key = jr.split(key, 2)
    model = NeuralODE(data_size=2, width_size=width, depth=depth, key=model_key)

    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adabelief(learning_rate=base_lr),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # --- outputs ---
    out_root = Path(out_root)
    run_id = f"segcur_w{width}_d{depth}_ntr{ntrain}_ns{nsamples}_lr{_fmt_lr_decimal(base_lr)}_seed{seed}_{window_mode}"
    out_dir = out_root / shape / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- losses/logging ---
    loss_hist = []
    bounds = _make_bounds_from_trajs(Y_full)

    # --- loss/step ---
    @eqx.filter_value_and_grad
    def loss_fn(m, ti, Yi):
        Y0 = Yi[:, 0, :]
        pred = jax.vmap(m, in_axes=(None, 0))(ti, Y0)
        return jnp.mean((pred - Yi) ** 2)

    @eqx.filter_jit
    def train_step(m, opt_state, ti, Yi):
        l, g = loss_fn(m, ti, Yi)
        updates, opt_state = optim.update(g, opt_state)
        m = eqx.apply_updates(m, updates)
        return l, m, opt_state

    # --- curriculum (fractions → lengths) ---
    fracs = [float(s.strip()) for s in curriculum_fracs.split(",") if s.strip()]
    fracs = [max(0.05, min(1.0, f)) for f in fracs]
    fracs = sorted(fracs)
    total_stages = len(fracs)

    step_count = 0
    t0 = time.time()

    # Only for mixed mode: carry history of starts
    hist_starts: List[int] = []

    for si, frac in enumerate(fracs, start=1):
        seg_len = max(8, int(round(frac * T_full)))
        ti = jnp.linspace(0.0, 1.0, seg_len)

        if window_mode == "sequential":
            stride = max(1, int(round(stride_frac * T_full)))
            nwin = count_windows(T_full, seg_len, stride)
            steps_per_epoch = int(np.ceil((Y_full.shape[0] * nwin) / batch_size))
            stage_steps = max(1, int(np.ceil(epochs_per_stage * steps_per_epoch)))
            loader = sequential_window_loader(Y_full, seg_len=seg_len, batch_size=batch_size, stride=stride)

            print(f"[{shape}] >>> Stage {si}/{total_stages} (SEQ) | frac={frac:.2f} "
                  f"| seg_len={seg_len} stride={stride} | windows/demo={nwin} "
                  f"| steps/epoch={steps_per_epoch} | epochs={epochs_per_stage} | planned_steps={stage_steps}")

            for _ in range(stage_steps):
                Yi = next(loader)
                loss, model, opt_state = train_step(model, opt_state, ti, Yi)
                loss_hist.append(float(loss))
                step_count += 1

                if (step_count % print_every) == 0:
                    print(f"[{shape}] stage {si}/{total_stages} frac={frac:.2f} "
                          f"| SEQ seg_len={seg_len} stride={stride} "
                          f"| step {step_count} | loss={float(loss):.6f}")

                if (step_count % save_every) == 0:
                    s0 = 0
                    ref_seg = np.array(Y_full[0, s0:s0 + seg_len, :], dtype=float)
                    roll = np.array(model(ti, jnp.asarray(ref_seg[0])))
                    _plot_xy_overlay(ref_seg, roll, out_dir / f"overlay_step{step_count}.png",
                                     f"{shape} — step {step_count} (seg_len={seg_len}, sequential)")

                if steps is not None and step_count >= steps:
                    break

        else:
            # Mixed mode: random current + replay history
            curr_stride = max(1, int(round(overlap_frac * seg_len)))
            curr_starts = compute_starts(T_full, seg_len, stride=curr_stride)
            # Approximate epoch: all demos × all current starts
            nwin = count_windows(T_full, seg_len, curr_stride)
            steps_per_epoch = int(np.ceil((Y_full.shape[0] * nwin) / batch_size))
            stage_steps = max(1, int(np.ceil(epochs_per_stage * steps_per_epoch)))

            print(f"[{shape}] >>> Stage {si}/{total_stages} (MIX) | frac={frac:.2f} "
                  f"| seg_len={seg_len} stride≈{curr_stride} | curr_windows/demo={nwin} "
                  f"| steps/epoch≈{steps_per_epoch} | epochs={epochs_per_stage} | planned_steps≈{stage_steps}")

            def alpha_at(k: int) -> float:
                if stage_steps <= 1: return alpha_end
                a = alpha_start + (alpha_end - alpha_start) * (k / (stage_steps - 1))
                return float(max(0.0, min(1.0, a)))

            loader = mixed_window_loader(
                Xfull=jnp.asarray(Y_full),
                seg_len=seg_len,
                batch_size=batch_size,
                key=loader_key,
                hist_starts=hist_starts,
                curr_starts=curr_starts,
                alpha=alpha_start,  # initial p(history)
            )

            for k in range(stage_steps):
                # Update alpha linearly over stage
                alpha = alpha_at(k)
                # Rebuild a loader with new alpha cheaply:
                # (cheap enough; or pass alpha into step and sample bernoulli there)
                Yi = next(mixed_window_loader(
                    Xfull=jnp.asarray(Y_full),
                    seg_len=seg_len,
                    batch_size=batch_size,
                    key=loader_key,
                    hist_starts=hist_starts,
                    curr_starts=curr_starts,
                    alpha=alpha,
                ))
                loss, model, opt_state = train_step(model, opt_state, ti, Yi)
                loss_hist.append(float(loss))
                step_count += 1

                if (step_count % print_every) == 0:
                    print(f"[{shape}] stage {si}/{total_stages} frac={frac:.2f} "
                          f"| MIX seg_len={seg_len} stride≈{curr_stride} alpha={alpha:.2f} "
                          f"| step {step_count} | loss={float(loss):.6f}")

                if (step_count % save_every) == 0:
                    s0 = (T_full - seg_len) // 3 if seg_len < T_full else 0
                    ref_seg = np.array(Y_full[0, s0:s0 + seg_len, :], dtype=float)
                    roll = np.array(model(ti, jnp.asarray(ref_seg[0])))
                    _plot_xy_overlay(ref_seg, roll, out_dir / f"overlay_step{step_count}.png",
                                     f"{shape} — step {step_count} (seg_len={seg_len}, mixed)")

                if steps is not None and step_count >= steps:
                    break

            # add current starts to history
            hist_starts = sorted(set(hist_starts).union(curr_starts))

        if steps is not None and step_count >= steps:
            break

    dur = time.time() - t0
    print(f"[{shape}] Done in {dur:.1f}s | final loss={loss_hist[-1]:.6e}")

    # --- save artifacts ---
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{shape}_NODE_segcur.eqx"
    eqx.tree_serialise_leaves(str(model_path), model)
    _plot_loss_curve(loss_hist, out_dir / "loss_curve.png")
    _plot_vector_field(model, bounds, out_dir / "vector_field.png", title=f"{shape} — learned vector field")

    # final qualitative (rollout full from demo 0 start)
    demo0 = np.array(Y_full[0], dtype=float)
    roll_full = np.array(model(jnp.linspace(0.0, 1.0, demo0.shape[0]), jnp.asarray(demo0[0])))
    _plot_xy_overlay(demo0, roll_full, out_dir / "overlay_full.png",
                     f"{shape} — rollout from first point (full length)")

    np.savez_compressed(out_dir / "preview_arrays.npz",
                        pos_true=demo0, pos_pred=roll_full, t=np.linspace(0.0, 1.0, demo0.shape[0]))
    meta = dict(
        shape=shape, nsamples=nsamples, ntrain=ntrain, seed=seed,
        width=width, depth=depth, order=1,
        steps=steps, batch_size=batch_size, base_lr=base_lr,
        curriculum_fracs=fracs, overlap_frac=overlap_frac,
        alpha_start=alpha_start, alpha_end=alpha_end,
        window_mode=window_mode, stride_frac=stride_frac, epochs_per_stage=epochs_per_stage,
        seconds=float(dur),
        paths=dict(
            model=str(model_path),
            loss=str(out_dir / "loss_curve.png"),
            overlay_full=str(out_dir / "overlay_full.png"),
            vector_field=str(out_dir / "vector_field.png"),
            arrays=str(out_dir / "preview_arrays.npz"),
            dir=str(out_dir),
        ),
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved model to: {model_path}")
    return str(model_path)


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser("Train NODE on IROS trajectories with epochs-over-windows curriculum.")
    p.add_argument("--shape", type=str, default="IShape")
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--nsamples", type=int, default=10000)
    p.add_argument("--ntrain", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--steps", type=int, default=3000000,
                   help="Global safety cap on total updates. Set large or None to disable.")
    p.add_argument("--base_lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--curriculum_fracs", type=str, default="0.20,0.35,0.55,0.75,1.00")
    p.add_argument("--overlap_frac", type=float, default=0.2)
    p.add_argument("--alpha_start", type=float, default=0.4)
    p.add_argument("--alpha_end", type=float, default=0.1)
    p.add_argument("--window_mode", type=str, default="mixed", choices=["mixed", "sequential"])
    p.add_argument("--stride_frac", type=float, default=0.10,
                   help="Sequential: stride as fraction of T_full.")
    p.add_argument("--epochs_per_stage", type=float, default=10.0,
                   help="Epochs over windows per stage (both modes).")
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = vars(args)
    # allow disabling global cap by passing negative or zero
    if cfg.get("steps", None) is not None and cfg["steps"] <= 0:
        cfg["steps"] = None
    train_one(**cfg)


if __name__ == "__main__":
    main()


# python src/periodic/train_node_periodic_epoch_windows.py \
#   --shape IShape \
#   --data_root IROS_dataset \
#   --out_root outputs_periodic/new_training \
#   --nsamples 10000 \
#   --ntrain 3 \
#   --window_mode mixed \
#   --curriculum_fracs "0.10,0.20,0.35,0.55,0.75,1.00" \
#   --epochs_per_stage 5 \
#   --batch_size 16 \
#   --overlap_frac 0.20 \
#   --alpha_start 0.50 \
#   --alpha_end 0.10 \
#   --steps 200000 \
#   --print_every 200 \
#   --save_every 5000

python src/periodic/train_node_periodic_epoch_windows.py \
  --shape IShape \
  --data_root IROS_dataset \
  --out_root outputs_periodic/new_training \
  --nsamples 10000 \
  --ntrain 3 \
  --window_mode sequential \
  --curriculum_fracs "0.10,0.20,0.35,0.55,0.75,1.00" \
  --stride_frac 0.10 \
  --epochs_per_stage 3 \
  --batch_size 16 \
  --steps 150000 \
  --print_every 200 \
  --save_every 5000
