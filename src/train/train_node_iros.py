# src/train/train_node_periodic.py
from __future__ import annotations

import argparse, json, time, os
from pathlib import Path
from typing import List, Tuple, Iterator
from decimal import Decimal, getcontext

import sys

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import optax
import diffrax

# -----------------------
# Minimal Neural ODE
# -----------------------
class Func(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, data_size, width_size, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=data_size, out_size=data_size,
            width_size=width_size, depth=depth,
            activation=jnn.tanh, key=key
        )
    def __call__(self, t, y, args):
        return self.mlp(y)

class NeuralODE(eqx.Module):
    func: Func
    def __init__(self, data_size, width_size, depth, *, key):
        self.func = Func(data_size, width_size, depth, key=key)
    def __call__(self, ts, y0):
        # ts: (L,), y0: (2,)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func), diffrax.Tsit5(),
            t0=ts[0], t1=ts[-1], dt0=ts[1] - ts[0], y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys  # (L,2)

# Defaults (repo-root relative)
DEFAULT_DATA_ROOT = "src/data/IROS_dataset"
DEFAULT_OUT_ROOT  = "src/data/IROS_dataset/iros_outputs_auto_run/models"

# -----------------------
# IROS dataset utils
# -----------------------
def _finite_diff(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    v = np.zeros_like(pos)
    # central diffs internal, one-sided ends
    v[1:-1] = (pos[2:] - pos[:-2]) / (t[2:, None] - t[:-2, None])
    v[0] = (pos[1] - pos[0]) / (t[1] - t[0])
    v[-1] = (pos[-1] - pos[-2]) / (t[-1] - t[-2])
    return v

def get_shape_files(root: str | Path):
    root = Path(root); return sorted(root.glob("*.npy"))

def get_shape_names(root: str | Path):
    return [p.stem for p in get_shape_files(root)]

def load_shape(shape: str, root: str | Path = DEFAULT_DATA_ROOT) -> dict:
    root = Path(root); path = root / f"{shape}.npy"
    if not path.exists():
        avail = ", ".join(get_shape_names(root))
        raise FileNotFoundError(f"{path} not found. data_root={root}. Available: [{avail}]")
    arr = np.asarray(np.load(path, allow_pickle=True))
    demos = []
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        K, N, _ = arr.shape; t = np.linspace(0.0, 1.0, N, dtype=float)
        for k in range(K):
            P = np.asarray(arr[k, :, :2], dtype=float)
            V = _finite_diff(P, t); demos.append(dict(pos=P, vel=V, t=t.copy()))
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        N = arr.shape[0]; t = np.linspace(0.0, 1.0, N, dtype=float)
        P = np.asarray(arr[:, :2], dtype=float); V = _finite_diff(P, t)
        demos.append(dict(pos=P, vel=V, t=t.copy()))
    else:
        raise ValueError(f"Unsupported IROS array shape: {arr.shape}")
    return dict(name=shape, demos=demos)

def resample(data: dict, nsamples: int = 2000) -> Tuple[list, list, np.ndarray]:
    # returns lists pos_rs[k], vel_rs[k] each (nsamples,2) and shared t_new (nsamples,)
    t_new = np.linspace(0.0, 1.0, nsamples, dtype=float)
    pos_rs, vel_rs = [], []
    from scipy import interpolate
    for d in data["demos"]:
        t, P, V = d["t"], d["pos"], d["vel"]
        fx = interpolate.interp1d(t, P[:, 0], kind="linear")
        fy = interpolate.interp1d(t, P[:, 1], kind="linear")
        fvx = interpolate.interp1d(t, V[:, 0], kind="linear")
        fvy = interpolate.interp1d(t, V[:, 1], kind="linear")
        pos_rs.append(np.stack([fx(t_new), fy(t_new)], axis=1))
        vel_rs.append(np.stack([fvx(t_new), fvy(t_new)], axis=1))
    return pos_rs, vel_rs, t_new

# -----------------------
# Plot helpers
# -----------------------
def _plot_xy_overlay(ref_xy, pred_xy, outpath: Path, title: str):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], color="red", lw=1.2, label="reference")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], lw=2.0, label="NODE rollout")
    ax.scatter([ref_xy[0, 0]], [ref_xy[0, 1]], c="k", s=18, label="start")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title(title); ax.legend(); fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)

def _plot_loss_curve(loss_hist, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 3.2))
    plt.plot(loss_hist, lw=1.8)
    plt.xlabel("step"); plt.ylabel("train loss (MSE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

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
    ax.set_aspect("equal"); ax.set_title(title); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(outpath, dpi=220); plt.close(fig)

# -----------------------
# Segment helpers
# -----------------------
def _fmt_lr_decimal(x) -> str:
    getcontext().prec = 24
    s = format(Decimal(str(x)), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s

def _make_bounds_from_trajs(trajs: np.ndarray, pad_ratio: float = 0.12):
    xmin = float(np.min(trajs[..., 0])); xmax = float(np.max(trajs[..., 0]))
    ymin = float(np.min(trajs[..., 1])); ymax = float(np.max(trajs[..., 1]))
    xr = xmax - xmin; yr = ymax - ymin
    pad = pad_ratio * max(xr, yr if (xr or yr) else 1.0)
    return (xmin - pad, xmax + pad), (ymin - pad, ymax + pad)

def compute_starts(T: int, seg_len: int, stride: int):
    if seg_len > T: seg_len = T
    return list(range(0, max(1, T - seg_len + 1), max(1, stride)))

# -----------------------
# Loaders (yield segments + start indices)
# -----------------------
def mixed_window_loader(
    Xfull: jnp.ndarray,          # (B, T, 2)
    seg_len: int,
    batch_size: int,
    key,
    hist_starts: List[int],      # accumulated from earlier stages
    curr_starts: List[int],      # current stage's starts
    alpha: float                 # prob of sampling from history
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    B, T, _ = Xfull.shape
    valid_hist = [s for s in hist_starts if s + seg_len <= T]
    valid_curr = [s for s in curr_starts if s + seg_len <= T]
    if not valid_curr:
        valid_curr = compute_starts(T, seg_len, stride=max(1, seg_len // 5))
    k = key
    while True:
        use_hist = jr.bernoulli(k, p=jnp.clip(alpha, 0.0, 1.0), shape=(batch_size,))
        k, = jr.split(k, 1)
        batch, starts = [], []
        for uh in np.array(use_hist):
            pool = valid_hist if (bool(uh) and len(valid_hist) > 0) else valid_curr
            bi = int(jr.randint(k, (), 0, B)); k, = jr.split(k, 1)
            si = int(pool[int(jr.randint(k, (), 0, len(pool)))]); k, = jr.split(k, 1)
            seg = Xfull[bi, si:si + seg_len, :]
            batch.append(seg); starts.append(si)
        yield jnp.asarray(np.stack(batch, axis=0)), jnp.asarray(starts)

def sequential_window_loader(
    Xfull: np.ndarray,           # (B, T, 2)
    seg_len: int,
    batch_size: int,
    stride: int
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    B, T, _ = Xfull.shape
    starts = compute_starts(T, seg_len, stride=stride) or [0]
    pairs = [(b, s) for s in starts for b in range(B)]
    idx = 0
    while True:
        batch, st = [], []
        for _ in range(batch_size):
            b, s = pairs[idx % len(pairs)]; idx += 1
            batch.append(Xfull[b, s:s + seg_len, :]); st.append(s)
        yield jnp.asarray(np.stack(batch, axis=0)), jnp.asarray(st)

# -----------------------
# Training
# -----------------------
def train_one(
    shape: str = "IShape",
    data_root: str = DEFAULT_DATA_ROOT,
    out_root: str  = DEFAULT_OUT_ROOT,
    nsamples: int = 10000,
    ntrain: int = 3,
    seed: int = 1234,
    width: int = 128,
    depth: int = 3,
    steps: int = 120000,
    base_lr: float = 5e-4,
    batch_size: int = 16,
    curriculum_fracs: str = "0.1,0.1,0.2,0.2,0.2",
    overlap_frac: float = 0.2,
    alpha_start: float = 0.4,
    alpha_end: float   = 0.1,
    window_mode: str = "mixed",      # "mixed" or "sequential"
    stride_frac: float = 0.10,       # sequential mode stride (fraction of T_full)
    epochs_per_stage: int = 1,       # sequential mode epochs per stage
    print_every: int = 500,
    save_every: int = 5000,
):
    # Data
    data = load_shape(shape, root=data_root)
    pos_rs, _, t_full_np = resample(data, nsamples=nsamples)
    ntrain = min(ntrain, len(pos_rs))
    Y_full = np.stack([pos_rs[i] for i in range(ntrain)], axis=0)  # (B,T,2)
    T_full = Y_full.shape[1]
    t_full = jnp.asarray(t_full_np)  # global grid in [0,1]

    # Model/opt
    key = jr.PRNGKey(seed); mkey, lkey = jr.split(key, 2)
    model = NeuralODE(data_size=2, width_size=width, depth=depth, key=mkey)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adabelief(learning_rate=base_lr))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Outputs
    out_root = Path(out_root)
    run_id = f"segcur_w{width}_d{depth}_ntr{ntrain}_ns{nsamples}_lr{_fmt_lr_decimal(base_lr)}_seed{seed}_{window_mode}"
    out_dir = out_root / shape / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss/step with per-sample time grids
    @eqx.filter_value_and_grad
    def loss_fn(m, Ti_b, Yi):
        # Ti_b: (Bseg, L) sliced from global t_full; Yi: (Bseg, L, 2)
        Y0 = Yi[:, 0, :]                           # (Bseg, 2)
        # vmap over samples: (L,) with (2,)
        pred = jax.vmap(m, in_axes=(0, 0))(Ti_b, Y0)  # (Bseg, L, 2)
        return jnp.mean((pred - Yi) ** 2)

    @eqx.filter_jit
    def train_step(m, opt_state, Ti_b, Yi):
        l, g = loss_fn(m, Ti_b, Yi)
        updates, opt_state = optim.update(g, opt_state)
        m = eqx.apply_updates(m, updates)
        return l, m, opt_state

    # Curriculum schedule (fractions → lengths; KEEP ORDER GIVEN)
    fracs = [float(s.strip()) for s in curriculum_fracs.split(",") if s.strip()]
    fracs = [max(0.05, min(1.0, f)) for f in fracs]

    total_stages = len(fracs)
    step_count = 0
    loss_hist: List[float] = []
    hist_starts: List[int] = []
    t0 = time.time()

    print(f"[{shape}] Training start: total steps={steps}, stages={total_stages}, ntrain={ntrain}, batch_size={batch_size}", flush=True)

    for si, frac in enumerate(fracs, start=1):
        seg_len = max(8, int(round(frac * T_full)))
        stage_steps = max(1, steps // total_stages) if si < total_stages else (steps - step_count)

        print(f"[{shape}] Starting stage {si}/{total_stages}: frac={frac:.3f}, seg_len={seg_len}, planned_steps={stage_steps}", flush=True)

        if window_mode == "sequential":
            stride = max(1, int(round(stride_frac * T_full)))
            loader = sequential_window_loader(Y_full, seg_len=seg_len, batch_size=batch_size, stride=stride)

            B = Y_full.shape[0]
            num_starts = max(1, len(compute_starts(T_full, seg_len, stride)))
            steps_per_epoch = int(np.ceil((B * num_starts) / batch_size))
            total_iters = steps_per_epoch * max(1, int(epochs_per_stage))
            iters = range(min(stage_steps, total_iters))

            for _ in iters:
                Yi, starts = next(loader)                     # (Bsz,L,2), (Bsz,)
                # per-sample time grid from the SAME global t_full (no renorm)
                Ti_b = jnp.stack([t_full[s:s+seg_len] for s in starts], axis=0)  # (Bsz,L)
                loss, model, opt_state = train_step(model, opt_state, Ti_b, Yi)
                loss_hist.append(float(loss)); step_count += 1

                if (step_count % print_every) == 0 or step_count == steps:
                      print(f"[{shape}] stage {si}/{total_stages} frac={frac:.2f} "
                          f"| SEQ seg_len={seg_len} stride={stride} "
                          f"| step {step_count:06d}/{steps} | loss={float(loss):.6f}", flush=True)

                if (step_count % save_every) == 0 or step_count == steps:
                    s0 = 0
                    ref_seg = np.array(Y_full[0, s0:s0+seg_len, :], dtype=float)
                    ti = np.array(t_full_np[s0:s0+seg_len], dtype=float)
                    roll = np.array(model(ti, jnp.asarray(ref_seg[0])))
                    _plot_xy_overlay(ref_seg, roll, out_dir / f"overlay_step{step_count}.png",
                                     f"{shape} — step {step_count} (seg_len={seg_len}, sequential)")

                if step_count >= steps: break

        else:
            # Mixed (random + replay)
            curr_starts = compute_starts(T_full, seg_len, stride=max(1, int(round(overlap_frac * seg_len))))

            def alpha_at(k: int) -> float:
                if stage_steps <= 1: return alpha_end
                a = alpha_start + (alpha_end - alpha_start) * (k / (stage_steps - 1))
                return float(max(0.0, min(1.0, a)))

            for k in range(stage_steps):
                alpha = alpha_at(k)
                Yi, starts = next(mixed_window_loader(
                    Xfull=jnp.asarray(Y_full),
                    seg_len=seg_len,
                    batch_size=batch_size,
                    key=lkey,
                    hist_starts=hist_starts,
                    curr_starts=curr_starts,
                    alpha=alpha,
                ))
                Ti_b = jnp.stack([t_full[s:s+seg_len] for s in starts], axis=0)
                loss, model, opt_state = train_step(model, opt_state, Ti_b, Yi)
                loss_hist.append(float(loss)); step_count += 1

                if (step_count % print_every) == 0 or step_count == steps:
                      print(f"[{shape}] stage {si}/{total_stages} frac={frac:.2f} "
                          f"| MIX seg_len={seg_len} stride≈{int(overlap_frac*seg_len)} alpha={alpha:.2f} "
                          f"| step {step_count:06d}/{steps} | loss={float(loss):.6f}", flush=True)

                if (step_count % save_every) == 0 or step_count == steps:
                    s0 = (T_full - seg_len) // 3 if seg_len < T_full else 0
                    ref_seg = np.array(Y_full[0, s0:s0+seg_len, :], dtype=float)
                    ti = np.array(t_full_np[s0:s0+seg_len], dtype=float)
                    roll = np.array(model(ti, jnp.asarray(ref_seg[0])))
                    _plot_xy_overlay(ref_seg, roll, out_dir / f"overlay_step{step_count}.png",
                                     f"{shape} — step {step_count} (seg_len={seg_len}, mixed)")

                if step_count >= steps: break

            # carry current starts into history
            hist_starts = sorted(set(hist_starts).union(curr_starts))

        if step_count >= steps: break

    dur = time.time() - t0
    print(f"[{shape}] Done in {dur:.1f}s | final loss={loss_hist[-1]:.6e}", flush=True)

    # save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{shape}_NODE_segcur.eqx"
    eqx.tree_serialise_leaves(str(model_path), model)
    _plot_loss_curve(loss_hist, out_dir / "loss_curve.png")
    _plot_vector_field(model, _make_bounds_from_trajs(Y_full), out_dir / "vector_field.png",
                       title=f"{shape} — learned vector field")

    # qualitative: full-trajectory rollout from first point (global t_full)
    demo0 = np.array(Y_full[0], dtype=float)
    roll_full = np.array(model(np.array(t_full_np), jnp.asarray(demo0[0])))
    _plot_xy_overlay(demo0, roll_full, out_dir / "overlay_full.png",
                     f"{shape} — rollout from first point (full length)")

    meta = dict(shape=shape, nsamples=nsamples, ntrain=ntrain, seed=seed, width=width, depth=depth,
                steps=steps, batch_size=batch_size, base_lr=base_lr, curriculum_fracs=fracs,
                window_mode=window_mode, seconds=float(dur),
                paths=dict(model=str(model_path), dir=str(out_dir)))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] Saved model to: {model_path}", flush=True)
    return str(model_path)

def parse_args():
    p = argparse.ArgumentParser("Train NODE on IROS trajectories with segment curriculum (global time slicing).")
    p.add_argument("--shape", type=str, default="IShape")
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--nsamples", type=int, default=10000)
    p.add_argument("--ntrain", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--steps", type=int, default=120000)
    p.add_argument("--base_lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--curriculum_fracs", type=str, default="0.1,0.1,0.2,0.2,0.2")
    p.add_argument("--overlap_frac", type=float, default=0.2)
    p.add_argument("--alpha_start", type=float, default=0.4)
    p.add_argument("--alpha_end", type=float, default=0.1)
    p.add_argument("--window_mode", type=str, default="mixed", choices=["mixed", "sequential"])
    p.add_argument("--stride_frac", type=float, default=0.10)
    p.add_argument("--epochs_per_stage", type=int, default=1)
    p.add_argument("--print_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=5000)
    return p.parse_args()

def main():
    args = parse_args()
    train_one(**vars(args))

if __name__ == "__main__":
    main()
