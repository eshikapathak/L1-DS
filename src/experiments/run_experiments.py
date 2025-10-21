from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx

from models.neural_ode import NeuralODE
from src.data.lasa import load_shape, resample
from .targets import TargetDTW
from .robust_ctrl import L1Adaptive
from .disturbances import big_mid_pulse
from .simulator import SimConfig, simulate
from .metrics_plots import dtw_distance, plot_all_together_with_dist, bar_with_ci


# ------------------------ tiny logger ------------------------
def _now(): return time.strftime("%H:%M:%S")
def log(msg: str, *, enabled: bool = True):
    if enabled: print(f"[{_now()}] {msg}", flush=True)


# ------------------------ helpers ------------------------
def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _fmt_lr_decimal(x) -> str:
    from decimal import Decimal, getcontext
    getcontext().prec = 24
    s = format(Decimal(str(x)), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s

def expected_model_dir(cfg: dict) -> Path:
    shape = cfg.get("shape", "Worm")
    order = int(cfg.get("order", 1)); width = int(cfg.get("width", 64)); depth = int(cfg.get("depth", 3))
    ntrain = int(cfg.get("ntrain", 4)); nsamp = int(cfg.get("nsamples", 1000)); seed = int(cfg.get("seed", 1000))
    curr = int(cfg.get("curriculum_steps", 1000)); base_lr_str = _fmt_lr_decimal(cfg.get("base_lr", 3e-4))
    out_root = Path(cfg.get("out_root", "outputs"))
    run_id = f"order{order}_w{width}_d{depth}_ntr{ntrain}_ns{nsamp}_curr{curr}_lr{base_lr_str}_seed{seed}"
    return out_root / f"{shape}/{run_id}"

def expected_model_path(cfg: dict) -> Path:
    run_dir = expected_model_dir(cfg)
    shape = cfg.get("shape", "Worm")
    order = int(cfg.get("order", 1)); width = int(cfg.get("width", 64)); depth = int(cfg.get("depth", 3))
    ntrain = int(cfg.get("ntrain", 4)); nsamp = int(cfg.get("nsamples", 1000))
    canonical = run_dir / f"{shape}_{'NODE' if order==1 else 'NODE2nd'}_w{width}_d{depth}_ntr{ntrain}_ns{nsamp}.eqx"
    if canonical.exists(): return canonical
    eqx_files = sorted(run_dir.glob("*.eqx"))
    if not eqx_files: raise FileNotFoundError(f"No .eqx found under: {run_dir}")
    return eqx_files[0]


# ------------------------ build a NODE reference (pos/vel) ------------------------
def rollout_node_reference(model, z0: np.ndarray, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discrete Euler rollout of NODE: z_{k+1}=z_k + dt * f(z_k). Returns (pos, vel_hat)."""
    t = np.asarray(t_grid).reshape(-1)
    dt = float((t[-1]-t[0]) / max(1, len(t)-1))
    Z = np.zeros((len(t)-1, 2), float)
    Vh = np.zeros_like(Z)
    z = z0.astype(float).copy()
    for k in range(len(t)-1):
        v_hat = np.array(model.func(0.0, jnp.asarray(z), None))
        Vh[k] = v_hat
        z = z + dt * v_hat
        Z[k] = z
    return Z, Vh


# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_yaml", type=str, default="configs/Worm/node_train.yaml")
    ap.add_argument("--model", type=str, default=None, help="Optional: explicit .eqx path")
    ap.add_argument("--out", type=str, default="outputs/experiments/Worm_suite")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--with_llc", action="store_true",
                help="Use plant + PD low-level controller (LLC) instead of direct/no_llc mode.")
    ap.add_argument("--matched", action="store_true",
                    help="Enable matched lower-level disturbance (acts on acceleration).")
    ap.add_argument("--unmatched", action="store_true",
                    help="Enable unmatched lower-level disturbance (acts on position rate).")
    ap.add_argument("--matched_type", type=str, default="sine", choices=["sine","pulse","const"])
    ap.add_argument("--unmatched_type", type=str, default="sine", choices=["sine","pulse","const"])

    args = ap.parse_args()

    # --- load config + model
    cfg_tr = load_yaml(args.train_yaml)
    log(f"Loaded train YAML: {args.train_yaml}", enabled=args.verbose)
    model_path = Path(args.model) if args.model else expected_model_path(cfg_tr)
    log(f"Using model: {model_path}", enabled=args.verbose)

    order = int(cfg_tr.get("order", 1)); width = int(cfg_tr.get("width", 64)); depth = int(cfg_tr.get("depth", 3))
    key = jax.random.PRNGKey(0)
    template = NeuralODE(data_size=2, width_size=width, depth=depth, key=key)
    model = eqx.tree_deserialise_leaves(str(model_path), template)
    log("Model deserialized successfully.", enabled=args.verbose)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # --- LASA data + resample
    shape = cfg_tr.get("shape", "Worm")
    lasa = load_shape(shape)
    nsamples = int(cfg_tr.get("nsamples", 1000))
    ntrain = int(cfg_tr.get("ntrain", 4))
    pos_rs, vel_rs, t_rs = resample(lasa, nsamples=nsamples)  # lists of arrays, same length / time grid

        # --- choose training demos indices: match your earlier convention [1:ntrain]
    train_idxs = list(range(1, ntrain)) if ntrain > 1 else [0]

    # --- average training demos (paper curve + DTW reference)
    demo_avg_pos = np.mean([pos_rs[i] for i in train_idxs], axis=0)   # (N,2)
    demo_avg_vel = np.mean([vel_rs[i] for i in train_idxs], axis=0)   # (N,2)

    # --- time grid from the averaged demo length (normalized to [0,1])
    demo_t = np.linspace(0.0, 1.0, demo_avg_pos.shape[0], dtype=float)

    # --- initial condition = first sample of the average demo
    init_state = np.hstack([demo_avg_pos[0], demo_avg_vel[0]])        # [px0,py0,vx0,vy0]

    # --- CLF reference must come from a NODE rollout from the *same averaged init*
    node_ref_pos, node_ref_vel = rollout_node_reference(
        model,
        z0=demo_avg_pos[0],  # same avg init point
        t_grid=demo_t
    )

    # ---- selector factories use NODE reference (not raw LASA) and same avg init
    class _SelWrap:
        def __init__(self, dt):
            self.sel = TargetDTW(node_ref_pos, node_ref_vel, W=50, H=40)
            self.sel.init_from(node_ref_pos[0])                # init on NODE ref
            self.l1 = L1Adaptive(Ts=dt, a=10.0, omega=12.0, x0=node_ref_pos[0])
        def get(self, hist): return self.sel.get(hist)

    def selector_with_l1(dt): return _SelWrap(dt)
    def selector_no_l1(dt):
        s = _SelWrap(dt)
        delattr(s, "l1")
        return s


    # ---- vector field function
    def field_fn(p_xy: np.ndarray):
        return np.array(model.func(0.0, jnp.asarray(p_xy), None))

    # ---- pretty bounds around the average demo
    x_range = float(np.ptp(demo_avg_pos[:, 0])); y_range = float(np.ptp(demo_avg_pos[:, 1]))
    pad = 0.15 * max(x_range, y_range) if max(x_range, y_range) > 0 else 0.1
    bounds = (
        (float(np.min(demo_avg_pos[:, 0])) - pad, float(np.max(demo_avg_pos[:, 0])) + pad),
        (float(np.min(demo_avg_pos[:, 1])) - pad, float(np.max(demo_avg_pos[:, 1])) + pad),
    )

    # --- BIG mid pulse disturbance (from disturbances.py)
    d_fn = big_mid_pulse(center=0.5, width=0.30, mag=30.0, ax_gain=(1.0, 0.8))
    dist_desc = "Lower-level disturbance: mid-interval rectangular pulse (mag 2.0, width 30%)"
    log(f"Disturbance: {dist_desc}", enabled=args.verbose)

    # ---- simulate 3 controllers with auto time base from demo_t
    controllers = [
        ("NODE",       dict(use_clf=False, use_l1=False), selector_no_l1),
        ("NODE+CLF",   dict(use_clf=True,  use_l1=False), selector_no_l1),
        ("NODE+CLF+L1",dict(use_clf=True,  use_l1=True ), selector_with_l1),
    ]

    cols = []
    names=[]; dtw_vals=[]
    last_logs = None
    for i, (name, flags, sel_factory) in enumerate(controllers, start=1):
        log(f"[{i}/{len(controllers)}] Running controller: {name}", enabled=args.verbose)

        # cfg = SimConfig(dt=None, T=None, target_mode="dtw", no_llc=True,
        #                 use_clf=flags["use_clf"], use_l1=flags["use_l1"])
        cfg = SimConfig(
            dt=None, T=None, target_mode="dtw",
            no_llc=not args.with_llc,
            use_clf=flags["use_clf"], use_l1=flags["use_l1"],
            matched=args.matched, unmatched=args.unmatched,
            matched_type=args.matched_type, unmatched_type=args.unmatched_type,
        )


        npz_name = f"{shape}_no_llc_midpulse_{name.replace('+','_')}.npz"
        npz_path = out_dir / npz_name

        t0 = time.time()
        logs = simulate(model, None, lambda: sel_factory(dt=float(1.0/(len(demo_t)-1))),
                        cfg, init_state, order=order,
                        direct_dist_fn=d_fn,
                        save_npz_path=str(npz_path),
                        t_grid=demo_t)
        elapsed = time.time() - t0

        cols.append((logs["z"], name))
        # DTW vs average training demo (your request)
        d = dtw_distance(logs["z"], demo_avg_pos)
        names.append(name); dtw_vals.append(d)
        last_logs = logs  # to capture t_norm and d_direct for the plot below

        log(f"Saved: {npz_path.name} | steps={len(logs['t'])} | DTW(avg demo)={d:.3f} | elapsed={elapsed:.2f}s",
            enabled=args.verbose)

    # ---- single figure: all controllers + disturbance subplot
    fig_title = f"{shape} – Tracking under Mid-Interval Pulse Disturbance"
    fig_path = out_dir / f"fig_{shape}_midpulse.png"
    plot_all_together_with_dist(
        rollouts=cols,
        demo=demo_avg_pos,
        field_fn=field_fn,
        field_bounds=bounds,
        subtitle=fig_title,
        outpath=fig_path,
        t_norm=last_logs.get("t_norm", None),
        d_direct=last_logs.get("d_direct", None),
    )
    log(f"Figure saved: {fig_path}", enabled=args.verbose)

    # ---- bar chart with numeric labels (DTW vs average demo)
    chart_path = out_dir / f"chart_{shape}_dtw_avg.png"
    bar_with_ci(names, np.array(dtw_vals), np.zeros_like(dtw_vals),
                ylabel="DTW(trajectory, avg demo)",
                title=f"{shape} – Distance to Average Training Demonstration",
                outpath=chart_path)
    log(f"Chart saved:  {chart_path}", enabled=args.verbose)

    # ---- meta
    meta = dict(
        model=str(model_path), order=order, width=width, depth=depth,
        train_yaml=args.train_yaml,
        shape=shape, nsamples=nsamples, ntrain=ntrain,
        demo_ref="NODE rollout from same init",
        dtw_ref="average of training demos",
        disturbance="mid-interval rectangular pulse (mag=2.0, width=0.30)",
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log("All done. See meta.json and .npz logs in the output folder.", enabled=args.verbose)


if __name__ == "__main__":
    main()
