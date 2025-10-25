from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx

from models.neural_ode import NeuralODE
from src.data.lasa import load_shape, resample
from .targets import TargetDTW, TargetLeastEffort
from .robust_ctrl import L1Adaptive
from .disturbances import big_mid_pulse, two_mid_pulses   # direct disturbance (no_llc only)
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


# ------------------------ label builders ------------------------
def _condition_tags(args) -> dict:
    """Return tags/strings for filenames and titles based on LLC + disturbance flags."""
    with_llc = bool(args.with_llc)
    mode_tag = "with_llc" if with_llc else "no_llc"
    mode_title = "LLC" if with_llc else "No LLC"

    if with_llc:
        if args.matched and args.unmatched:
            dist_mode = "matched+unmatched"
            dist_tag = f"matched-{args.matched_type}_unmatched-{args.unmatched_type}"
            dist_title = f"Matched ({args.matched_type}) + Unmatched ({args.unmatched_type})"
        elif args.matched:
            dist_mode = "matched"
            dist_tag = f"matched-{args.matched_type}"
            dist_title = f"Matched ({args.matched_type})"
        elif args.unmatched:
            dist_mode = "unmatched"
            dist_tag = f"unmatched-{args.unmatched_type}"
            dist_title = f"Unmatched ({args.unmatched_type})"
        else:
            dist_mode = "none"
            dist_tag = "none"
            dist_title = "No disturbance"
    else:
        # direct disturbance path (we currently use mid-pulse)
        dist_mode = "direct"
        dist_tag = "direct-midpulse"
        dist_title = "Direct disturbance (mid-pulse)"

    base_tag = f"{mode_tag}_{dist_tag}"
    title = f"{mode_title} — Disturbance: {dist_title}"
    return dict(
        mode_tag=mode_tag, mode_title=mode_title,
        dist_mode=dist_mode, dist_tag=dist_tag, dist_title=dist_title,
        base_tag=base_tag,
        fig_title=title
    )


# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_yaml", type=str, default="configs/Worm/node_train.yaml")
    ap.add_argument("--model", type=str, default=None, help="Optional: explicit .eqx path")
    ap.add_argument("--out", type=str, default="outputs/experiments/Worm_suite")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
    "--selector",
    type=str,
    default="dtw",
    choices=["dtw", "least_effort"],
    help="Target selection policy. Default=dtw."
    )

    # plant/LLC flags
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
    pos_rs, vel_rs, _ = resample(lasa, nsamples=nsamples)

    # --- training subset & average demo
    train_idxs = list(range(1, ntrain)) if ntrain > 1 else [0]
    demo_avg_pos = np.mean([pos_rs[i] for i in train_idxs], axis=0)   # (N,2)
    demo_avg_vel = np.mean([vel_rs[i] for i in train_idxs], axis=0)   # (N,2)
    demo_t = np.linspace(0.0, 1.0, demo_avg_pos.shape[0], dtype=float)

    # --- initial condition from the average demo
    init_state = np.hstack([demo_avg_pos[0], demo_avg_vel[0]])

    # --- NODE rollout reference for CLF/L1
    node_ref_pos, node_ref_vel = rollout_node_reference(model, demo_avg_pos[0], demo_t)

        # --- SAVE references
    avg_demo_path = out_dir / f"{shape}_avg_demo.npz"
    np.savez_compressed(avg_demo_path, pos=demo_avg_pos, vel=demo_avg_vel, t=demo_t)

    node_ref_path = out_dir / f"{shape}_node_ref_from_avg_init.npz"
    np.savez_compressed(node_ref_path, pos=node_ref_pos, vel=node_ref_vel, t=demo_t)

    log(f"Saved references: {avg_demo_path.name}, {node_ref_path.name}", enabled=args.verbose)

    # ---- selector factories (track NODE rollout)
    # class _SelWrap:
    #     def __init__(self, dt):
    #         self.sel = TargetDTW(node_ref_pos, node_ref_vel, W=50, H=40)
    #         self.sel.init_from(node_ref_pos[0])
    #         self.l1 = L1Adaptive(Ts=dt, a=10.0, omega=20.0, x0=node_ref_pos[0])
    #     def get(self, hist): return self.sel.get(hist)

    # def selector_with_l1(dt): return _SelWrap(dt)
    # def selector_no_l1(dt):
    #     s = _SelWrap(dt); delattr(s, "l1"); return s
    def make_selector_factory(args, node_ref_pos, node_ref_vel):
        """Return two factories (with_l1, no_l1) that create selector objects per run."""
        use_dtw = (args.selector == "dtw")

        class _SelWrap:
            def __init__(self, dt):
                if use_dtw:
                    self.sel = TargetDTW(node_ref_pos, node_ref_vel, W=50, H=40)
                    self.sel.init_from(node_ref_pos[0])
                else:
                    # least-effort: no history needed; uses current state in simulate()
                    self.sel = TargetLeastEffort(node_ref_pos, node_ref_vel, dt=dt)
                self.l1 = L1Adaptive(Ts=dt, a=10.0, omega=20.0, x0=node_ref_pos[0])

            def get(self, inp):
                # simulate() passes: hist if target_mode=="dtw", else current z_true
                return self.sel.get(inp)

        def selector_with_l1(dt): return _SelWrap(dt)
        def selector_no_l1(dt):
            s = _SelWrap(dt); delattr(s, "l1"); return s

        return selector_with_l1, selector_no_l1

    selector_with_l1, selector_no_l1 = make_selector_factory(args, node_ref_pos, node_ref_vel)

    # ---- vector field function
    def field_fn(p_xy: np.ndarray):
        return np.array(model.func(0.0, jnp.asarray(p_xy), None))

    # ---- plot bounds around average demo
    x_range = float(np.ptp(demo_avg_pos[:, 0])); y_range = float(np.ptp(demo_avg_pos[:, 1]))
    pad = 0.15 * max(x_range, y_range) if max(x_range, y_range) > 0 else 0.1
    bounds = (
        (float(np.min(demo_avg_pos[:, 0])) - pad, float(np.max(demo_avg_pos[:, 0])) + pad),
        (float(np.min(demo_avg_pos[:, 1])) - pad, float(np.max(demo_avg_pos[:, 1])) + pad),
    )

    # ---- condition tags & titles
    tags = _condition_tags(args)
    sel_tag = "sel-dtw" if args.selector == "dtw" else "sel-le"
    log(f"Condition: {tags['fig_title']}", enabled=args.verbose)

    # ---- direct disturbance for no_llc (ignored when with_llc)
    d_fn = None
    if tags["mode_tag"] == "no_llc":
        # You can parameterize these if you add CLI knobs later.
        # d_fn = big_mid_pulse(center=0.6, width=0.8, mag=45.0, ax_gain=(1.0, 1.0))
        d_fn = two_mid_pulses(
            center1=0.30, width1=0.20, mag1=40.0, ax_gain1=(1.0, -1.0),
            center2=0.8, width2=0.50, mag2=50.0, ax_gain2=(1.0, 1.0),
        )

    # ---- simulate all controllers
    controllers = [
        ("NODE",         dict(use_clf=False, use_l1=False), selector_no_l1),
        ("NODE+CLF",     dict(use_clf=True,  use_l1=False), selector_no_l1),
        ("NODE+CLF+L1",  dict(use_clf=True,  use_l1=True ), selector_with_l1),
    ]

    cols = []
    names, dtw_vals = [], []
    last_logs = None

    for i, (name, flags, sel_factory) in enumerate(controllers, start=1):
        log(f"[{i}/{len(controllers)}] Running controller: {name}", enabled=args.verbose)

        target_mode = "dtw" if args.selector == "dtw" else "least_effort"

        cfg = SimConfig(
            dt=None, T=None, target_mode=target_mode,
            no_llc=(tags["mode_tag"] == "no_llc"),
            use_clf=flags["use_clf"], use_l1=flags["use_l1"],
            matched=args.matched, unmatched=args.unmatched,
            matched_type=args.matched_type, unmatched_type=args.unmatched_type,
        )

        # base = f"{shape}_{tags['base_tag']}"
        base = f"{shape}_{tags['base_tag']}_{sel_tag}"
        npz_name = f"{base}_{name.replace('+','_')}.npz"
        npz_path = out_dir / npz_name

        t0 = time.time()
        logs = simulate(
            model, None, lambda: sel_factory(dt=float(1.0/(len(demo_t)-1))),
            cfg, init_state, order=order,
            direct_dist_fn=d_fn,
            save_npz_path=str(npz_path),
            t_grid=demo_t
        )
        elapsed = time.time() - t0

        cols.append((logs["z"], name))
        d = dtw_distance(logs["z"], demo_avg_pos)  # DTW vs average training demo
        names.append(name); dtw_vals.append(d)
        last_logs = logs

        log(f"Saved: {npz_path.name} | steps={len(logs['t'])} | DTW(avg demo)={d:.3f} | elapsed={elapsed:.2f}s",
            enabled=args.verbose)

    # ---- figure (overlay + disturbance subplot)
    # fig_title = f"{shape} — {tags['fig_title']}"
    fig_title = f"{shape} — {tags['fig_title']} ({'DTW' if args.selector=='dtw' else 'Least-Effort'} selector)"
    fig_path = out_dir / f"fig_{shape}_{tags['base_tag']}.png"

    ref_curves = [
    ("NODE ref (from avg init)", node_ref_pos),
    ("Avg training demo",        demo_avg_pos),
    ]

    # plot_all_together_with_dist(
    #     rollouts=cols,
    #     demo=demo_avg_pos,
    #     field_fn=field_fn,
    #     field_bounds=bounds,
    #     subtitle=fig_title,
    #     outpath=fig_path,
    #     t_norm=last_logs.get("t_norm", None),
    #     d_direct=last_logs.get("d_direct", None),
    # )
    plot_all_together_with_dist(
    rollouts=cols,
    demo=None,
    field_fn=field_fn,
    field_bounds=bounds,
    subtitle=fig_title,
    outpath=fig_path,
    t_norm=last_logs.get("t_norm", None),
    d_direct=last_logs.get("d_direct", None),
    ref_curves=ref_curves,
    d_matched=last_logs.get("sigma", None),     # <-- NEW
    d_unmatched=last_logs.get("d_p", None),     # <-- NEW
    )


    log(f"Figure saved: {fig_path}", enabled=args.verbose)

    # ---- DTW chart (numeric labels)
    chart_path = out_dir / f"chart_{shape}_{tags['base_tag']}_dtw_avg.png"
    bar_with_ci(
        names, np.array(dtw_vals), np.zeros_like(dtw_vals),
        ylabel="DTW(trajectory, avg demo)",
        title=f"{shape} — {tags['fig_title']}",
        outpath=chart_path
    )
    log(f"Chart saved:  {chart_path}", enabled=args.verbose)

    # ---- meta
    meta = dict(
        model=str(model_path), order=order, width=width, depth=depth,
        train_yaml=args.train_yaml,
        shape=shape, nsamples=nsamples, ntrain=ntrain,
        demo_ref="NODE rollout from average-demo init",
        dtw_ref="average of training demos",
        condition=dict(
            with_llc=(tags["mode_tag"]=="with_llc"),
            disturbance_mode=tags["dist_mode"],
            matched=args.matched, unmatched=args.unmatched,
            matched_type=args.matched_type, unmatched_type=args.unmatched_type,
        ),
        outputs=dict(
            fig=str(fig_path), chart=str(chart_path),
            logs=[f"{shape}_{tags['base_tag']}_{n.replace('+','_')}.npz" for n,_,_ in controllers]
        )
    )

    meta["references"] = dict(
    avg_demo=str(avg_demo_path),
    node_ref=str(node_ref_path),
    )

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log("Done.", enabled=args.verbose)


if __name__ == "__main__":
    main()
