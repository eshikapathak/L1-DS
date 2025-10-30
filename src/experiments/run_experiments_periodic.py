from __future__ import annotations

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
# now this works:
from models.neural_ode import NeuralODE


import argparse, json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx

from models.neural_ode import NeuralODE
from src.train.train_node_periodic import *
from src.experiments.targets import TargetDTW
from src.experiments.robust_ctrl import L1Adaptive
from src.experiments.disturbances import big_mid_pulse, two_mid_pulses
from src.experiments.simulator import SimConfig, simulate
from src.experiments.metrics_plots import dtw_distance, plot_all_together_with_dist, bar_with_ci

# ---------- helpers ----------
def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _fmt_lr_decimal(x) -> str:
    from decimal import Decimal, getcontext
    getcontext().prec = 24
    s = format(Decimal(str(x)), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s

def expected_model_path(cfg: dict) -> Path:
    shape = cfg.get("shape", "IShape")
    order = int(cfg.get("order", 1))
    width = int(cfg.get("width", 100)); depth = int(cfg.get("depth", 3))
    ntrain = int(cfg.get("ntrain", 4)); ns = int(cfg.get("nsamples", 1000))
    curr = int(cfg.get("curriculum_steps", 200))
    lr = _fmt_lr_decimal(cfg.get("base_lr", 5e-4))
    seed = int(cfg.get("seed", 1385))
    out_root = Path(cfg.get("out_root", "outputs_periodic"))
    run_id = f"order{order}_w{width}_d{depth}_ntr{ntrain}_ns{ns}_curr{curr}_lr{lr}_seed{seed}"
    run_dir = out_root / shape / run_id
    cand = run_dir / f"{shape}_NODE_w{width}_d{depth}_ntr{ntrain}_ns{ns}.eqx"
    if not cand.exists():
        raise FileNotFoundError(f"Model not found: {cand}")
    return cand

def rollout_node_reference(model, z0: np.ndarray, t_grid: np.ndarray):
    t = np.asarray(t_grid).reshape(-1)
    dt = float((t[-1] - t[0]) / max(1, len(t)-1))
    Z = np.zeros((len(t)-1, 2), float)
    Vh = np.zeros_like(Z)
    z = z0.astype(float).copy()
    for k in range(len(t)-1):
        v_hat = np.array(model.func(0.0, jnp.asarray(z), None))
        Vh[k] = v_hat
        z = z + dt * v_hat
        Z[k] = z
    return Z, Vh

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_yaml", type=str, default="configs_periodic/IShape/node_train.yaml")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs_periodic/experiments/IShape_suite")
    ap.add_argument("--verbose", action="store_true")
    # LLC + disturbances
    ap.add_argument("--with_llc", action="store_true", help="Use plant/LLC path.")
    ap.add_argument("--matched", action="store_true")
    ap.add_argument("--unmatched", action="store_true")
    ap.add_argument("--matched_type", type=str, default="sine", choices=["sine","pulse","const"])
    ap.add_argument("--unmatched_type", type=str, default="sine", choices=["sine","pulse","const"])
    # direct (no_llc) uses pulse by default
    args = ap.parse_args()

    # --- load model via YAML
    cfg_tr = load_yaml(args.train_yaml)
    # model_path = Path(args.model) if args.model else expected_model_path(cfg_tr)
    # model_path = "outputs_periodic/new_0.4/IShape/segcur_w128_d3_ntr3_ns10000_lr0.0005_seed1234_mixed/IShape_NODE_segcur.eqx"
    model_path = "outputs_periodic/0.3/IShape/segcur_w128_d3_ntr3_ns10000_lr0.0005_seed1234_mixed/IShape_NODE_segcur.eqx"
    order = 1 #int(cfg_tr.get("order", 1))
    width = 128 #int(cfg_tr.get("width", 100)); 
    depth = 3 #int(cfg_tr.get("depth", 3))
    nsamples = 10000 #int(cfg_tr.get("nsamples", 1000)); 
    ntrain = 1 #int(cfg_tr.get("ntrain", 4))
    shape = cfg_tr.get("shape", "IShape")

    key = jax.random.PRNGKey(0)
    template = NeuralODE(data_size=2, width_size=width, depth=depth, key=key)
    model = eqx.tree_deserialise_leaves(str(model_path), template)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # --- data
    iros = load_shape(shape)
    pos_rs, vel_rs, t_rs = resample(iros, nsamples=nsamples)
    # print(t_rs)
    train_idxs = list(range(min(ntrain, len(pos_rs))))
    demo_avg_pos = np.mean([pos_rs[i] for i in train_idxs], axis=0)
    demo_avg_vel = np.mean([vel_rs[i] for i in train_idxs], axis=0)
    demo_t = np.linspace(0.0, 1.0, demo_avg_pos.shape[0], dtype=float)
    # print(len(demo_t))
    init_state = np.hstack([demo_avg_pos[0], demo_avg_vel[0]])

    # --- NODE rollout reference (used by CLF/L1 selector)
    node_ref_pos, node_ref_vel = rollout_node_reference(model, demo_avg_pos[0], demo_t)
    # print(node_ref_pos.shape)

    # --- selector factories
    class _SelWrap:
        def __init__(self, dt):
            self.sel = TargetDTW(node_ref_pos, node_ref_vel, W=50, H=40)
            self.sel.init_from(node_ref_pos[0])
            self.l1 = L1Adaptive(Ts=dt, a=10.0, omega=12.0, x0=node_ref_pos[0])
        def get(self, hist): return self.sel.get(hist)

    def selector_with_l1(dt): return _SelWrap(dt)
    def selector_no_l1(dt):
        s = _SelWrap(dt); delattr(s, "l1"); return s

    # --- bounds
    xr = float(np.ptp(demo_avg_pos[:,0])); yr = float(np.ptp(demo_avg_pos[:,1]))
    pad = 0.15 * max(xr, yr) if max(xr, yr) > 0 else 0.1
    # pad = 1
    bounds = ((float(np.min(demo_avg_pos[:,0])-pad), float(np.max(demo_avg_pos[:,0])+pad)),
              (float(np.min(demo_avg_pos[:,1])-pad), float(np.max(demo_avg_pos[:,1])+pad)))

    # --- disturbances: direct for no_llc; matched/unmatched for with_llc
    mode_tag = "with_llc" if args.with_llc else "no_llc"
    if not args.with_llc:
        d_fn = two_mid_pulses(0.3,0.20,0.0,(1.0,1.0), 0.7,0.20,0.0,(1.0,-1.0))
        dist_title = "Direct: two pulses"
        base_tag = f"no_llc_direct_two_pulses"
    else:
        d_fn = None
        parts=[]
        if args.matched: parts.append(f"matched-{args.matched_type}")
        if args.unmatched: parts.append(f"unmatched-{args.unmatched_type}")
        dist_title = " + ".join(parts) if parts else "No disturbance"
        base_tag = f"with_llc_{'_'.join(parts) if parts else 'none'}"

    # --- vector field fn
    def field_fn(p_xy: np.ndarray):
        return np.array(model.func(0.0, jnp.asarray(p_xy), None))

    # --- simulate three controllers
    controllers = [
        ("NODE",         dict(use_clf=False, use_l1=False), selector_no_l1),
        ("NODE+CLF",     dict(use_clf=True,  use_l1=False), selector_no_l1),
        ("NODE+CLF+L1",  dict(use_clf=True,  use_l1=True ), selector_with_l1),
    ]

    cols=[]; names=[]; dtw_vals=[]; last_logs=None
    for name, flags, sel_factory in controllers:
        cfg = SimConfig(
            dt=None, T=None, target_mode="dtw",
            no_llc=(not args.with_llc),
            use_clf=flags["use_clf"], use_l1=flags["use_l1"],
            matched=args.matched, unmatched=args.unmatched,
            matched_type=args.matched_type, unmatched_type=args.unmatched_type
        )
        npz_name = f"{shape}_{base_tag}_{name.replace('+','_')}.npz"
        logs = simulate(
            model, None, lambda: sel_factory(dt=float(1.0/(len(demo_t)-1))),
            cfg, init_state, order=order, direct_dist_fn=d_fn,
            save_npz_path=str(out_dir / npz_name), t_grid=demo_t
        )
        cols.append((logs["z"], name))
        names.append(name); dtw_vals.append(dtw_distance(logs["z"], demo_avg_pos))
        last_logs = logs

    # --- figure + chart
    title = f"{shape} — {('LLC' if args.with_llc else 'No LLC')} — {dist_title}"
    fig_path = out_dir / f"fig_{shape}_{base_tag}.png"
    plot_all_together_with_dist(
        rollouts=cols,
        demo=None,
        field_fn=field_fn,
        field_bounds=bounds,
        subtitle=title,
        outpath=fig_path,
        t_norm=last_logs.get("t_norm"), d_direct=last_logs.get("d_direct"),
        ref_curves=[("NODE ref (from avg init)", node_ref_pos),
                    ("Avg training demo",        demo_avg_pos)]
    )
    chart_path = out_dir / f"chart_{shape}_{base_tag}_dtw_avg.png"
    bar_with_ci(names, np.array(dtw_vals), np.zeros_like(dtw_vals),
                ylabel="DTW(trajectory, avg demo)", title=title, outpath=chart_path)

    # --- meta
    with open(out_dir / "meta.json", "w") as f:
        json.dump(dict(
            model=str(model_path), shape=shape, nsamples=nsamples, ntrain=ntrain,
            with_llc=args.with_llc, matched=args.matched, unmatched=args.unmatched,
            matched_type=args.matched_type, unmatched_type=args.unmatched_type,
            figure=str(fig_path), chart=str(chart_path)
        ), f, indent=2)

    print(f"[OK] Wrote: {fig_path}\n[OK] Wrote: {chart_path}")

if __name__ == "__main__":
    main()
