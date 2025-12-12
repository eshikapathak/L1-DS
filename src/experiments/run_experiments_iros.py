# src/experiments/run_experiments_iros.py
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx

# Project modules
from models.neural_ode import NeuralODE
from src.train.train_node_iros import load_shape, resample
from src.experiments.targets import TargetDTW, TargetLeastEffort
from src.experiments.robust_ctrl import L1Adaptive
from src.experiments.disturbances_iros import big_mid_pulse, two_mid_pulses
from src.experiments.simulator_iros import SimConfig, simulate
from src.experiments.metrics_plots import dtw_distance, plot_all_together_with_dist, bar_with_ci


def rollout_node_reference(model: NeuralODE, z0: np.ndarray, t_grid: np.ndarray):
    """Discrete Euler roll (same as LASA helper) to build a NODE reference."""
    t = np.asarray(t_grid).reshape(-1)
    dt = float((t[-1] - t[0]) / max(1, len(t) - 1))
    Z = np.zeros((len(t) - 1, 2), float)
    Vh = np.zeros_like(Z)
    z = z0.astype(float).copy()
    for k in range(len(t) - 1):
        v_hat = np.array(model.func(0.0, jnp.asarray(z), None))
        Vh[k] = v_hat
        z = z + dt * v_hat
        Z[k] = z
    return Z, Vh


def parse_args():
    ap = argparse.ArgumentParser("Run periodic experiments (NODE / NODE+CLF / NODE+CLF+L1).")
    ap.add_argument("--shape", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, help="Path to trained .eqx model")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--nsamples", type=int, default=10000)
    ap.add_argument("--ntrain", type=int, default=3)
    ap.add_argument("--data_root", type=str, default="IROS_dataset")

    # Target selection policy (NEW: parity with LASA script)
    ap.add_argument("--selector", type=str, default="dtw",
                    choices=["dtw", "least_effort"],
                    help="Target selection policy. Default=dtw.")

    # Plant/LLC path & disturbance types
    ap.add_argument("--with_llc", action="store_true", help="Use plant/LLC path.")
    ap.add_argument("--matched", action="store_true")
    ap.add_argument("--unmatched", action="store_true")
    ap.add_argument("--matched_type", type=str, default="sine",
                    choices=["sine","pulse","const","multisine","chirp"])
    ap.add_argument("--unmatched_type", type=str, default="sine",
                    choices=["sine","pulse","const","multisine","chirp"])

    # Direct (no LLC):
    ap.add_argument("--no_llc_zero", action="store_true",
                    help="No-LLC mode with zero disturbance (overrides default pulses).")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Load model (width/depth template just sets shapes; leaves are deserialized)
    key = jax.random.PRNGKey(0)
    template = NeuralODE(data_size=2, width_size=128, depth=3, key=key)
    model = eqx.tree_deserialise_leaves(str(args.model), template)

    # Data: average training demo (pos/vel/t), consistent with training resample
    iros = load_shape(args.shape, root=args.data_root)
    pos_rs, vel_rs, _ = resample(iros, nsamples=args.nsamples)
    train_idxs = list(range(min(args.ntrain, len(pos_rs))))
    demo_avg_pos = np.mean([pos_rs[i] for i in train_idxs], axis=0)
    demo_avg_vel = np.mean([vel_rs[i] for i in train_idxs], axis=0)
    demo_t = np.linspace(0.0, 1.0, demo_avg_pos.shape[0], dtype=float)
    init_state = np.hstack([demo_avg_pos[0], demo_avg_vel[0]])

    # NODE rollout reference (for DTW selector & CLF/L1 context as desired)
    node_ref_pos, node_ref_vel = rollout_node_reference(model, demo_avg_pos[0], demo_t)

    # Selector factories (DTW default; Least-Effort optional) — mirrors LASA
    selector_kind = args.selector.lower()  # "dtw" or "least_effort"

    class _SelWrap:
        def __init__(self, dt: float):
            self.dt = float(dt)
            if selector_kind == "least_effort":
                # Least-Effort uses the learned model directly as the task dynamics
                self.sel = TargetLeastEffort(
                    model=model,
                    dt=self.dt,
                    t_span=float(demo_t[-1] - demo_t[0]),  # normalized span (=1.0)
                    lookahead_N=35,
                    y0_seed=node_ref_pos[0],
                    wrap=False,
                )
            else:
                # DTW on the NODE reference (robust to time warping)
                self.sel = TargetDTW(node_ref_pos, node_ref_vel, W=50, H=40)
                self.sel.init_from(node_ref_pos[0])

            # L1 adaptive outer loop at the same sampling
            self.l1 = L1Adaptive(Ts=self.dt, a=10.0, omega=12.0, x0=node_ref_pos[0])

        def get(self, hist_or_state):
            if selector_kind == "least_effort":
                # simulate() may pass history; consume current point
                if isinstance(hist_or_state, (list, tuple)):
                    x = np.asarray(hist_or_state[-1], dtype=float)
                else:
                    x = np.asarray(hist_or_state, dtype=float)
                return self.sel.get(x)
            else:
                return self.sel.get(hist_or_state)

    def selector_with_l1(dt: float): return _SelWrap(dt)
    def selector_no_l1(dt: float):
        s = _SelWrap(dt)
        if hasattr(s, "l1"):
            delattr(s, "l1")
        return s

    # Plot bounds around avg demo
    xr = float(np.ptp(demo_avg_pos[:, 0])); yr = float(np.ptp(demo_avg_pos[:, 1]))
    pad = 0.15 * max(xr, yr) if max(xr, yr) > 0 else 0.1
    bounds = (
        (float(np.min(demo_avg_pos[:, 0]) - pad), float(np.max(demo_avg_pos[:, 0]) + pad)),
        (float(np.min(demo_avg_pos[:, 1]) - pad), float(np.max(demo_avg_pos[:, 1]) + pad)),
    )

    # Disturbances
    if not args.with_llc:
        if args.no_llc_zero:
            def d_zero(_t): return np.array([0.0, 0.0], dtype=float)
            d_fn = lambda t: d_zero(t)
            dist_title = "Direct: zero disturbance"
            base_tag = "no_llc_zero"
        else:
            d_fn = two_mid_pulses(0.3, 0.20, 4.0, (1.0, 1.0), 0.7, 0.20, 2.0, (1.0, -1.0))
            dist_title = "Direct: two pulses"
            base_tag = "no_llc_direct_two_pulses"
    else:
        d_fn = None
        parts = []
        if args.matched: parts.append(f"matched-{args.matched_type}")
        if args.unmatched: parts.append(f"unmatched-{args.unmatched_type}")
        dist_title = " + ".join(parts) if parts else "No disturbance"
        base_tag = f"with_llc_{'_'.join(parts) if parts else 'none'}"

    # NODE field for background streamplot
    def field_fn(p_xy: np.ndarray):
        return np.array(model.func(0.0, jnp.asarray(p_xy), None))

    # Controllers to run
    controllers = [
        ("NODE",         dict(use_clf=False, use_l1=False), selector_no_l1),
        ("NODE+CLF",     dict(use_clf=True,  use_l1=False), selector_no_l1),
        ("NODE+CLF+L1",  dict(use_clf=True,  use_l1=True ), selector_with_l1),
    ]

    # Run
    cols = []; names = []; dtw_vals = []; last_logs = None
    sel_tag = "sel-dtw" if selector_kind == "dtw" else "sel-le"
    for name, flags, sel_factory in controllers:
        print("Running", name)
        cfg = SimConfig(
            dt=None, T=None,
            target_mode=("dtw" if selector_kind == "dtw" else "least_effort"),
            no_llc=(not args.with_llc),
            use_clf=flags["use_clf"], use_l1=flags["use_l1"],
            matched=args.matched, unmatched=args.unmatched,
            matched_type=args.matched_type, unmatched_type=args.unmatched_type,
        )
        npz_name = f"{args.shape}_{base_tag}_{sel_tag}_{name.replace('+','_')}.npz"
        logs = simulate(
            model, None, lambda: sel_factory(dt=float(1.0/(len(demo_t)-1))),
            cfg, init_state, order=1, direct_dist_fn=d_fn,
            save_npz_path=str(out_dir / npz_name), t_grid=demo_t
        )
        cols.append((logs["z"], name))
        names.append(name)
        dtw_vals.append(dtw_distance(logs["z"], demo_avg_pos))
        last_logs = logs

    title = f"{args.shape} — {('LLC' if args.with_llc else 'No LLC')} — {dist_title} ({'DTW' if selector_kind=='dtw' else 'Least-Effort'})"
    fig_path = out_dir / f"fig_{args.shape}_{base_tag}_{sel_tag}.png"
    plot_all_together_with_dist(
        rollouts=cols, demo=None, field_fn=field_fn, field_bounds=bounds,
        subtitle=title, outpath=fig_path,
        t_norm=last_logs.get("t_norm"), d_direct=last_logs.get("d_direct"),
        ref_curves=[("NODE ref (from avg init)", node_ref_pos),
                    ("Avg training demo",        demo_avg_pos)]
    )

    chart_path = out_dir / f"chart_{args.shape}_{base_tag}_{sel_tag}_dtw_avg.png"
    bar_with_ci(names, np.array(dtw_vals), np.zeros_like(dtw_vals),
                ylabel="DTW(trajectory, avg demo)", title=title, outpath=chart_path)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(dict(
            model=str(args.model), shape=args.shape, nsamples=args.nsamples, ntrain=args.ntrain,
            selector=selector_kind,
            with_llc=args.with_llc, matched=args.matched, unmatched=args.unmatched,
            matched_type=args.matched_type, unmatched_type=args.unmatched_type,
            figure=str(fig_path), chart=str(chart_path)
        ), f, indent=2)

    print(f"[OK] Wrote: {fig_path}\n[OK] Wrote: {chart_path}")


if __name__ == "__main__":
    main()
