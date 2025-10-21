from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .plant_llc import ContinuousDoubleIntegrator2D, PDLowLevel
from .disturbances import make_matched_fn, make_unmatched_fn
from .robust_ctrl import clf_qp, L1Adaptive

@dataclass
class SimConfig:
    # dt/T inferred from t_grid when None
    dt: Optional[float] = None
    T: Optional[float] = None
    Kp: float = 25.0
    Kd: float = 10.0
    target_mode: str = "dtw"          # "dtw"|"least_effort"|"oracle"
    no_llc: bool = True               # LASA non-periodic uses no_llc by default
    matched: bool = False             # only for with_llc
    unmatched: bool = False           # only for with_llc
    matched_type: str = "sine"
    unmatched_type: str = "sine"
    use_clf: bool = True              # <-- NEW: enable/disable CLF
    use_l1: bool = False

def _infer_dt_T_from_grid(t_grid: np.ndarray) -> tuple[float, float, int]:
    t_grid = np.asarray(t_grid).reshape(-1)
    if len(t_grid) < 2:
        raise ValueError("t_grid must have at least 2 samples.")
    diffs = np.diff(t_grid)
    dt = float(np.mean(diffs))
    T = float(t_grid[-1] - t_grid[0])
    N = len(t_grid) - 1
    return dt, T, N

def simulate(model,
             f_oracle: Optional[Callable[[float], Tuple[np.ndarray,np.ndarray,np.ndarray]]],
             get_selector: Callable[[], object],
             cfg: SimConfig,
             init_state: np.ndarray,
             order: int = 1,
             direct_dist_fn: Optional[Callable[[float], np.ndarray]] = None,
             save_npz_path: Optional[str] = None,
             t_grid: Optional[np.ndarray] = None):
    """
    If cfg.dt/T are None and t_grid is provided, infer dt and T from t_grid.
    If cfg.no_llc=True: discrete kinematics with direct disturbance d(t_norm).
    Else: PD + continuous plant with matched/unmatched disturbances.

    Returns logs dict and (optionally) saves compressed .npz with raw arrays.
    """
    # ---- time base
    if (cfg.dt is None or cfg.T is None) and (t_grid is not None):
        dt_sim, T_sim, N = _infer_dt_T_from_grid(t_grid)
        t_arr = np.asarray(t_grid).reshape(-1)
        t0 = float(t_arr[0])
        # also keep a normalized time for disturbance shaping: map t_arr->[0,1]
        t_norm = (t_arr - t_arr[0]) / max(1e-12, (t_arr[-1] - t_arr[0]))
    else:
        if cfg.dt is None or cfg.T is None:
            raise ValueError("Either provide t_grid for auto time base, or set cfg.dt and cfg.T.")
        dt_sim = float(cfg.dt)
        T_sim = float(cfg.T)
        N = int(T_sim / dt_sim)
        t0 = 0.0
        t_arr = np.linspace(0.0, T_sim, N+1)
        t_norm = np.linspace(0.0, 1.0, N+1)

    # ---- components
    plant = ContinuousDoubleIntegrator2D(dt_sim)
    low   = PDLowLevel(cfg.Kp, cfg.Kd)
    sigma_fn = make_matched_fn(cfg.matched, cfg.matched_type)
    dp_fn    = make_unmatched_fn(cfg.unmatched, cfg.unmatched_type)

    s = init_state.astype(float)  # [px,py,vx,vy]

    Z=[]; V=[]; Zref=[]; Vref_true=[]; Vhat=[]; Vnom=[]; Va=[]; Vref_cmd=[]; Tvec=[]
    Ddir=[]        # <-- NEW: log direct disturbance (for plotting)
    hist=[]

    selector = get_selector()
    if getattr(cfg, "use_l1", False) and not hasattr(selector, "l1"):
        selector.l1 = L1Adaptive(Ts=dt_sim, a=10.0, omega=12.0, x0=s[:2])

    t = t0
    for k in range(N):
        z_true, v_true = s[:2], s[2:]
        hist.append(z_true.copy()); hist = hist[-40:]

        # reference
        if cfg.target_mode == "oracle" and f_oracle is not None:
            z_ref, v_ref_t, _ = f_oracle(t)
        else:
            z_ref, v_ref_t = selector.get(np.array(hist, dtype=float) if cfg.target_mode=="dtw" else z_true)

        # learned drift
        if order == 1:
            v_hat = np.array(model.func(0.0, jnp.asarray(z_true), None))
        else:
            v_hat = v_true.copy()

        v_nom = clf_qp(x=z_true, x_ref=z_ref, f_x=v_hat, f_ref=v_ref_t) if cfg.use_clf else np.zeros(2)
        va = selector.l1.update(x_true=z_true, f_ref=v_hat, v_nom=v_nom) if hasattr(selector, "l1") else np.zeros(2)
        v_cmd = v_hat + v_nom + va

        if cfg.no_llc:
            d = np.zeros(2) if direct_dist_fn is None else direct_dist_fn(float(t_norm[k]))
            v_next = v_cmd + d
            p_next = z_true + dt_sim * v_next
            s = np.hstack([p_next, v_next])
            t = float(t_arr[k+1])
            Ddir.append(d.copy())
        else:
            u = low(z_ref, v_cmd, z_true, v_true)
            t, s = plant.step(t, s, u_k=u, sigma_m_fn=sigma_fn, d_p_fn=dp_fn)
            Ddir.append(np.zeros(2))

        # logs
        Z.append(s[:2].copy()); V.append(s[2:].copy())
        Zref.append(np.asarray(z_ref)); Vref_true.append(np.asarray(v_ref_t))
        Vhat.append(v_hat); Vnom.append(v_nom); Va.append(va); Vref_cmd.append(v_cmd); Tvec.append(t)

    out = {
        "t": np.array(Tvec, float),
        "t_norm": t_norm[:-1].copy(),               # aligned with steps (length N)
        "z": np.array(Z, float),
        "v": np.array(V, float),
        "z_ref": np.array(Zref, float),
        "v_ref_true": np.array(Vref_true, float),
        "v_hat": np.array(Vhat, float),
        "v_nom": np.array(Vnom, float),
        "v_a": np.array(Va, float),
        "v_ref_cmd": np.array(Vref_cmd, float),
        "d_direct": np.array(Ddir, float),          # <-- NEW: direct disturbance (N,2)
        "config": {**cfg.__dict__, "dt_used": dt_sim, "T_used": T_sim, "N": N},
    }
    if save_npz_path:
        np.savez_compressed(save_npz_path, **out)
    return out
