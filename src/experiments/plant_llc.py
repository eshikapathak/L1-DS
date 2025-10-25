from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


class ContinuousDoubleIntegrator2D:
    """
    s = [px, py, vx, vy]
    ṗ = v + d_p(t)         # unmatched disturbance
    v̇ = u + σ_m(t)         # matched disturbance
    """
    def __init__(self, dt: float, rtol=1e-7, atol=1e-9):
        self.dt = float(dt)
        self.rtol = rtol
        self.atol = atol

    def _rhs(self, tau, s, u_k, sigma_m_fn, d_p_fn):
        p = s[:2]; v = s[2:]
        dp = v + d_p_fn(tau)
        dv = u_k + sigma_m_fn(tau)
        return np.hstack([dp, dv])

    def step(self, t, s, u_k, sigma_m_fn, d_p_fn):
        sol = solve_ivp(
            lambda tau, y: self._rhs(tau, y, u_k, sigma_m_fn, d_p_fn),
            (t, t + self.dt), s, method="RK45", rtol=self.rtol, atol=self.atol
        )
        return sol.t[-1], sol.y[:, -1]


class PDLowLevel:
    """ u = Kp(p_ref - p) + Kd(v_ref - v) """
    def __init__(self, Kp: float, Kd: float):
        self.Kp, self.Kd = Kp, Kd
    def __call__(self, p_ref, v_ref, p, v):
        return self.Kp*(p_ref - p) + self.Kd*(v_ref - v)

class PIDLowLevel:
    """
    u = Kp (p_ref - p) + Kd (v_ref - v) + Ki ∫ (p_ref - p) dt
    - Integral on position error (good for bias/unmatched disturbances).
    - Simple anti-windup via integral clamping and optional output clamp.
    """
    def __init__(self, Kp: float, Kd: float, Ki: float = 0.0,
                 i_clamp: float = np.inf, u_limit: float | None = None):
        self.Kp, self.Kd, self.Ki = float(Kp), float(Kd), float(Ki)
        self.i_clamp = float(i_clamp)
        self.u_limit = None if u_limit is None else float(u_limit)
        self.i = np.zeros(2, dtype=float)  # integral state (x,y)

    def reset(self):
        self.i[:] = 0.0

    def __call__(self, p_ref, v_ref, p, v, dt: float):
        e_p = np.asarray(p_ref, float) - np.asarray(p, float)
        e_v = np.asarray(v_ref, float) - np.asarray(v, float)

        # integrate position error
        self.i += e_p * float(dt)

        # anti-windup (component-wise clamp)
        if np.isfinite(self.i_clamp):
            self.i = np.clip(self.i, -self.i_clamp, +self.i_clamp)

        u = self.Kp * e_p + self.Kd * e_v + self.Ki * self.i

        # optional actuator limit
        if self.u_limit is not None:
            u = np.clip(u, -self.u_limit, +self.u_limit)

        return u