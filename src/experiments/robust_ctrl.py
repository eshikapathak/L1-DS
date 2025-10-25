from __future__ import annotations
import numpy as np
from scipy.linalg import expm

def clf_qp(x, x_ref, f_x, f_ref, alpha_h=15.0, lambda_v=10.0):
    """
    Min ||v||^2 + λ·relu(Gv−h)  s.t.  Gv ≤ h
    with G=2(x−x_ref)^T,  h = −2(x−x_ref)^T(f_x−f_ref) − α||x−x_ref||^2
    """
    import cvxpy as cp
    n = 2
    G = 2*(x - x_ref).T
    h = -2*(x - x_ref).T @ (f_x - f_ref) - alpha_h*np.dot(x-x_ref, x-x_ref)
    v = cp.Variable(n)
    cost = cp.quad_form(v, np.eye(n)) + lambda_v * cp.pos(G @ v - h)
    prob = cp.Problem(cp.Minimize(cost), [G @ v <= h])
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    except Exception:
        return np.zeros(n, dtype=float)
    vv = np.asarray(v.value, dtype=float).reshape(n) if (v.value is not None) else np.zeros(n)
    return vv if np.all(np.isfinite(vv)) else np.zeros(n)


class L1Adaptive:
    """Discrete-time L1 compensator working on v-domain (velocity)."""
    def __init__(self, Ts, a=10.0, omega=12.0, x0=None):
        As = -a*np.eye(2); Kf = omega*np.eye(2)
        self.As, self.Ts, self.Kf = As, Ts, Kf
        self.expAsTs = expm(As*Ts)
        self.Phi_inv = np.linalg.inv(np.linalg.inv(As) @ (self.expAsTs - np.eye(2)))
        self.x_hat = np.zeros(2) if x0 is None else np.asarray(x0, dtype=float).copy()
        self.q = np.zeros(2); self.sigma_hat = np.zeros(2)

    def update(self, x_true, f_ref, v_nom):
        x_true = np.asarray(x_true, dtype=float).reshape(2)
        f_ref  = np.asarray(f_ref,  dtype=float).reshape(2)
        v_nom  = np.zeros(2) if (v_nom is None) else np.asarray(v_nom, dtype=float).reshape(2)
        xt = self.x_hat - x_true
        mu = self.expAsTs @ xt
        self.q        += (-self.Kf @ self.q + self.Kf @ self.sigma_hat) * self.Ts
        self.sigma_hat = -(self.Phi_inv @ mu)
        v_a            = -self.q
        self.x_hat    += (f_ref + v_nom + v_a + self.sigma_hat +
                          self.As @ (self.x_hat - x_true)) * self.Ts
        return v_a
