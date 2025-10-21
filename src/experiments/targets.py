from __future__ import annotations
import numpy as np
import jax.numpy as jnp

def integrate_f_theta(model, y0, dt, steps):
    """Euler-integrate learned field: y' = fθ(y) (first-order)."""
    T = np.zeros((int(steps), 2), dtype=float)
    y = np.array(y0, dtype=float)
    for k in range(int(steps)):
        T[k] = y
        f = np.array(model.func(0.0, jnp.asarray(y), None))
        y = y + dt * f
    return T

class TargetLeastEffort:
    """Precompute T by integrating fθ; pick forward point minimizing QP effort."""
    def __init__(self, model, dt, t_span=18.0, lookahead_N=35, y0_seed=None, wrap=True, clf_qp_fn=None):
        from .robust_ctrl import clf_qp as _clf_qp
        self.model = model
        self.dt = float(dt)
        self.look = int(lookahead_N)
        self.wrap = wrap
        self.clf_qp = _clf_qp if clf_qp_fn is None else clf_qp_fn
        steps = max(2, int(t_span / dt))
        seed = np.zeros(2) if y0_seed is None else y0_seed
        self.T = integrate_f_theta(model, seed, dt, steps)
        self.K = self.T.shape[0]

    def _f(self, y):  # learned velocity at y
        return np.array(self.model.func(0.0, jnp.asarray(y), None))

    def _fw_idxs(self, m):
        idxs = np.arange(m, m + self.look)
        return np.mod(idxs, self.K) if self.wrap else idxs[idxs < self.K]

    def get(self, x):
        d = np.linalg.norm(self.T - x[None, :], axis=1)
        m = int(np.argmin(d)); idxs = self._fw_idxs(m); f_x = self._f(x)
        best_j = None; best_norm = np.inf
        for j in idxs:
            y = self.T[j]; f_ref = self._f(y)
            v_nom = self.clf_qp(x=x, x_ref=y, f_x=f_x, f_ref=f_ref)
            nrm = float(np.dot(v_nom, v_nom))
            if nrm < best_norm: best_norm, best_j = nrm, j
        y_star = self.T[best_j]; f_star = self._f(y_star)
        return y_star, f_star

def _dtw(P, Q):
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean as _euclid
        d, _ = fastdtw(P.tolist(), Q.tolist(), dist=_euclid)
        return float(d)
    except Exception:
        n, m = len(P), len(Q)
        D = np.full((n+1, m+1), np.inf, dtype=float); D[0,0]=0.0
        for i in range(1, n+1):
            for j in range(1, m+1):
                c = np.linalg.norm(P[i-1]-Q[j-1])
                D[i,j] = c + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        return float(D[n,m])

class TargetDTW:
    """Windowed DTW against a demo reference; keeps internal pointer k_prev."""
    def __init__(self, r_pos, r_vel, W=50, H=40, Hprime=None):
        self.r_pos = np.asarray(r_pos); self.r_vel = np.asarray(r_vel)
        self.N = len(r_pos); self.W = int(W); self.H = int(H)
        self.Hp = int(Hprime) if (Hprime is not None) else int(H)
        self.k_prev = 0

    def init_from(self, p0):
        d = np.linalg.norm(self.r_pos - p0[None, :], axis=1)
        self.k_prev = int(np.argmin(d))

    def get(self, X_hist):
        start = self.k_prev; stop = min(self.N-1, self.k_prev + self.W)
        best_k = start; best_cost = np.inf
        for k in range(start, stop+1):
            k0 = max(0, k - self.Hp + 1)
            Rk = self.r_pos[k0:k+1]
            cost = _dtw(X_hist, Rk)
            if cost < best_cost:
                best_cost, best_k = cost, k
        self.k_prev = best_k
        return self.r_pos[best_k], self.r_vel[best_k]
