# models/neural_ode2.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
import diffrax


class AccelMLP(eqx.Module):
    """Maps (x, y, vx, vy) -> (ax, ay)."""
    mlp: eqx.nn.MLP

    def __init__(self, width_size: int, depth: int, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=4,
            out_size=2,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )
        # Orthogonal init on weights; keep default bias init
        init = jnn.initializers.orthogonal()
        subkeys = jr.split(key, depth + 1)
        for i in range(depth + 1):
            w_loc = lambda m: m.layers[i].weight
            w_shape = self.mlp.layers[i].weight.shape
            self.mlp = eqx.tree_at(w_loc, self.mlp, replace=init(subkeys[i], w_shape))

    @eqx.filter_jit
    def __call__(self, y):
        # y: (..., 4) = [x,y,vx,vy] -> (ax, ay)
        return self.mlp(y)


class NeuralODE2nd(eqx.Module):
    """
    Second-order NODE:
      state y = [x, y, vx, vy]
      rhs   f(y) = [vx, vy, ax, ay], where (ax,ay) = AccelMLP([x,y,vx,vy])
    """
    accel: AccelMLP

    def __init__(self, width_size: int, depth: int, *, key):
        self.accel = AccelMLP(width_size=width_size, depth=depth, key=key)

    @eqx.filter_jit
    def rhs(self, t, y, args=None):
        v = y[..., 2:]              # (vx, vy)
        a = self.accel(y)           # (ax, ay)
        return jnp.concatenate([v, a], axis=-1)

    def rollout(self, ts, y0):
        term = diffrax.ODETerm(lambda t, y, args: self.rhs(t, y, args))
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys  # (T, 4) = [x,y,vx,vy]

    def __call__(self, ts, y0):
        return self.rollout(ts, y0)
