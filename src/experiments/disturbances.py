from __future__ import annotations
import numpy as np

def constant_vec(val=(0.0, 0.0)):
    v = np.array(val, dtype=float)
    return lambda t: v.copy()

def _sine_vec(t, mag, freq, ax_gain=(1.0, 0.8), phase=(0.0, 0.0)):
    w = 2*np.pi*freq
    sx = ax_gain[0]*np.sin(w*t + 2*np.pi*freq*phase[0])
    sy = ax_gain[1]*np.cos(0.75*w*t + 2*np.pi*freq*phase[1])
    return mag*np.array([sx, sy])

def _pulse_scalar(t, period, duty, phase):
    tau = (t + phase) % period
    return 1.0 if tau < duty*period else 0.0

def _pulse_vec(t, mag, freq, duty, phase_x=0.0, phase_y=0.0, ax_gain=(1.0, 0.8)):
    period = 1.0 / max(freq, 1e-12)
    px = _pulse_scalar(t, period, duty, phase_x)
    py = _pulse_scalar(t, period, duty, phase_y)
    return mag*np.array([ax_gain[0]*px, ax_gain[1]*py])

# ---------- with_llc disturbances (enter through plant) ----------
def make_matched_fn(on=True, kind="sine", MAG=1.5, FREQ=0.5, **kw):
    if not on: return constant_vec((0.0, 0.0))
    if kind == "sine":  return lambda t: _sine_vec(t, MAG, FREQ, ax_gain=kw.get("ax_gain",(0.6,0.9)), phase=kw.get("phase",(0.0,0.0)))
    if kind == "pulse": return lambda t: _pulse_vec(t, MAG, FREQ, kw.get("duty",0.25), kw.get("phase_x",0.0), kw.get("phase_y",0.3), ax_gain=kw.get("ax_gain",(0.6,0.9)))
    if kind == "const": return constant_vec(kw.get("val",(0.2, -0.1)))
    raise ValueError(f"Unknown matched kind: {kind}")

def make_unmatched_fn(on=True, kind="sine", MAG=1.2, FREQ=0.4, **kw):
    if not on: return constant_vec((0.0, 0.0))
    if kind == "sine":  return lambda t: _sine_vec(t, MAG, FREQ, ax_gain=kw.get("ax_gain",(1.0,0.8)), phase=kw.get("phase",(0.1,0.2)))
    if kind == "pulse": return lambda t: _pulse_vec(t, MAG, FREQ, kw.get("duty",0.35), kw.get("phase_x",0.1), kw.get("phase_y",0.2), ax_gain=kw.get("ax_gain",(1.0,0.8)))
    if kind == "const": return constant_vec(kw.get("val",(0.15, 0.1)))
    raise ValueError(f"Unknown unmatched kind: {kind}")

# ---------- no_llc disturbance (added directly to \dot z) ----------
def make_direct_fn(on=True, kind="sine", MAG=0.8, FREQ=0.5, **kw):
    """Used when cfg.no_llc=True; this is the d(t) in ż = ... + d(t)."""
    if not on: return constant_vec((0.0, 0.0))
    if kind == "sine":  return lambda t: _sine_vec(t, MAG, FREQ, ax_gain=kw.get("ax_gain",(1.0,0.8)), phase=kw.get("phase",(0.0,0.3)))
    if kind == "pulse": return lambda t: _pulse_vec(t, MAG, FREQ, kw.get("duty",0.25), kw.get("phase_x",0.0), kw.get("phase_y",0.3), ax_gain=kw.get("ax_gain",(1.0,0.8)))
    if kind == "const": return constant_vec(kw.get("val",(0.2, 0.0)))
    raise ValueError(f"Unknown direct kind: {kind}")


def big_mid_pulse(center: float = 0.5, width: float = 0.30,
                  mag: float = 2.0, ax_gain=(1.0, 0.8)):
    """
    A strong, rectangular pulse active only in the middle portion of a normalized time grid [0,1].
    Returns a callable f(t_norm)->R^2 (with t_norm in [0,1]).
    """
    t0 = float(center - width / 2.0)
    t1 = float(center + width / 2.0)
    t0, t1 = max(0.0, t0), min(1.0, t1)

    def f(t_norm: float) -> np.ndarray:
        on = (t_norm >= t0) and (t_norm <= t1)
        scale = mag if on else 0.0
        return np.array([ax_gain[0] * scale, ax_gain[1] * scale], dtype=float)

    return f
