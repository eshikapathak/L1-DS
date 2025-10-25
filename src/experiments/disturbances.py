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
def make_matched_fn(on=True, kind="sine", MAG=10.0, FREQ=0.5, **kw):
    if not on: return constant_vec((0.0, 0.0))
    if kind == "sine":  return lambda t: _sine_vec(t, MAG, FREQ, ax_gain=kw.get("ax_gain",(0.6,0.9)), phase=kw.get("phase",(0.0,0.0)))
    if kind == "pulse": return lambda t: _pulse_vec(t, MAG, FREQ, kw.get("duty",0.25), kw.get("phase_x",0.0), kw.get("phase_y",0.3), ax_gain=kw.get("ax_gain",(0.6,0.9)))
    if kind == "const": return constant_vec(kw.get("val",(0.2, -0.1)))
    raise ValueError(f"Unknown matched kind: {kind}")

def make_unmatched_fn(on=True, kind="sine", MAG=20.5, FREQ=0.8, **kw):
    if not on: return constant_vec((0.0, 0.0))
    if kind == "sine":  return lambda t: _sine_vec(t, MAG, FREQ, ax_gain=kw.get("ax_gain",(1.0,0.8)), phase=kw.get("phase",(0.1,0.2)))
    if kind == "pulse": return lambda t: _pulse_vec(t, 30, FREQ, kw.get("duty",0.35), kw.get("phase_x",0.1), kw.get("phase_y",0.2), ax_gain=kw.get("ax_gain",(1.0,1.0)))
    if kind == "const": return constant_vec(kw.get("val",(10, 10)))
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

def two_mid_pulses(
    center1: float = 0.35, width1: float = 0.15, mag1: float = 2.0, ax_gain1=(1.0, 0.8),
    center2: float = 0.70, width2: float = 0.15, mag2: float = 2.0, ax_gain2=(1.0, 0.8),
):
    """
    Two rectangular pulses on the normalized time grid [0,1].
    Each pulse k is active for t_norm in [center_k - width_k/2, center_k + width_k/2].
    Returns a callable f(t_norm) -> R^2.

    Args
    ----
    center1, width1, mag1, ax_gain1 : parameters for the first pulse
    center2, width2, mag2, ax_gain2 : parameters for the second pulse
    """
    t01 = float(center1 - width1/2.0); t11 = float(center1 + width1/2.0)
    t02 = float(center2 - width2/2.0); t12 = float(center2 + width2/2.0)
    t01, t11 = max(0.0, t01), min(1.0, t11)
    t02, t12 = max(0.0, t02), min(1.0, t12)

    def f(t_norm: float) -> np.ndarray:
        on1 = (t_norm >= t01) and (t_norm <= t11)
        on2 = (t_norm >= t02) and (t_norm <= t12)
        s1 = mag1 if on1 else 0.0
        s2 = mag2 if on2 else 0.0
        # superpose both pulses (vector-valued disturbance)
        return np.array(
            [ax_gain1[0]*s1 + ax_gain2[0]*s2,
             ax_gain1[1]*s1 + ax_gain2[1]*s2],
            dtype=float,
        )

    return f