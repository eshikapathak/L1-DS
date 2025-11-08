# disturbances_periodic.py
from __future__ import annotations
import numpy as np

# -------------------------------
# Basic building blocks
# -------------------------------
def constant_vec(val=(0.0, 0.0)):
    v = np.array(val, dtype=float)
    return lambda t: v.copy()

def _sine_vec(t, mag, freq, ax_gain=(1.0, 0.8), phase=(0.0, 0.0)):
    w = 2*np.pi*freq
    sx = ax_gain[0]*np.sin(w*t + 2*np.pi*phase[0])
    sy = ax_gain[1]*np.cos(0.75*w*t + 2*np.pi*phase[1])
    return mag*np.array([sx, sy], dtype=float)

def _pulse_scalar(t, period, duty, phase):
    tau = (t + phase) % period
    return 1.0 if tau < duty*period else 0.0

def _pulse_vec(t, mag, freq, duty, phase_x=0.0, phase_y=0.0, ax_gain=(1.0, 0.8)):
    period = 1.0 / max(freq, 1e-12)
    px = _pulse_scalar(t, period, duty, phase_x)
    py = _pulse_scalar(t, period, duty, phase_y)
    return mag*np.array([ax_gain[0]*px, ax_gain[1]*py], dtype=float)

def _square_vec(t, mag, freq, ax_gain=(1.0, 1.0), phase=(0.0, 0.0)):
    # sign(sin) square wave; phase in cycles
    w = 2*np.pi*freq
    sx = ax_gain[0]*np.sign(np.sin(w*t + 2*np.pi*phase[0]))
    sy = ax_gain[1]*np.sign(np.sin(0.8*w*t + 2*np.pi*phase[1]))
    return mag*np.array([sx, sy], dtype=float)

def _saw_vec(t, mag, freq, ax_gain=(1.0, 1.0), phase=(0.0, 0.0)):
    # sawtooth via fractional part
    # saw(t) in [-1,1]: s = 2*(frac(ft+phi) - 0.5)
    def saw(f, ph):
        frac = (f*t + ph) - np.floor(f*t + ph)
        return 2.0*(frac - 0.5)
    sx = ax_gain[0]*saw(freq, phase[0])
    sy = ax_gain[1]*saw(0.6*freq, phase[1])
    return mag*np.array([sx, sy], dtype=float)

def _multisine_vec_factory(mag=10.0, K=7, fmin=0.2, fmax=2.0,
                           ax_gain=(1.0, 1.0), seed=123, phases=None):
    """
    Deterministic sum of K sinusoids with random freqs & phases in [fmin,fmax].
    Different mixes on x/y to avoid trivial coupling.
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(fmin, fmax, size=K)
    # if phases provided, broadcast; else random in [0,1) cycles
    if phases is None:
        phx = rng.uniform(0.0, 1.0, size=K)
        phy = rng.uniform(0.0, 1.0, size=K)
    else:
        phx = np.asarray(phases[0]).reshape(-1)
        phy = np.asarray(phases[1]).reshape(-1)
        if len(phx) < K: phx = np.pad(phx, (0, K-len(phx)), mode="wrap")
        if len(phy) < K: phy = np.pad(phy, (0, K-len(phy)), mode="wrap")

    # slight detuning for y to avoid perfect overlap
    detune = rng.uniform(0.85, 1.15, size=K)

    def f(t: float):
        w = 2*np.pi*freqs
        sx = np.sum(np.sin(w*t + 2*np.pi*phx))
        sy = np.sum(np.sin((w*detune)*t + 2*np.pi*phy))
        return mag*np.array([ax_gain[0]*sx, ax_gain[1]*sy], dtype=float) / max(K, 1)
    return f


def _chirp_vec_factory(mag=10.0, f0=0.2, f1=2.0, T=8.0,
                       ax_gain=(1.0, 1.0), phase=(0.0, 0.0)):
    """
    Linear chirp factory. Returns f(t)->R^2.
    phase given in cycles. x and y sweep at slightly different rates.
    """
    kx = (f1 - f0) / max(T, 1e-9)
    ky = 0.7 * kx

    def f(t: float):
        phx = 2*np.pi*(f0*t + 0.5*kx*t*t) + 2*np.pi*phase[0]
        phy = 2*np.pi*(0.8*f0*t + 0.5*ky*t*t) + 2*np.pi*phase[1]
        sx = ax_gain[0]*np.sin(phx)
        sy = ax_gain[1]*np.cos(phy)
        return mag*np.array([sx, sy], dtype=float)

    return f


def _bandlimited_noise_vec_factory(mag=10.0, K=16, fmin=0.2, fmax=3.0,
                                   ax_gain=(1.0, 1.0), seed=1234):
    """
    Deterministic band-limited 'noise' from a sum of K random sinusoids.
    Returns f(t)->R^2. Same freqs for x/y, slight detune on y to avoid overlap.
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(fmin, fmax, size=K)
    phx   = rng.uniform(0.0, 1.0, size=K)  # cycles
    phy   = rng.uniform(0.0, 1.0, size=K)
    detune = rng.uniform(0.9, 1.1, size=K)
    amp = 1.0 / np.sqrt(max(K, 1))         # keep variance roughly bounded

    def f(t: float):
        w = 2*np.pi*freqs
        sx = np.sum(np.sin(w*t + 2*np.pi*phx))
        sy = np.sum(np.sin((w*detune)*t + 2*np.pi*phy))
        return mag*amp*np.array([ax_gain[0]*sx, ax_gain[1]*sy], dtype=float)

    return f


# -------------------------------
# with_llc disturbances (enter through plant)
# -------------------------------
# ---------- with_llc disturbances (enter through plant) ----------
def make_matched_fn(on=True, kind="sine", MAG=10.0, FREQ=10.0, **kw):
    if not on: return constant_vec((0.0, 0.0))
    if kind == "const":      return constant_vec(kw.get("val", (0.2, -0.1)))
    if kind == "sine":       return lambda t: _sine_vec(t, MAG, FREQ,
                                   ax_gain=kw.get("ax_gain",(0.6,0.9)),
                                   phase=kw.get("phase",(0.0,0.0)))
    if kind == "pulse":      return lambda t: _pulse_vec(t, MAG, FREQ,
                                   kw.get("duty",0.25),
                                   kw.get("phase_x",0.0), kw.get("phase_y",0.3),
                                   ax_gain=kw.get("ax_gain",(0.6,0.9)))
    # ✅ square = pulse with 50% duty
    if kind == "square":     return lambda t: _pulse_vec(t, MAG, FREQ,
                                   0.5, kw.get("phase_x",0.0), kw.get("phase_y",0.3),
                                   ax_gain=kw.get("ax_gain",(0.6,0.9)))
    if kind == "multisine":  return _multisine_vec_factory(
                                   mag=kw.get("MAG", MAG), K=kw.get("K", 7),
                                   fmin=kw.get("fmin", 1.5), fmax=kw.get("fmax", 10.0),
                                   ax_gain=kw.get("ax_gain", (0.6,0.9)), seed=kw.get("seed", 123))
    if kind == "chirp":      return _chirp_vec_factory(
                                   mag=kw.get("MAG", MAG), f0=kw.get("f0", 1.0), f1=kw.get("f1", 10.0),
                                   T=kw.get("T", 8.0), ax_gain=kw.get("ax_gain",(0.6,0.9)),
                                   phase=kw.get("phase",(0.0,0.0)))
    if kind == "noise":      return _bandlimited_noise_vec_factory(
                                   mag=kw.get("MAG", MAG), K=kw.get("K", 16),
                                   fmin=kw.get("fmin", 1.0), fmax=kw.get("fmax", 10.0),
                                   ax_gain=kw.get("ax_gain",(0.7,0.7)), seed=kw.get("seed", 999))
    raise ValueError(f"Unknown matched kind: {kind}")

def make_unmatched_fn(on=True, kind="sine", MAG=10.0, FREQ=10.0, **kw):
    if not on: return constant_vec((0.0, 0.0))
    if kind == "const":      return constant_vec(kw.get("val",(8.0, 6.0)))
    if kind == "sine":       return lambda t: _sine_vec(t, MAG, FREQ,
                                   ax_gain=kw.get("ax_gain",(1.0,0.8)),
                                   phase=kw.get("phase",(0.1,0.2)))
    if kind == "pulse":      return lambda t: _pulse_vec(t, MAG, FREQ,
                                   kw.get("duty",0.35),
                                   kw.get("phase_x",0.1), kw.get("phase_y",0.2),
                                   ax_gain=kw.get("ax_gain",(1.0,1.0)))
    # ✅ square
    if kind == "square":     return lambda t: _pulse_vec(t, MAG, FREQ,
                                   0.5, kw.get("phase_x",0.1), kw.get("phase_y",0.2),
                                   ax_gain=kw.get("ax_gain",(1.0,1.0)))
    if kind == "multisine":  return _multisine_vec_factory(
                                   mag=kw.get("MAG", MAG), K=kw.get("K", 11),
                                   fmin=kw.get("fmin", 1.0), fmax=kw.get("fmax", 10.0),
                                   ax_gain=kw.get("ax_gain", (1.0,1.0)), seed=kw.get("seed", 456))
    if kind == "chirp":      return _chirp_vec_factory(
                                   mag=kw.get("MAG", MAG), f0=kw.get("f0", 1.0), f1=kw.get("f1", 10.0),
                                   T=kw.get("T", 1.0), ax_gain=kw.get("ax_gain",(1.0,1.0)),
                                   phase=kw.get("phase",(0.0,0.0)))
    if kind == "noise":      return _bandlimited_noise_vec_factory(
                                   mag=kw.get("MAG", MAG), K=kw.get("K", 24),
                                   fmin=kw.get("fmin", 0.2), fmax=kw.get("fmax", 4.0),
                                   ax_gain=kw.get("ax_gain",(1.0,1.0)), seed=kw.get("seed", 2025))
    raise ValueError(f"Unknown unmatched kind: {kind}")

def make_direct_fn(on=True, kind="sine", MAG=0.8, FREQ=0.5, **kw):
    if not on: return constant_vec((0.0, 0.0))
    if kind == "const":      return constant_vec(kw.get("val",(0.2, 0.0)))
    if kind == "sine":       return lambda t: _sine_vec(t, MAG, FREQ,
                                   ax_gain=kw.get("ax_gain",(1.0,0.8)),
                                   phase=kw.get("phase",(0.0,0.3)))
    if kind == "pulse":      return lambda t: _pulse_vec(t, MAG, FREQ,
                                   kw.get("duty",0.25),
                                   kw.get("phase_x",0.0), kw.get("phase_y",0.3),
                                   ax_gain=kw.get("ax_gain",(1.0,0.8)))
    # ✅ square
    if kind == "square":     return lambda t: _pulse_vec(t, MAG, FREQ,
                                   0.5, kw.get("phase_x",0.0), kw.get("phase_y",0.3),
                                   ax_gain=kw.get("ax_gain",(1.0,0.8)))
    if kind == "multisine":  return _multisine_vec_factory(
                                   mag=kw.get("MAG", MAG), K=kw.get("K", 9),
                                   fmin=kw.get("fmin", 0.12), fmax=kw.get("fmax", 2.2),
                                   ax_gain=kw.get("ax_gain",(1.0,0.8)), seed=kw.get("seed", 314))
    if kind == "chirp":      return _chirp_vec_factory(
                                   mag=kw.get("MAG", MAG), f0=kw.get("f0", 0.08), f1=kw.get("f1", 2.2),
                                   T=kw.get("T", 1.0), ax_gain=kw.get("ax_gain",(1.0,0.8)),
                                   phase=kw.get("phase",(0.0,0.0)))
    if kind == "noise":      return _bandlimited_noise_vec_factory(
                                   mag=kw.get("MAG", MAG), K=kw.get("K", 18),
                                   fmin=kw.get("fmin", 1.5), fmax=kw.get("fmax", 20.0),
                                   ax_gain=kw.get("ax_gain",(1.0,0.8)), seed=kw.get("seed", 2718))
    raise ValueError(f"Unknown direct kind: {kind}")

# -------------------------------
# Convenience “mid pulses” used in some experiments
# -------------------------------
def big_mid_pulse(center: float = 0.5, width: float = 0.30,
                  mag: float = 2.0, ax_gain=(1.0, 0.8)):
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
    t01 = float(center1 - width1/2.0); t11 = float(center1 + width1/2.0)
    t02 = float(center2 - width2/2.0); t12 = float(center2 + width2/2.0)
    t01, t11 = max(0.0, t01), min(1.0, t11)
    t02, t12 = max(0.0, t02), min(1.0, t12)
    def f(t_norm: float) -> np.ndarray:
        on1 = (t_norm >= t01) and (t_norm <= t11)
        on2 = (t_norm >= t02) and (t_norm <= t12)
        s1 = mag1 if on1 else 0.0
        s2 = mag2 if on2 else 0.0
        return np.array(
            [ax_gain1[0]*s1 + ax_gain2[0]*s2,
             ax_gain1[1]*s1 + ax_gain2[1]*s2],
            dtype=float,
        )
    return f
