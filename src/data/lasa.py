# src/data/lasa.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import interpolate
import pyLasaDataset as lasa

# Map public names -> pyLasaDataset attributes
def get_shape_names() -> List[str]:
    # Discover shapes from pyLasaDataset.DataSet
    all_names = [n for n in dir(lasa.DataSet) if not n.startswith("_")]
    # Keep only those with .demos and .dt
    keep = []
    for n in all_names:
        obj = getattr(lasa.DataSet, n)
        if hasattr(obj, "demos") and hasattr(obj, "dt"):
            keep.append(n)
    return sorted(keep)

@dataclass
class LasaShape:
    name: str
    dt: float
    # pos/vel arrays: (num_demos, T, 2), times: (T,)
    pos: np.ndarray
    vel: np.ndarray
    t:   np.ndarray

def load_shape(name: str) -> LasaShape:
    assert hasattr(lasa.DataSet, name), f"Unknown LASA shape: {name}"
    data = getattr(lasa.DataSet, name)
    demos = data.demos
    nd = len(demos)
    T  = demos[0].t.shape[-1]
    pos = np.zeros((nd, T, 2))
    vel = np.zeros((nd, T, 2))
    for i, d in enumerate(demos):
        pos[i] = d.pos.T
        vel[i] = d.vel.T
    # normalize time to [0,1] (keeps original dt in object)
    t = demos[0].t.reshape(-1)
    t_norm = t / t[-1]
    return LasaShape(name=name, dt=float(data.dt), pos=pos, vel=vel, t=t_norm)

def resample(shape: LasaShape, nsamples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample each demo to nsamples uniformly in normalized time.
       Returns (pos_rs, vel_rs, t_rs) with shapes:
       pos_rs:(D, nsamples, 2), vel_rs:(D, nsamples, 2), t_rs:(nsamples,) in [0,1].
    """
    D, _, dim = shape.pos.shape
    t_rs = np.linspace(0.0, 1.0, nsamples)
    pos_rs = np.zeros((D, nsamples, dim))
    vel_rs = np.zeros((D, nsamples, dim))
    for i in range(D):
        for j in range(dim):
            f_p = interpolate.interp1d(shape.t, shape.pos[i, :, j], kind="linear")
            f_v = interpolate.interp1d(shape.t, shape.vel[i, :, j], kind="linear")
            pos_rs[i, :, j] = f_p(t_rs)
            vel_rs[i, :, j] = f_v(t_rs)
    return pos_rs, vel_rs, t_rs

def train_test_split(pos, vel, ntrain: int) -> Tuple[Tuple[np.ndarray,np.ndarray], Tuple[np.ndarray,np.ndarray]]:
    assert 0 < ntrain < pos.shape[0], "ntrain must be between 1 and num_demos-1"
    return (pos[:ntrain], vel[:ntrain]), (pos[ntrain:], vel[ntrain:])
