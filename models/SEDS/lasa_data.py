from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import interpolate
import pyLasaDataset as lasa
import warnings

# Use the official list of names provided by the package
def get_shape_names() -> List[str]:
    """Returns the official list of shape names from pyLasaDataset."""
    try:
        # Check if the .names attribute exists and is populated
        if hasattr(lasa.DataSet, 'names') and lasa.DataSet.names:
            return sorted(list(lasa.DataSet.names))
        else:
            # Fallback: Discover shapes by inspecting attributes (less reliable)
            warnings.warn("pyLasaDataset.DataSet.names not found or empty. Falling back to attribute inspection.")
            all_names = [n for n in dir(lasa.DataSet) if not n.startswith("_")]
            keep = []
            for n in all_names:
                try:
                    obj = getattr(lasa.DataSet, n)
                    if hasattr(obj, "demos") and hasattr(obj, "dt"):
                        keep.append(n)
                except Exception:
                    continue # Ignore attributes that cause errors
            if not keep:
                 raise RuntimeError("Could not discover any valid LASA shapes by attribute inspection.")
            return sorted(keep)

    except Exception as e:
        print(f"--- ERROR accessing pyLasaDataset shapes ---")
        print(f"Error: {e}")
        print("Please ensure pyLasaDataset is installed correctly and accessible.")
        # Return empty list or re-raise, depending on desired strictness
        # raise RuntimeError("Failed to get shape names from pyLasaDataset") from e
        return []


@dataclass
class LasaShape:
    name: str
    dt: float
    # pos/vel arrays: (num_demos, T, 2), times: (T,)
    pos: np.ndarray
    vel: np.ndarray
    t: np.ndarray # Original time vector
    duration: float # Duration of the original demonstration

def load_shape(name: str) -> LasaShape:
    """Loads a LASA shape, including original time and duration."""
    if not hasattr(lasa.DataSet, name):
         # Attempt to refresh names if load fails initially
         available_names = get_shape_names()
         if name not in available_names:
             raise AttributeError(f"Unknown LASA shape: '{name}'. Available: {', '.join(available_names)}")

    data = getattr(lasa.DataSet, name)
    demos = data.demos
    nd = len(demos)

    # Use the time vector from the first demo to determine original shape/duration
    t_orig = demos[0].t.reshape(-1)
    T = t_orig.shape[0]
    duration_orig = t_orig[-1] - t_orig[0]

    # Load all demos - assuming they have the same number of points T
    pos = np.zeros((nd, T, 2))
    vel = np.zeros((nd, T, 2))
    all_times_match = True
    for i, d in enumerate(demos):
        if d.pos.shape[-1] != T or d.vel.shape[-1] != T:
             raise ValueError(f"Inconsistent number of samples in demo {i} for shape {name}. Expected {T}.")
        pos[i] = d.pos.T
        vel[i] = d.vel.T
        if not np.allclose(d.t.reshape(-1), t_orig):
            all_times_match = False

    if not all_times_match:
         warnings.warn(f"Time vectors inconsistent across demos for shape {name}. Using time from first demo.")

    return LasaShape(name=name, dt=float(data.dt), pos=pos, vel=vel, t=t_orig, duration=duration_orig)

def resample(shape: LasaShape, nsamples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample each demo to nsamples uniformly in normalized time [0, 1].

    Args:
        shape (LasaShape): The loaded LASA shape object.
        nsamples (int): The target number of samples.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - pos_rs (np.ndarray): Resampled positions (D, nsamples, dim).
            - vel_rs (np.ndarray): Resampled velocities (D, nsamples, dim).
            - t_rs (np.ndarray): Normalized time vector [0, 1] (nsamples,).
    """
    D, _, dim = shape.pos.shape
    t_orig = shape.t # Original time vector
    t_norm = (t_orig - t_orig[0]) / (t_orig[-1] - t_orig[0]) # Normalize original time to [0, 1]

    t_rs = np.linspace(0.0, 1.0, nsamples) # Target normalized time points
    pos_rs = np.zeros((D, nsamples, dim))
    vel_rs = np.zeros((D, nsamples, dim))

    for i in range(D):
        for j in range(dim):
            try:
                # Use cubic interpolation for smoother results if possible
                f_p = interpolate.interp1d(t_norm, shape.pos[i, :, j], kind="cubic", bounds_error=False, fill_value="extrapolate")
                f_v = interpolate.interp1d(t_norm, shape.vel[i, :, j], kind="cubic", bounds_error=False, fill_value="extrapolate")
                pos_rs[i, :, j] = f_p(t_rs)
                vel_rs[i, :, j] = f_v(t_rs)
            except ValueError:
                 # Fallback to linear if cubic fails (e.g., too few points)
                 f_p = interpolate.interp1d(t_norm, shape.pos[i, :, j], kind="linear", bounds_error=False, fill_value="extrapolate")
                 f_v = interpolate.interp1d(t_norm, shape.vel[i, :, j], kind="linear", bounds_error=False, fill_value="extrapolate")
                 pos_rs[i, :, j] = f_p(t_rs)
                 vel_rs[i, :, j] = f_v(t_rs)

    return pos_rs, vel_rs, t_rs

# Note: train_test_split is not used by SEDS training script, kept for compatibility maybe?
def train_test_split(pos, vel, ntrain: int) -> Tuple[Tuple[np.ndarray,np.ndarray], Tuple[np.ndarray,np.ndarray]]:
    assert 0 < ntrain < pos.shape[0], "ntrain must be between 1 and num_demos-1"
    return (pos[:ntrain], vel[:ntrain]), (pos[ntrain:], vel[ntrain:])
