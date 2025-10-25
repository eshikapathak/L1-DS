from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy import interpolate

# ---------- Discovery ----------

def get_shape_files(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted([p for p in root.glob("*.npy")])

def get_shape_names(root: str | Path) -> List[str]:
    return [p.stem for p in get_shape_files(root)]

# ---------- Loading ----------

def _finite_diff(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central-difference velocity with endpoints via one-sided diff."""
    dt = np.diff(t)
    dt = np.concatenate([dt[:1], dt, dt[-1:]])  # pad for indexing
    v = np.zeros_like(pos)
    # central
    v[1:-1] = (pos[2:] - pos[:-2]) / (t[2:] - t[:-2]).reshape(-1, 1)
    # ends
    v[0] = (pos[1] - pos[0]) / dt[0]
    v[-1] = (pos[-1] - pos[-2]) / dt[-1]
    return v

def load_shape(shape: str, root: str | Path = "iros_dataset") -> dict:
    """
    Returns:
      dict with keys:
        - demos: list of dict(pos(N,2), vel(N,2), t(N,))
        - name: shape name
    Accepts files that are (K,N,2) or (N,2). If (K,N,2), K demos.
    Time base normalized to [0,1].
    """
    path = Path(root) / f"{shape}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Available: {get_shape_names(root)}")

    arr = np.asarray(np.load(path, allow_pickle=True))
    demos = []

    if arr.ndim == 3 and arr.shape[-1] >= 2:
        K, N, _ = arr.shape
        t = np.linspace(0.0, 1.0, N, dtype=float)
        for k in range(K):
            pos = np.asarray(arr[k, :, :2], dtype=float)
            vel = _finite_diff(pos, t)
            demos.append(dict(pos=pos, vel=vel, t=t.copy()))
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        N = arr.shape[0]
        t = np.linspace(0.0, 1.0, N, dtype=float)
        pos = np.asarray(arr[:, :2], dtype=float)
        vel = _finite_diff(pos, t)
        demos.append(dict(pos=pos, vel=vel, t=t.copy()))
    else:
        raise ValueError(f"Unsupported IROS array shape: {arr.shape}")

    return dict(name=shape, demos=demos)

# ---------- Resampling (uniform in normalized time) ----------

def resample(data: dict, nsamples: int = 1000) -> Tuple[list, list, np.ndarray]:
    """Return lists: pos_rs[k](nsamples,2), vel_rs[k](nsamples,2), and shared t(nsamples,)."""
    t_new = np.linspace(0.0, 1.0, nsamples, dtype=float)
    pos_rs, vel_rs = [], []
    for d in data["demos"]:
        t, P, V = d["t"], d["pos"], d["vel"]
        fx = interpolate.interp1d(t, P[:, 0], kind="linear")
        fy = interpolate.interp1d(t, P[:, 1], kind="linear")
        fvx = interpolate.interp1d(t, V[:, 0], kind="linear")
        fvy = interpolate.interp1d(t, V[:, 1], kind="linear")
        pos_rs.append(np.stack([fx(t_new), fy(t_new)], axis=1))
        vel_rs.append(np.stack([fvx(t_new), fvy(t_new)], axis=1))
    return pos_rs, vel_rs, t_new
