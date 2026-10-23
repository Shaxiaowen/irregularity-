"""Signal preprocessing and low-groundspeed airflow proxy."""
import numpy as np


def active_segment(current, threshold):
    """Indices between the first and last samples at which current exceeds the threshold."""
    idx = np.flatnonzero(current > threshold)
    return slice(idx[0], idx[-1] + 1)


def resample(t, x, dt):
    """Linear interpolation onto a uniform grid with spacing dt."""
    tu = t[0] + np.arange(int(np.floor((t[-1] - t[0]) / dt)) + 1) * dt
    return tu, np.interp(tu, t, x)


def detrend_standardize(t, x, order=2):
    """Remove a polynomial trend and scale the residual to zero mean and unit variance."""
    r = x - np.polyval(np.polyfit(t, x, order), t)
    return (r - r.mean()) / r.std()


def airflow_proxy(reading, vx, vy, v0=0.5, min_samples=20):
    """Mean anemometer reading over samples with horizontal groundspeed below v0.

    For |v_g| < v0 the reading differs from ambient wind speed by less than v0 (triangle inequality).
    """
    low = np.hypot(vx, vy) < v0
    return reading[low].mean() if low.sum() >= min_samples else np.nan, low


def low_speed_runs(low, min_len):
    """Contiguous low-groundspeed runs of at least min_len samples, as (start, end) pairs."""
    d = np.diff(np.r_[0, low.astype(int), 0])
    return [(s, e) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if e - s >= min_len]
