"""Nonlinear and spectral descriptors of a standardized residual series z."""
from collections import Counter

import nolds
import numpy as np
from MFDFA import MFDFA
from scipy import signal
from scipy.spatial import cKDTree


def dfa_alpha(z):
    """DFA scaling exponent with ordinary least-squares fit."""
    return nolds.dfa(z, fit_exp='poly')


def generalized_hurst(z, q=np.r_[-5:0, 1:6], lag_min=10, frac=4, n_lags=20):
    """Generalized Hurst exponents h(q) from MFDFA; returns q, h(q) and the standard error of h(2)."""
    lags = np.unique(np.geomspace(lag_min, max(20, len(z) // frac), n_lags).astype(int))
    lags, F = MFDFA(z, lags, q=q.astype(float), order=1)
    h = np.array([np.polyfit(np.log(lags), np.log(F[:, k]), 1)[0] for k in range(len(q))])
    _, cov = np.polyfit(np.log(lags), np.log(F[:, list(q).index(2)]), 1, cov=True)
    return q, h, np.sqrt(cov[0, 0])


def sample_entropy(z, m=2, r_frac=0.2, valid=None):
    """Sample entropy; if `valid` is given, only templates lying entirely in valid samples are counted."""
    n = len(z); r = r_frac * (z[valid].std() if valid is not None else z.std())
    start = np.arange(n - m)
    if valid is not None:
        start = start[[valid[i:i + m + 1].all() for i in start]]
    count = lambda k: cKDTree(np.array([z[i:i + k] for i in start])).query_ball_point(
        np.array([z[i:i + k] for i in start]), r, p=np.inf, return_length=True).sum() - len(start)
    B, A = count(m), count(m + 1)
    return -np.log(A / B) if A > 0 and B > 0 else np.nan


def correlation_dimension(z, emb_dim=6, rvals=np.logspace(np.log10(0.3), np.log10(4), 20)):
    return nolds.corr_dim(z, emb_dim=emb_dim, rvals=rvals, fit='poly')


def visibility_degrees(z):
    """Degree sequence of the natural visibility graph."""
    n = len(z); deg = np.zeros(n, int)
    for i in range(n - 1):
        j = np.arange(i + 1, n)
        slopes = (z[j] - z[i]) / (j - i)
        vis = j[slopes > np.maximum.accumulate(np.r_[-np.inf, slopes[:-1]])]
        deg[i] += len(vis); deg[vis] += 1
    return deg


def degree_tail_slope(deg, kmin=4):
    k, c = np.unique(deg, return_counts=True); m = k >= kmin
    return -np.polyfit(np.log(k[m]), np.log(c[m] / c.sum()), 1)[0]


def time_irreversibility(z, tau=1):
    d = z[tau:] - z[:-tau]
    return np.mean(d ** 3) / np.mean(d ** 2) ** 1.5


def high_frequency_fraction(z, fs, f_split=0.5, f_min=0.02):
    f, P = signal.welch(z, fs=fs, nperseg=min(128, len(z)))
    return P[f >= f_split].sum() / P[f >= f_min].sum()


def spectral_exponent(z, fs, band=(0.05, 1.0)):
    f, P = signal.welch(z, fs=fs, nperseg=min(128, len(z)))
    m = (f >= band[0]) & (f <= band[1])
    return -np.polyfit(np.log(f[m]), np.log(P[m]), 1)[0]


def zero_one_test(z, n_c=8, seed=0):
    """Gottwald-Melbourne 0-1 test (median K over random c)."""
    rng = np.random.default_rng(seed); n = len(z); ncut = max(10, n // 10); j = np.arange(1, n + 1); K = []
    for c in rng.uniform(np.pi / 5, 4 * np.pi / 5, n_c):
        p, q = np.cumsum(z * np.cos(c * j)), np.cumsum(z * np.sin(c * j)); nn = np.arange(1, ncut + 1)
        M = np.array([np.mean((p[k:] - p[:-k]) ** 2 + (q[k:] - q[:-k]) ** 2) for k in nn])
        D = M - np.mean(z) ** 2 * (1 - np.cos(nn * c)) / (1 - np.cos(c))
        K.append(np.corrcoef(nn, D)[0, 1])
    return float(np.median(K))


def lyapunov(z, emb_dim=4, min_tsep=5):
    return nolds.lyap_r(z, emb_dim=emb_dim, min_tsep=min_tsep, fit='poly')


def iaaft(z, rng, n_iter=50):
    """Iterative amplitude-adjusted Fourier transform surrogate."""
    amp, zs, s = np.abs(np.fft.rfft(z)), np.sort(z), rng.permutation(z).astype(float)
    for _ in range(n_iter):
        s = np.fft.irfft(amp * np.exp(1j * np.angle(np.fft.rfft(s))), n=len(z))
        s = zs[np.argsort(np.argsort(s))]
    return s


def surrogate_z(z, statistic, n_surr=25, seed=0):
    """Standardized deviation of a statistic from its IAAFT surrogate distribution."""
    rng = np.random.default_rng(seed)
    ref = np.array([statistic(iaaft(z, rng)) for _ in range(n_surr)])
    return (statistic(z) - ref.mean()) / ref.std()


def transfer_entropy(x, y, bins=4, lag=1):
    """Binned transfer entropy from x to y (bits)."""
    disc = lambda v: np.clip(np.digitize(v, np.quantile(v, np.linspace(0, 1, bins + 1))[1:-1]), 0, bins - 1)
    xs, ys = disc(x), disc(y)
    x0, y0, y1 = xs[:-lag], ys[:-lag], ys[lag:]
    n = len(y1); cxyz, cyz, cy, cxy = Counter(zip(y1, y0, x0)), Counter(zip(y1, y0)), Counter(y0), Counter(zip(y0, x0))
    return sum(v / n * np.log2((v / cxy[(b, c)]) / (cyz[(a, b)] / cy[b])) for (a, b, c), v in cxyz.items())


def coherence_band(u, y, fs, band=(0.1, 1.0), nperseg=64):
    f, C = signal.coherence(u, y, fs=fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1])
    return C[m].mean()


def arx_residual(y, u, p=4):
    """Residual of a linear ARX model of order p driven by the input u."""
    X = np.array([np.r_[y[k - p:k][::-1], u[k - p:k + 1][::-1], 1.0] for k in range(p, len(y))])
    beta = np.linalg.lstsq(X, y[p:], rcond=None)[0]
    return y[p:] - X @ beta
