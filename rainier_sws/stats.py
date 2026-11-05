"""Statistics for splitting results: axial (circular) statistics and bootstrap intervals.

A fast direction phi is an *axis*, not a direction: 30 deg and 210 deg are the
same crack orientation. The standard trick (Fisher, 1993, Statistical Analysis of
Circular Data, ch. 2 and 4) is to double the angles, do ordinary circular
statistics on 2*phi, then halve the result.
"""
from __future__ import annotations

import numpy as np


def axial_mean(phi_deg, weights=None):
    """Mean fast direction in [0, 180) and the mean resultant length R in [0, 1].

    R near 1: tightly clustered; R near 0: no preferred orientation.
    """
    phi = np.asarray(phi_deg, dtype=float)
    w = np.ones_like(phi) if weights is None else np.asarray(weights, dtype=float)
    theta = np.deg2rad(2.0 * phi)
    C, S = np.sum(w * np.cos(theta)) / w.sum(), np.sum(w * np.sin(theta)) / w.sum()
    R = float(np.hypot(C, S))
    mean = np.rad2deg(np.arctan2(S, C)) / 2.0
    return float(np.mod(mean, 180.0)), R


def axial_std(phi_deg):
    """Circular standard deviation of an axial sample, in degrees (Fisher eq. 2.15, halved)."""
    _, R = axial_mean(phi_deg)
    if R <= 0:
        return np.inf
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))) / 2.0)


def rayleigh_test(phi_deg):
    """Rayleigh test for a preferred axis. Returns (z, p). Small p: not uniform."""
    n = len(phi_deg)
    _, R = axial_mean(phi_deg)
    z = n * R ** 2
    # Approximation good for n >= 10 (Fisher 1993, eq. 4.17)
    p = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n) - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2))
    return float(z), float(np.clip(p, 0, 1))


def axial_diff(a_deg, b_deg):
    """Smallest angle between two axes, in [0, 90]."""
    d = np.mod(np.asarray(a_deg) - np.asarray(b_deg), 180.0)
    return np.minimum(d, 180.0 - d)


def near_axis(phi_deg, ref_deg, width_deg=20.0, perpendicular=True):
    """Boolean mask: phi within ``width_deg`` of the axis ``ref_deg`` (or of the axis 90 deg from it).

    For a null screen, ``ref_deg`` is the initial S polarization (``Measurement.src_pol``); against the
    back azimuth it is an artifact screen instead (a component error lines phi up with the rays).
    """
    d = axial_diff(phi_deg, ref_deg)
    return (d <= width_deg) | (d >= 90.0 - width_deg) if perpendicular else d <= width_deg


def chance_near_axis(width_deg=20.0, perpendicular=True):
    """Fraction of uniformly random axes that ``near_axis`` would flag (0.44 for 20 deg, both axes)."""
    return min(1.0, (4.0 if perpendicular else 2.0) * width_deg / 180.0)


def bootstrap(values, statistic, n_boot=1000, ci=68, seed=0):
    """Nonparametric bootstrap of ``statistic`` (a function of a 1-D array).

    Returns (point_estimate, lower, upper, all_boot_values).
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    boots = np.array([statistic(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(statistic(v)), float(lo), float(hi), boots


def bootstrap_axial_mean(phi_deg, n_boot=1000, ci=68, seed=0):
    """Bootstrap interval on the mean fast direction, handling the 0/180 wrap.

    We measure each bootstrap mean as its signed offset from the full-sample mean,
    so the interval is reported as mean -lo/+hi degrees.
    """
    rng = np.random.default_rng(seed)
    phi = np.asarray(phi_deg, dtype=float)
    m0, _ = axial_mean(phi)
    offs = []
    for _ in range(n_boot):
        m, _ = axial_mean(phi[rng.integers(0, len(phi), len(phi))])
        d = np.mod(m - m0 + 90.0, 180.0) - 90.0  # signed, in (-90, 90]
        offs.append(d)
    offs = np.array(offs)
    lo, hi = np.percentile(offs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return m0, float(lo), float(hi), offs


def block_bootstrap_axial_mean(phi_deg, blocks, n_boot=1000, ci=68, seed=0):
    """Like ``bootstrap_axial_mean``, but resampling whole blocks (e.g. days) instead of events.

    Events close in time are not independent (a burst of similar earthquakes on similar paths), so
    resampling them one by one understates the interval; resampling days keeps each burst together.
    """
    rng = np.random.default_rng(seed)
    phi = np.asarray(phi_deg, dtype=float)
    labels = np.asarray(blocks)
    groups = [phi[labels == b] for b in np.unique(labels)]
    m0, _ = axial_mean(phi)
    offs = []
    for _ in range(n_boot):
        m, _ = axial_mean(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
        offs.append(np.mod(m - m0 + 90.0, 180.0) - 90.0)
    offs = np.array(offs)
    lo, hi = np.percentile(offs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return m0, float(lo), float(hi), offs


def permutation_test(a, b, statistic=np.mean, n_perm=2000, axial=False, seed=0):
    """Two-sample permutation test of a difference in ``statistic``. Returns (difference, p).

    With ``axial=True`` the statistic is the axial mean and the difference is the smallest angle
    between the two means. p is the fraction of random relabellings with at least as large a difference.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if axial:
        def diff(x, y):
            return float(axial_diff(axial_mean(x)[0], axial_mean(y)[0]))
    else:
        def diff(x, y):
            return float(abs(statistic(x) - statistic(y)))
    d0, pooled, n = diff(a, b), np.concatenate([a, b]), len(a)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        hits += diff(p[:n], p[n:]) >= d0
    return d0, (hits + 1) / (n_perm + 1)
