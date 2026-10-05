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
