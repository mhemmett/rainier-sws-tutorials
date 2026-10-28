"""Acceptance criteria used by the research pipeline, written down in one place.

Two layers of filtering happen in the real pipeline:

PRE-processing gates (decide whether we even attempt a measurement):
    rectilinearity >= 0.5    the P wave looks like a P wave (instrument sane)
    incidence     <= 45 deg  inside the "shear-wave window" at the free surface
    SNR           >= 2.0     the S wave stands out from the noise

POST-processing filter (decides whether a measurement is trusted):
    snr_horizontal >= 2.0
    quality (Q_w)  >= 0.75    (the research repo's cut since October 2026; it used 0.5 before)
    dt_error       <= 0.05 s
    phi_error      <= 20 deg
    dt             <= dominant_period / 2   (longer = cycle skip, not a delay)

Both sets are choices. A large part of this tutorial series is asking how much the
answer depends on them. Keep them as named parameters, never magic numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreGates:
    rectilinearity_min: float = 0.5
    incidence_max_deg: float = 45.0
    snr_min: float = 2.0


@dataclass(frozen=True)
class PostFilter:
    snr_min: float = 2.0
    qw_min: float = 0.75
    dt_error_max: float = 0.05
    phi_error_max: float = 20.0
    dt_period_fraction: float = 0.5  # dt <= dominant_period * fraction

    def as_dict(self):
        return asdict(self)


DEFAULT_PRE = PreGates()
DEFAULT_POST = PostFilter()


MEASUREMENT_COLUMNS = ["success", "phi", "dt", "phi_error", "dt_error", "quality", "snr_horizontal", "dominant_period"]


def _complete(results: pd.DataFrame) -> pd.DataFrame:
    """A station where no event reached the measurement has no phi/dt columns at all
    (pandas only writes columns that some row filled). Add them, empty, so every
    filter below works on every station."""
    r = results.copy()
    for c in MEASUREMENT_COLUMNS:
        if c not in r.columns:
            r[c] = False if c == "success" else np.nan
    r["success"] = r["success"].fillna(False).astype(bool)
    return r


def successful(results: pd.DataFrame) -> pd.DataFrame:
    """Rows where the pipeline reached the end and returned finite phi, dt."""
    r = _complete(results)
    ok = r[(r["stage"] == "ok") & r["success"]]
    return ok[np.isfinite(ok["phi"]) & np.isfinite(ok["dt"])]


def post_filter_mask(results: pd.DataFrame, g: PostFilter = DEFAULT_POST) -> pd.Series:
    """Boolean mask of rows that pass the post-processing filter."""
    results = _complete(results)
    return (
        (results["stage"] == "ok") & results["success"]
        & (results["snr_horizontal"] >= g.snr_min)
        & (results["quality"] >= g.qw_min)
        & (results["dt_error"] <= g.dt_error_max)
        & (results["phi_error"] <= g.phi_error_max)
        & (results["dt"] <= results["dominant_period"] * g.dt_period_fraction)
    )


def post_filtered(results: pd.DataFrame, g: PostFilter = DEFAULT_POST) -> pd.DataFrame:
    """The rows that pass the post-processing filter (with every measurement column, even if empty)."""
    results = _complete(results)
    return results[post_filter_mask(results, g)].copy()


def stage_counts(results: pd.DataFrame) -> pd.Series:
    """How many events stopped at each stage (nothing is silently dropped)."""
    return results["stage"].value_counts()


def funnel(results: pd.DataFrame, g: PostFilter = DEFAULT_POST) -> pd.Series:
    """Attempted -> successful -> passed the post-processing filter, as a 3-row Series."""
    return pd.Series({
        "attempted": len(results),
        "successful": len(successful(results)),
        "filtered": int(post_filter_mask(results, g).sum()),
    })
