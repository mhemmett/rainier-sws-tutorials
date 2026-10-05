"""Acceptance criteria used by the research pipeline, written down in one place.

Two layers of filtering happen in the real pipeline:

PRE-processing gates (decide whether we even attempt a measurement):
    rectilinearity >= 0.5    the P wave looks like a P wave (instrument sane)
    incidence     <= 45 deg  inside the "shear-wave window" at the free surface
    SNR           >= 2.0     the S wave stands out from the noise

POST-processing filter ("grade 3", decides whether a measurement is trusted):
    snr_horizontal >= 2.0
    quality (Q_w)  >= 0.5     (some scripts use 0.75)
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
class Grade3:
    snr_min: float = 2.0
    qw_min: float = 0.5
    dt_error_max: float = 0.05
    phi_error_max: float = 20.0
    dt_period_fraction: float = 0.5  # dt <= dominant_period * fraction

    def as_dict(self):
        return asdict(self)


DEFAULT_PRE = PreGates()
DEFAULT_GRADE3 = Grade3()


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


def grade3_mask(results: pd.DataFrame, g: Grade3 = DEFAULT_GRADE3) -> pd.Series:
    """Boolean mask of rows that pass the grade-3 post-processing filter."""
    results = _complete(results)
    return (
        (results["stage"] == "ok") & results["success"]
        & (results["snr_horizontal"] >= g.snr_min)
        & (results["quality"] >= g.qw_min)
        & (results["dt_error"] <= g.dt_error_max)
        & (results["phi_error"] <= g.phi_error_max)
        & (results["dt"] <= results["dominant_period"] * g.dt_period_fraction)
    )


def grade3(results: pd.DataFrame, g: Grade3 = DEFAULT_GRADE3) -> pd.DataFrame:
    return results[grade3_mask(results, g)].copy()


def stage_counts(results: pd.DataFrame) -> pd.Series:
    """How many events stopped at each stage (nothing is silently dropped)."""
    return results["stage"].value_counts()


def funnel(results: pd.DataFrame, g: Grade3 = DEFAULT_GRADE3) -> pd.Series:
    """Attempted -> successful -> grade 3, as a 3-row Series."""
    return pd.Series({
        "attempted": len(results),
        "successful": len(successful(results)),
        "grade3": int(grade3_mask(results, g).sum()),
    })
