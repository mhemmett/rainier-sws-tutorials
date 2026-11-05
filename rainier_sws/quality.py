"""Acceptance criteria used by the research pipeline, written down in one place.

Three layers of filtering happen in the real pipeline:

PICK quality (decides which earthquake-station pairs enter at all):
    P pick quality >= 0.75 and S pick quality >= 0.75   (PNSN's 0-1 scale; 1 = best)

PRE-processing gates (decide whether we even attempt a measurement):
    clip fraction  < 0.97    the S wave did not saturate the digitizer (stations with a known clip level)
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
class PickQuality:
    """Both picks of an earthquake-station pair must reach these qualities (PNSN's 0-1 scale).

    The research pipeline's results files hold exactly the pairs with both qualities >= 0.75; the
    catalog files in ``data/catalog/`` keep everything with both >= 0.5.
    """
    p_min: float = 0.75
    s_min: float = 0.75


def pick_quality_mask(catalog: pd.DataFrame, q: PickQuality = PickQuality()) -> pd.Series:
    """Rows of a pick catalog whose P and S picks are both good enough."""
    return (catalog["p_quality"] >= q.p_min) & (catalog["s_quality"] >= q.s_min)


@dataclass(frozen=True)
class PreGates:
    # peak |counts| in the first second after S over the clip level (data.CLIP_COUNTS). The research
    # re-run on corrected components applies it; the archived results did not, so pass np.inf to
    # reproduce those.
    clip_fraction_max: float = 0.97
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

# Stricter pre-processing gates, used by notebook 04: rays inside the shear-wave window (35 deg, about
# the critical angle of 35.3 deg for Vp/Vs = 1.73) and a cleaner P wave (rectilinearity 0.7). These are
# the axial-sws defaults that the research pipeline relaxed to 45 deg and 0.5 for the land stations.
# The results files and notebooks 05 to 07 use DEFAULT_PRE.
SHEAR_WINDOW_PRE = PreGates(rectilinearity_min=0.7, incidence_max_deg=35.0, snr_min=2.0)


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
