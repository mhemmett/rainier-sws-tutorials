"""The single-event shear-wave splitting measurement, step by step.

This mirrors ``process_event`` in the research repo's ``run_sws_generic.py`` so
that a measurement made here is the same measurement Michael reported, with one
deliberate difference: the research pipeline computes the incidence angle by
3-D ray tracing through a Vs model (PyKonal). That dependency is heavy, so here
you pass the incidence angle in. For STAR events the ray-traced value is stored
in ``results/sws_star_strict_results.csv`` and can be looked up; for new events
you can use the P-wave polarization (Jurkevics) estimate as a stand-in and treat
the difference as one more uncertainty.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime

from . import sws_functions as swsf
from .quality import PreGates, DEFAULT_PRE
from swspy.splitting import split_windowcheck


@dataclass
class SwsParams:
    """swspy window / grid-search parameters, with the research-pipeline defaults."""
    s_pick_uncertainty: float = 0.03     # s; S-pick jitter the windows sweep over
    first_window_start: float = 2        # window boundaries in units of dominant period
    last_window_start: float = 1
    first_window_end: float = 1.5
    last_window_end: float = 2.5
    n_win: int = 7                       # number of window positions tried
    max_t_shift_s: float = 0.2           # largest dt searched
    cluster_eps: float = 0.15
    cluster_min_samples: int = 15
    coord_system: str = "LQT"
    sws_method: str = "EV_and_XC"


@dataclass
class Measurement:
    evid: int
    station: str
    stage: str = "ok"
    back_azimuth: float = np.nan
    rectilinearity: float = np.nan
    incidence_angle_jurkevics: float = np.nan
    incidence_angle: float = np.nan
    snr_horizontal: float = np.nan
    filter_band: tuple | None = None
    phi: float = np.nan
    dt: float = np.nan
    phi_error: float = np.nan
    dt_error: float = np.nan
    quality: float = np.nan
    dominant_period: float = np.nan
    success: bool = False
    error: str = ""
    splitting_object: object = field(default=None, repr=False)
    stream_filtered: obspy.Stream | None = field(default=None, repr=False)

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("splitting_object", "stream_filtered")}
        d["filter_band"] = str(self.filter_band)
        return d


def pick_offsets(stream: obspy.Stream, row: pd.Series):
    """(origin, p_time, s_time) as UTCDateTime and (p_offset, s_offset) in s from trace start."""
    from .data import to_utc
    origin, p_time, s_time = to_utc(row["origin_time"]), to_utc(row["p_time"]), to_utc(row["s_time"])
    t0 = stream[0].stats.starttime
    return origin, p_time, s_time, p_time - t0, s_time - t0


def measure_event(row: pd.Series, stream: obspy.Stream, station: str, sta_lat: float, sta_lon: float,
                  incidence_angle: float | None = None, gates: PreGates = DEFAULT_PRE,
                  params: SwsParams = SwsParams(), verbose: bool = False) -> Measurement:
    """Run the full single-event measurement and return a Measurement.

    ``incidence_angle``: degrees from vertical used for the QC gate and the LQT
    rotation. ``None`` -> use the Jurkevics P-wave estimate from the data.
    """
    m = Measurement(evid=int(row["evid"]), station=station)
    z, n, e = stream.select(channel="??Z")[0], stream.select(channel="??N")[0], stream.select(channel="??E")[0]
    origin, p_time, s_time, p_off, s_off = pick_offsets(stream, row)

    # 1. Geometry and P-wave polarization checks
    m.back_azimuth = swsf.calculate_back_azimuth(row["event_lat"], row["event_lon"], sta_lat, sta_lon)
    m.rectilinearity = float(np.real(swsf.calculate_rectilinearity_jurkevics(z, n, e, p_off)))
    m.incidence_angle_jurkevics = float(np.real(swsf.calculate_incidence_angle_eigenvalue_jurkevics(z, n, e, p_off)))
    m.incidence_angle = float(incidence_angle) if incidence_angle is not None else m.incidence_angle_jurkevics

    # 2. Filter-bank: choose the band that maximizes S-wave SNR
    raw = obspy.Stream([z, n, e])
    band, best_stream, snr, dom_period_samples, snr_by_band = swsf.try_filters(raw, p_time, s_time)
    m.snr_horizontal, m.filter_band, m.stream_filtered = float(snr), band, best_stream

    # 3. Pre-processing gates (in the pipeline's order)
    if np.isnan(m.rectilinearity) or m.rectilinearity < gates.rectilinearity_min:
        m.stage = "fail_rectilinearity"; return m
    if np.isnan(m.incidence_angle) or m.incidence_angle > gates.incidence_max_deg:
        m.stage = "fail_incidence"; return m
    if band is None or np.isnan(snr) or snr < gates.snr_min:
        m.stage = "fail_snr"; return m

    # 4. The splitting measurement itself (swspy)
    try:
        sink = io.StringIO()
        with (contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)):
            so = split_windowcheck.create_splitting_object(
                best_stream, stations_in=[station], back_azis_all_stations=[m.back_azimuth],
                receiver_inc_angles_all_stations=[m.incidence_angle],
                S_phase_arrival_times=[s_time], origin_times=[origin],
                first_window_start=params.first_window_start, last_window_start=params.last_window_start,
                first_window_end=params.first_window_end, last_window_end=params.last_window_end,
                n_win=params.n_win, s_pick_uncertainty=params.s_pick_uncertainty,
                max_t_shift_s=params.max_t_shift_s)
            so.perform_sws_analysis(coord_system=params.coord_system, sws_method=params.sws_method,
                                    cluster_eps=params.cluster_eps, cluster_min_samples=params.cluster_min_samples)
        m.splitting_object = so
        df = so.sws_result_df
        if df is None or len(df) == 0:
            m.stage = "fail_no_result"; return m
        r0 = df.iloc[0]
        m.phi, m.dt = float(r0["phi_from_N"]), float(r0["dt"])
        m.phi_error, m.dt_error = float(r0["phi_err"]), float(r0["dt_err"])
        m.quality, m.dominant_period = float(r0["Q_w"]), float(so.Tmid)
        m.success = bool(np.isfinite(m.phi) and np.isfinite(m.dt))
    except Exception as exc:  # keep going; record why
        m.stage, m.error = "fail_exception", f"{type(exc).__name__}: {exc}"
    return m


def phi_0_180(phi: np.ndarray | float) -> np.ndarray:
    """Fold a fast direction (axial, clockwise from north) into [0, 180)."""
    return np.mod(np.asarray(phi, dtype=float), 180.0)
