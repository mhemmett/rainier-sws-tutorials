"""Tests for rainier_sws.timing (notebook 05b). Run with: pytest -q tests/test_timing.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rainier_sws import data, paths, timing  # noqa: E402


def test_catalog_residuals_have_sensible_velocities():
    cat = pd.read_csv(paths.STRICT_CATALOG_CSV)
    sta = data.load_stations().set_index("station")
    res, fit = timing.travel_time_residuals(cat, sta)
    assert 4.5 < fit["vp_km_s"] < 7.5 and 2.5 < fit["vs_km_s"] < 4.5
    assert {"res_p", "res_s", "R_km"} <= set(res.columns)
    assert abs(res.res_p.median()) < 0.02            # constants absorb the mean


def test_star_residuals_move_together():
    """The observation notebook 05b is built on: at STAR, daily P and S residuals co-vary."""
    cat = pd.read_csv(paths.STRICT_CATALOG_CSV)
    sta = data.load_stations().set_index("station")
    res, _ = timing.travel_time_residuals(cat, sta)
    d = timing.daily_residuals(res, "STAR")
    r = np.corrcoef(d.res_p, d.res_s)[0, 1]
    assert len(d) >= 15 and r > 0.5
    assert (d.res_p.max() - d.res_p.min()) > 0.06    # at least 60 ms peak to peak


@pytest.mark.slow
def test_synthetic_clock_drift_is_recovered():
    def truth_fn(t):
        return 0.002 * np.asarray(t, float) / 3600.0  # 2 ms per hour

    st, truth = timing.synthetic_pair(hours=12, fs=20.0, lag_s=1.2, delta_t_fn=truth_fn, snr=0.7, seed=3)
    a, b, fs, t0 = timing.align_pair(st[0], st[1])
    track = timing.run_pair(a, b, fs, t0)
    dt, _ = timing.delta_t_from_lags(track.peak_lag_s.values, anchor=1.2)
    assert len(dt) == 12
    assert np.nanmedian(np.abs(dt - truth)) < 0.010   # well below the 50 ms sample quantum


def test_align_pair_interpolates_fractional_offsets():
    st, _ = timing.synthetic_pair(hours=0.5, fs=20.0, seed=4)
    tr = st[1].copy()
    tr.stats.starttime += 0.017                       # not on the 50 ms grid
    a, b, fs, t0 = timing.align_pair(st[0], tr)
    assert len(a) == len(b) and t0.timestamp == int(t0.timestamp)


@pytest.mark.skipif(not timing.CHRONOS_AVAILABLE, reason="chronos/chronfix not installed")
def test_clock_model_round_trip():
    hours = np.arange("2025-07-10T00", "2025-07-10T06", dtype="datetime64[h]")
    dt_model = np.array([0.0, 0.01, 0.02, np.nan, 0.05, 0.05])
    trig = np.array([False, False, False, True, False, False])
    m = timing.clock_model(hours, dt_model, trig, station="X")
    assert len(m.trigger_starts) == 1
    assert m.interp_delta_t(np.datetime64("2025-07-10T01:30:00")) == pytest.approx(0.015)
    assert np.isnan(m.interp_delta_t(np.datetime64("2025-07-10T03:30:00")))
