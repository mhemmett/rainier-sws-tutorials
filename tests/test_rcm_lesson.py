"""What notebook 04 (one RCM earthquake, step by step) relies on: the pick-quality filter, the ComCat
picks, the stricter gates, the PyKonal ray tracing, the measurement of event 62137112 and the
diagnostic figure. All offline: the QuakeML, the waveforms and the velocity model are in the repo."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import obspy
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rainier_sws import data, measure, paths, plots, quality  # noqa: E402

EVID = 62137112
WAVEFORMS = paths.WAVEFORMS / "notebook04" / f"UW.RCM.{EVID}.mseed"


def _row():
    r = data.load_station_catalog("RCM").set_index("evid").loc[EVID].copy()
    r["evid"] = EVID
    return r


def _ref():
    return pd.read_csv(paths.station_results_csv("RCM")).set_index("evid").loc[EVID]


@pytest.fixture(scope="module")
def ray():
    pytest.importorskip("pykonal")
    from rainier_sws import raytrace
    lat, lon, el = data.station_coords("RCM")
    tracer = raytrace.StationRayTracer()
    tracer.precompute_station("RCM", lat, lon, el)
    r = _row()
    return tracer.trace("RCM", r.event_lat, r.event_lon, r.event_depth)


@pytest.fixture(scope="module")
def m():
    lat, lon, _ = data.station_coords("RCM")
    st = data.orient_zne(obspy.read(str(WAVEFORMS)))
    return measure.measure_event(_row(), st, "RCM", lat, lon, incidence_angle=_ref().incidence_angle,
                                 gates=quality.SHEAR_WINDOW_PRE)


def test_pick_quality_filter_gives_the_results_file():
    cat = data.load_station_catalog("RCM")                       # falls back to the combined strict catalog
    kept = cat[quality.pick_quality_mask(cat)]
    res = pd.read_csv(paths.station_results_csv("RCM"))
    assert len(cat) == 856 and len(kept) == 509
    assert set(kept.evid) == set(res.evid)


def test_comcat_picks_carry_pnsn_quality():
    picks = data.load_event_picks(EVID, allow_download=False)
    rcm = picks[picks.station == "RCM"].set_index("phase")
    row = _row()
    assert rcm.loc["P", "quality"] == row.p_quality == 1.0
    assert rcm.loc["S", "quality"] == row.s_quality == 0.75
    assert abs(rcm.loc["S", "time"] - data.to_utc(row.s_time)) < 0.006      # ComCat rounds to 0.01 s


def test_shear_window_gates():
    g = quality.SHEAR_WINDOW_PRE
    assert (g.incidence_max_deg, g.rectilinearity_min, g.snr_min) == (35.0, 0.7, 2.0)
    assert quality.DEFAULT_PRE.incidence_max_deg == 45.0 and quality.DEFAULT_PRE.rectilinearity_min == 0.5


def test_pykonal_reproduces_the_research_pipeline(ray):
    ref = _ref()
    assert ray["travel_time_s"] == pytest.approx(ref.travel_time_s, abs=1e-4)
    assert ray["incidence_deg"] == pytest.approx(ref.incidence_angle, abs=1e-3)


@pytest.mark.slow
def test_rcm_event_reproduces_and_passes(m):
    ref = _ref()
    assert m.stage == "ok"
    for c in ["phi", "dt", "phi_error", "dt_error", "quality", "snr_horizontal", "rectilinearity"]:
        assert getattr(m, c) == pytest.approx(ref[c], abs=1e-9), c
    assert bool(quality.post_filter_mask(pd.DataFrame([m.as_row()])).iloc[0])
    assert 100 * m.dt / ref.travel_time_s == pytest.approx(ref.pct_anisotropy)


@pytest.mark.slow
def test_diagnostic_figure_draws(m):
    fig = plots.diagnostic_figure(m, _row(), depth_km=0.9)
    assert len(fig.axes) >= 9
