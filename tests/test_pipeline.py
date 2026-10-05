"""Small tests that pin down what the notebooks claim. Run with: pytest -q

They are deliberately few and fast, except the reproduction test (about 15 s),
which is the single most important one: it asserts that this repo's measurement
of event 61504668 at STAR equals the row in Michael's results CSV.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rainier_sws import data, measure, quality, stats, paths  # noqa: E402


def test_axial_mean_wraps():
    m, R = stats.axial_mean([170, 175, 5, 10])
    assert abs(m - 0.0) < 1e-6 or abs(m - 180.0) < 1e-6
    assert R > 0.95


def test_axial_diff():
    assert stats.axial_diff(10, 170) == pytest.approx(20)
    assert stats.axial_diff(0, 90) == pytest.approx(90)


def test_grade3_counts_match_readme():
    r = pd.read_csv(paths.station_results_csv("STAR"))
    f = quality.funnel(r)
    assert (f["attempted"], f["successful"], f["grade3"]) == (903, 709, 121)


def test_grade3_on_station_with_no_measurements():
    r = pd.read_csv(paths.station_results_csv("PANH"))
    assert quality.funnel(r)["grade3"] == 0
    assert quality.stage_counts(r)["fail_incidence"] > 400


def test_cache_returns_three_channels():
    cat = data.load_station_catalog("STAR")
    row = cat[cat.evid == 61504668].iloc[0]
    st = data.get_event_stream("STAR", row, allow_download=False)
    assert sorted(tr.stats.channel[-1] for tr in st) == ["E", "N", "Z"]
    assert st[0].stats.sampling_rate == 100.0


@pytest.mark.slow
def test_reproduce_figure3_event():
    ref = pd.read_csv(paths.station_results_csv("STAR")).set_index("evid").loc[61504668]
    cat = data.load_station_catalog("STAR")
    row = cat[cat.evid == 61504668].iloc[0]
    st = data.get_event_stream("STAR", row, allow_download=False)
    lat, lon, _ = data.station_coords("STAR")
    m = measure.measure_event(row, st, "STAR", lat, lon, incidence_angle=ref.incidence_angle)
    assert m.stage == "ok" and m.success
    assert m.phi == pytest.approx(ref.phi, abs=1e-6)
    assert m.dt == pytest.approx(ref["dt"], abs=1e-9)   # ref.dt would be the pandas .dt accessor
    assert m.phi_error == pytest.approx(ref.phi_error)
    assert m.dt_error == pytest.approx(ref.dt_error)
    assert m.quality == pytest.approx(ref.quality, abs=1e-6)
    assert str(m.filter_band) == ref.filter_band
