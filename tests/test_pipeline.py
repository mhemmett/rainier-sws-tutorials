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
from rainier_sws import data, measure, quality, reference, stats, paths  # noqa: E402


def test_axial_mean_wraps():
    m, R = stats.axial_mean([170, 175, 5, 10])
    assert abs(m - 0.0) < 1e-6 or abs(m - 180.0) < 1e-6
    assert R > 0.95


def test_axial_diff():
    assert stats.axial_diff(10, 170) == pytest.approx(20)
    assert stats.axial_diff(0, 90) == pytest.approx(90)


def test_filtered_counts_match_readme():
    r = pd.read_csv(paths.station_results_csv("STAR"))
    f = quality.funnel(r)
    assert (f["attempted"], f["successful"], f["filtered"]) == (903, 709, 110)


def test_oriented_star_counts_match_readme():
    # the research re-run on true Z/N/E with clipped events removed
    r = pd.read_csv(paths.station_results_csv("STAR", oriented=True))
    assert (r.stage == "fail_clipped").sum() == 33
    assert (len(quality.successful(r)), quality.funnel(r)["filtered"]) == (659, 194)


def test_post_filter_on_station_with_no_measurements():
    r = pd.read_csv(paths.station_results_csv("PANH"))
    assert quality.funnel(r)["filtered"] == 0
    assert quality.stage_counts(r)["fail_incidence"] > 400
    f = quality.post_filtered(r)                      # empty, but with the columns a summary needs
    assert len(f) == 0 and {"phi", "dt", "quality"} <= set(f.columns)


def test_topography_covers_the_summit():
    elev, (lon_min, lon_max, lat_min, lat_max) = data.load_topography()
    assert lon_min < -121.76 < lon_max and lat_min < 46.853 < lat_max
    assert 4300 < elev.max() < 4400          # Columbia Crest is 4392 m


def test_star_response_is_short_period_and_shared():
    # notebook 03: a 2 Hz geophone through telemetry filters at 20-30 Hz, the same on every channel
    inv = data.load_inventory("STAR")
    f = np.array([1.0, 10.0, 40.0])
    amps = [np.abs(ch.response.get_evalresp_response_for_frequencies(f, output="VEL")) for ch in inv[0][0]]
    assert len(amps) == 3 and all(np.allclose(a, amps[0]) for a in amps)
    assert amps[0][1] > 4 * amps[0][0] and amps[0][1] > 20 * amps[0][2]


def test_cache_returns_three_channels():
    cat = data.load_station_catalog("STAR")
    row = cat[cat.evid == 61504668].iloc[0]
    st = data.get_event_stream("STAR", row, allow_download=False)
    assert sorted(tr.stats.channel[-1] for tr in st) == ["E", "N", "Z"]
    assert st[0].stats.sampling_rate == 100.0


def _synthetic(station, channels, azimuths, dips, t0="2025-07-15"):
    """Unit ground motion pointing north, recorded on channels with the given orientations."""
    import obspy
    from obspy import UTCDateTime
    st = obspy.Stream()
    for cha, az, dip in zip(channels, azimuths, dips):
        tr = obspy.Trace(np.full(10, np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(dip))))
        tr.stats.update(dict(network="UW", station=station, location="", channel=cha,
                             starttime=UTCDateTime(t0), sampling_rate=100.0))
        st += tr
    return st


def test_star_vertical_and_north_are_exchanged():
    cat = data.load_station_catalog("STAR")
    row = cat[cat.evid == 61504668].iloc[0]
    raw = data.get_event_stream("STAR", row, allow_download=False, correct_orientation=False)
    fixed = data.get_event_stream("STAR", row, allow_download=False)
    assert [tr.stats.channel for tr in fixed] == ["EHZ", "EHN", "EHE"]
    assert np.array_equal(fixed[0].data, raw.select(channel="EHN")[0].data)   # true vertical was labelled N
    assert np.array_equal(fixed[1].data, raw.select(channel="EHZ")[0].data)   # true north was labelled Z
    assert np.array_equal(fixed[2].data, raw.select(channel="EHE")[0].data)   # east is as labelled


def test_nominal_orientation_is_left_alone():
    st = _synthetic("RCM", ["HHZ", "HHN", "HHE"], [0, 0, 90], [-90, 0, 0])
    out = data.orient_zne(st)
    assert all(a.data is b.data for a, b in zip(out, st))   # untouched, not even copied


def test_rotation_from_metadata():
    # RCM 2014-2018: horizontals EH1 at 23 deg and EH2 at 113 deg in the archived metadata
    st = _synthetic("RCM", ["EHZ", "EH1", "EH2"], [0, 23, 113], [-90, 0, 0], t0="2016-07-01")
    st[0].data[:] = 0.0                                   # no vertical motion
    z, n, e = data.orient_zne(st)
    assert (z.stats.channel, n.stats.channel, e.stats.channel) == ("EHZ", "EHN", "EHE")
    assert np.allclose(n.data, 1.0) and np.allclose(e.data, 0.0, atol=1e-12) and np.allclose(z.data, 0.0)


@pytest.mark.slow
def test_reproduce_figure3_event():
    # The results CSVs were computed before STAR's exchanged Z and N were found, so reproducing
    # them needs the data as archived. Notebook 05 shows what the corrected data give.
    ref = pd.read_csv(paths.station_results_csv("STAR")).set_index("evid").loc[61504668]
    cat = data.load_station_catalog("STAR")
    row = cat[cat.evid == 61504668].iloc[0]
    st = data.get_event_stream("STAR", row, allow_download=False, correct_orientation=False)
    lat, lon, _ = data.station_coords("STAR")
    m = measure.measure_event(row, st, "STAR", lat, lon, incidence_angle=ref.incidence_angle)
    assert m.stage == "ok" and m.success
    assert m.phi == pytest.approx(ref.phi, abs=1e-6)
    assert m.dt == pytest.approx(ref["dt"], abs=1e-9)   # ref.dt would be the pandas .dt accessor
    assert m.phi_error == pytest.approx(ref.phi_error)
    assert m.dt_error == pytest.approx(ref.dt_error)
    assert m.quality == pytest.approx(ref.quality, abs=1e-6)
    assert str(m.filter_band) == ref.filter_band
    assert np.isfinite(m.src_pol) and m.clip_fraction < quality.DEFAULT_PRE.clip_fraction_max


def test_dt_step_and_reference_directions():
    assert measure.dt_step(100.0) == pytest.approx(0.02)
    assert 0 <= reference.REGIONAL_COMPRESSION_AZ < 180 and 0 <= reference.STRUCTURE_2009_STRIKE < 180


def test_near_axis_screen():
    assert list(stats.near_axis([0, 30, 85, 100], 0, 20)) == [True, False, True, True]
    assert list(stats.near_axis([0, 30, 85, 100], 0, 20, perpendicular=False)) == [True, False, False, False]
    assert stats.chance_near_axis(20) == pytest.approx(80 / 180)


def test_permutation_and_block_bootstrap():
    rng = np.random.default_rng(1)
    a, b = rng.normal(30, 10, 60), rng.normal(45, 10, 60)
    d, p = stats.permutation_test(a, b, axial=True, n_perm=500)
    assert d == pytest.approx(15, abs=5) and p < 0.01
    _, p_same = stats.permutation_test(a[:30], a[30:], n_perm=500)
    assert p_same > 0.05
    m0, lo, hi, _ = stats.block_bootstrap_axial_mean(a, np.repeat(np.arange(12), 5), n_boot=200)
    assert m0 == pytest.approx(stats.axial_mean(a)[0]) and lo < 0 < hi


@pytest.mark.slow
def test_clip_screen_reproduces_research_rerun():
    # the research re-run on corrected components stopped 33 STAR events at fail_clipped
    r = pd.read_csv(paths.station_results_csv("STAR", oriented=True))
    r = r[r.clip_fraction.notna()]
    cat = data.load_station_catalog("STAR").set_index("evid")
    flagged = set()
    for evid in r.evid:
        row = cat.loc[evid].copy(); row["evid"] = evid
        st = data.get_event_stream("STAR", row, allow_download=False)
        if measure.clip_fraction(st, measure.pick_offsets(st, row)[4]) >= quality.DEFAULT_PRE.clip_fraction_max:
            flagged.add(evid)
    assert flagged == set(r.evid[r.stage == "fail_clipped"])


def test_remeasure_rows():
    r = pd.read_csv(paths.station_results_csv("STAR", oriented=True)).set_index("evid")
    rows = r.loc[[61504668, 62135967]].reset_index()             # the figure-3 event and a clipped one
    out = measure.remeasure(rows, "STAR")
    assert list(out.stage) == ["ok", "fail_clipped"]
    assert out.phi.iloc[0] == pytest.approx(r.loc[61504668, "phi"], abs=1e-6)
