"""Tests for notebook 04b's additions: the two-sided clock test and the cached RCM P windows.

Run with: pytest -q tests/test_timing_two_sided.py
"""
import sys
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rainier_sws import paths, timing  # noqa: E402

LAG = np.arange(-6, 6.0001, 0.002)


def _two_sided(clock=0.0, dvv=0.0, t_ab=3.0, f0=3.0):
    """A noise correlation with a packet on each side, as in notebook 04b, section 5."""
    def packet(t0):
        return np.exp(-((LAG - t0) * f0 / 1.2) ** 2) * np.cos(2 * np.pi * f0 * (LAG - t0))
    t = t_ab / (1 + dvv)
    return (packet(t + clock) + 0.6 * packet(-t + clock))[None, :]


@pytest.mark.parametrize("clock, dvv", [(0.03, 0.0), (0.0, -0.01), (0.02, 0.005)])
def test_half_sum_is_the_clock_and_half_difference_the_travel_time(clock, dvv):
    cc = _two_sided(clock, dvv)
    pos = timing.peak_lag(cc, LAG, side="pos")[0]
    neg = timing.peak_lag(cc, LAG, side="neg")[0]
    assert (pos + neg) / 2 == pytest.approx(clock, abs=1e-3)
    assert (pos - neg) / 2 == pytest.approx(3.0 / (1 + dvv), abs=1e-3)


def test_rcm_p_windows_match_their_index():
    base = paths.DATA / "timing" / "rcm_p_windows"
    index = pd.read_csv(base.with_suffix(".csv"), comment="#")
    st = obspy.read(str(base.with_suffix(".mseed")))
    assert len(st) == len(index) > 300 and index.evid.is_unique
    assert {tr.id for tr in st} == {"UW.RCM..HHZ"}
    starts = np.array([tr.stats.starttime.timestamp for tr in st])
    for _, r in index.iterrows():
        p = obspy.UTCDateTime(pd.Timestamp(r.rcm_p_time).timestamp())
        k = np.argmin(np.abs(starts - (p - 2.0).timestamp))
        assert abs(starts[k] - (p - 2.0).timestamp) <= 0.01      # each window starts 2 s before RCM's P pick
        assert st[int(k)].stats.npts >= 400
