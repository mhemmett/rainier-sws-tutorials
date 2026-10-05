"""Loading catalogs, station coordinates and waveforms.

Design rule: every waveform you work with is cached on disk under
``data/waveforms/<STATION>/``. ``get_event_stream`` looks in the cache first and
only asks EarthScope for data that is not there. That makes the notebooks
runnable offline, keeps re-runs fast, and means two people running the same
notebook see the same bytes (replicability).
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from . import paths

# Window used by the research pipeline: [origin - 5 s, origin + 15 s]
PRE_S, POST_S = 5.0, 15.0
FDSN_URL = "https://service.earthscope.org"
TIME_COLUMNS = ["origin_time", "start_time", "p_time", "s_time"]


# ----------------------------------------------------------------------------
# Catalogs
# ----------------------------------------------------------------------------
def load_station_catalog(station: str) -> pd.DataFrame:
    """One row per earthquake with a reviewed P and S pick at ``station``.

    Times come back as timezone-aware pandas Timestamps (UTC).
    """
    cat = pd.read_csv(paths.station_catalog_csv(station))
    for c in TIME_COLUMNS:
        cat[c] = pd.to_datetime(cat[c], utc=True, format="ISO8601")
    return cat


def load_events() -> pd.DataFrame:
    """PNSN event list for the Rainier area (one row per earthquake).

    Gotcha inherited from the research repo: the column named ``datetime`` holds
    a Unix epoch float; ``timestamp`` holds the ISO-8601 string. We add a proper
    ``time`` column so you never have to remember that.
    """
    ev = pd.read_csv(paths.EVENTS_CSV)
    ev["time"] = pd.to_datetime(ev["datetime"], unit="s", utc=True)
    return ev


def load_stations() -> pd.DataFrame:
    """Station coordinates (network, station, lat, lon, elevation in m)."""
    return pd.read_csv(paths.STATIONS_CSV)


def station_coords(station: str) -> tuple[float, float, float]:
    """(lat, lon, elevation_m) for a station code."""
    st = load_stations()
    row = st[st["station"] == station.upper()]
    if row.empty:
        raise KeyError(f"{station} not in {paths.STATIONS_CSV.name}")
    r = row.iloc[0]
    return float(r["lat"]), float(r["lon"]), float(r["elevation"])


def to_utc(ts) -> UTCDateTime:
    """pandas Timestamp (or string) -> ObsPy UTCDateTime, keeping sub-second precision."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return UTCDateTime(ts.timestamp())


# ----------------------------------------------------------------------------
# Waveforms: cache first, EarthScope second
# ----------------------------------------------------------------------------
_CACHE: dict[str, obspy.Stream] = {}


def load_cached_station(station: str, refresh: bool = False) -> obspy.Stream:
    """Read every cached miniSEED file for a station into one Stream (memoized)."""
    key = station.upper()
    if key in _CACHE and not refresh:
        return _CACHE[key]
    st = obspy.Stream()
    for f in sorted(glob.glob(str(paths.station_waveform_dir(key) / "*.mseed"))):
        st += obspy.read(f)
    _CACHE[key] = st
    return st


def _select_window(st: obspy.Stream, t0: UTCDateTime, tol_s: float = 2.0) -> obspy.Stream:
    """Traces whose start time is within ``tol_s`` of ``t0`` (how the cache is keyed)."""
    return obspy.Stream([tr for tr in st if abs(tr.stats.starttime - t0) < tol_s])


def fetch_from_earthscope(network: str, station: str, t0: UTCDateTime, t1: UTCDateTime,
                          channel: str = "?H?", location: str = "*") -> obspy.Stream:
    """Live request to EarthScope's FDSN dataselect service.

    ``?H?`` matches EHZ/EHN/EHE, HHZ/..., BHZ/... (the seismic channels) without
    pulling the dozens of non-seismic channel codes some sites carry.
    Location is ``*`` because STAR's location code is ``01``, not blank.
    """
    client = Client(FDSN_URL)
    return client.get_waveforms(network, station, location, channel, t0, t1)


def get_event_stream(station: str, row: pd.Series, network: str = "UW",
                     allow_download: bool = True, save: bool = True) -> obspy.Stream:
    """Three-component Stream for one catalog row at one station.

    1. Look in the on-disk cache (``data/waveforms/<STATION>/``).
    2. If absent and ``allow_download``, request it from EarthScope and, if
       ``save``, append it to the cache as ``fetched_<evid>.mseed``.

    Returns a Stream with exactly Z, N, E (or raises with a clear message).
    """
    station = station.upper()
    t0 = to_utc(row["start_time"])
    t1 = t0 + PRE_S + POST_S
    st = _select_window(load_cached_station(station), t0)
    source = "cache"
    if len(st) < 3:
        if not allow_download:
            raise FileNotFoundError(f"evid {row['evid']} not in cache for {station} and downloads disabled")
        st = fetch_from_earthscope(network, station, t0, t1)
        source = "earthscope"
        if save and len(st):
            out = paths.station_waveform_dir(station) / f"fetched_{row['evid']}.mseed"
            out.parent.mkdir(parents=True, exist_ok=True)
            st.write(str(out), format="MSEED")
            load_cached_station(station, refresh=True)
    st = st.copy()
    z, n, e = st.select(channel="??Z"), st.select(channel="??N"), st.select(channel="??E")
    if not (len(z) == len(n) == len(e) == 1):
        raise ValueError(f"evid {row['evid']} at {station}: expected 1 trace each of Z/N/E, got "
                         f"Z={len(z)} N={len(n)} E={len(e)} (source={source})")
    out = obspy.Stream([z[0], n[0], e[0]])
    out.source = source  # informal tag so notebooks can say where data came from
    return out


def describe_cache(station: str) -> pd.DataFrame:
    """Small table: which catalog events are in the cache with 3 channels."""
    cat = load_station_catalog(station)
    st = load_cached_station(station)
    rows = []
    for _, r in cat.iterrows():
        n = len(_select_window(st, to_utc(r["start_time"])))
        rows.append({"evid": r["evid"], "n_traces": n, "complete": n == 3})
    return pd.DataFrame(rows)
