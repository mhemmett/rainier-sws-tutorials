"""Loading catalogs, station coordinates and waveforms.

Design rule: every waveform you work with is cached on disk under
``data/waveforms/<STATION>/``. ``get_event_stream`` looks in the cache first and
only asks EarthScope for data that is not there. That makes the notebooks
runnable offline, keeps re-runs fast, and means two people running the same
notebook see the same bytes (replicability).
"""
from __future__ import annotations

import glob
import urllib.request
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

# USGS 3DEP elevation service and the box the repo's DEM covers (lon_min, lat_min, lon_max, lat_max)
DEM_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
DEM_BBOX = (-122.06, 46.65, -121.46, 47.05)
DEM_SIZE = (1080, 720)  # columns, rows: 2 arc-seconds (about 60 m) per pixel


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
# Topography: repo copy first, USGS second
# ----------------------------------------------------------------------------
def download_topography(path: Path = paths.DEM_TIF) -> Path:
    """Request the Rainier DEM (16-bit GeoTIFF, metres) from USGS 3DEP and save it to ``path``."""
    query = (f"bbox={','.join(map(str, DEM_BBOX))}&bboxSR=4326&imageSR=4326"
             f"&size={DEM_SIZE[0]},{DEM_SIZE[1]}&format=tiff&pixelType=S16&compression=LZ77"
             "&interpolation=RSP_BilinearInterpolation&f=image")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{DEM_URL}?{query}", timeout=60) as resp:
        path.write_bytes(resp.read())
    return path


def load_topography(path: Path = paths.DEM_TIF) -> tuple[np.ndarray, list[float]]:
    """Elevation in metres around Rainier, and its extent [lon_min, lon_max, lat_min, lat_max].

    The extent is in the order ``plt.imshow(elev, extent=extent)`` expects. Row 0
    of ``elev`` is the northern edge. Reads the copy in ``data/topography/``,
    downloading it from USGS 3DEP first if it is missing.
    """
    from PIL import Image  # installed with matplotlib
    if not path.exists():
        download_topography(path)
    with Image.open(path) as img:
        elev = np.array(img).astype(float)
        lon0, lat0 = img.tag_v2[33922][3:5]   # GeoTIFF tie point: the top-left corner
        dlon, dlat = img.tag_v2[33550][:2]    # GeoTIFF pixel size in degrees
    nrows, ncols = elev.shape
    return elev, [lon0, lon0 + ncols * dlon, lat0 - nrows * dlat, lat0]


# Counts at which a station's seismic channels saturate, where known (README: STAR's EH channels at about
# 2.0e4, found by S. Han, 2026-10-19). Stations not listed are not checked for clipping.
CLIP_COUNTS = {("UW", "STAR"): 2.0e4}


# ----------------------------------------------------------------------------
# Instrument response: repo copy first, EarthScope second
# ----------------------------------------------------------------------------
# The epoch of the cached waveforms (the 2025 swarm); instruments change between epochs
RESPONSE_EPOCH = (UTCDateTime("2025-07-01"), UTCDateTime("2025-09-01"))


def load_inventory(station: str, network: str = "UW") -> obspy.Inventory:
    """StationXML for a station's seismic channels during the swarm, down to the instrument response.

    Reads ``data/stationxml/<NET>.<STA>.xml`` and asks EarthScope only if it is missing,
    saving what it gets. Mind STAR: the orientations in it are the archived ones (see
    ``ORIENTATION_FIXES``), and its amplitudes are 15 to 18 times below what its response
    implies (README); the shape of the response is fine.
    """
    path = paths.station_xml(station, network)
    if not path.exists():
        inv = Client(FDSN_URL).get_stations(network=network, station=station.upper(), location="*", channel="?H?",
                                            starttime=RESPONSE_EPOCH[0], endtime=RESPONSE_EPOCH[1], level="response")
        path.parent.mkdir(parents=True, exist_ok=True)
        inv.write(str(path), format="STATIONXML")
    return obspy.read_inventory(str(path))


# ----------------------------------------------------------------------------
# Component orientation: archived metadata, corrected where we know it is wrong
# ----------------------------------------------------------------------------
# Orientations that differ from the archived metadata, keyed by (network, station, location, channel):
# (azimuth, dip) the channel really has. Add an entry with a dated note whenever a new one is found.
ORIENTATION_FIXES = {
    # STAR's vertical and north channels are exchanged: the channel labelled EHN records vertical
    # (up-positive) motion and EHZ records north; EHE is east as labelled. So ELZ, the low-gain copy
    # of EHZ, records north too. Found in October 2026 from the P wave of a regional M4.2 and from
    # neighbouring stations (S. Han, "UW.STAR.01 and UW.RCS: channel orientation, amplitude, and
    # clipping", 2026-10-19; the north polarity is the less certain part). EarthScope and IRIS MDA
    # still list EHZ at dip -90 and EHN at azimuth 0. Location 01 is the 2014-onward installation;
    # whether the 2009 installation (location --) was also exchanged is not known.
    ("UW", "STAR", "01", "EHZ"): (0.0, 0.0),
    ("UW", "STAR", "01", "EHN"): (0.0, -90.0),
}
_ORIENTATIONS: pd.DataFrame | None = None


def load_orientations() -> pd.DataFrame:
    """Azimuth and dip of every seismic channel epoch at our stations, as archived (memoized)."""
    global _ORIENTATIONS
    if _ORIENTATIONS is None:
        o = pd.read_csv(paths.ORIENTATIONS_CSV, comment="#", dtype={"location": str}, keep_default_na=False)
        o["start"] = pd.to_datetime(o["start"], utc=True)
        o["end"] = pd.to_datetime(o["end"].replace("", "2599-12-31T00:00:00"), utc=True)
        _ORIENTATIONS = o
    return _ORIENTATIONS


def channel_orientation(tr: obspy.Trace) -> tuple[float, float, str]:
    """(azimuth, dip, source) for one trace: from ORIENTATION_FIXES, else the archived metadata.

    Azimuth is degrees clockwise from north, dip degrees down from horizontal (so an
    upward vertical has dip -90), both in the SEED convention.
    """
    s = tr.stats
    fix = ORIENTATION_FIXES.get((s.network, s.station, s.location, s.channel))
    if fix is not None:
        return fix[0], fix[1], "fix"
    o = load_orientations()
    t = pd.Timestamp(s.starttime.datetime, tz="UTC")
    hit = o[(o.network == s.network) & (o.station == s.station) & (o.location == s.location)
            & (o.channel == s.channel) & (o.start <= t) & (o.end > t)]
    if hit.empty:
        raise KeyError(f"no orientation for {tr.id} at {s.starttime}; add it to {paths.ORIENTATIONS_CSV.name}")
    return float(hit.azimuth.iloc[0]), float(hit.dip.iloc[0]), "metadata"


def _is_nominal(channel: str, azimuth: float, dip: float) -> bool:
    """True when a channel points where its last letter says (Z up, N north, E east)."""
    return {"Z": dip == -90,
            "N": dip == 0 and azimuth % 360 == 0,
            "E": dip == 0 and azimuth % 360 == 90}.get(channel[-1], False)


def orient_zne(st: obspy.Stream) -> obspy.Stream:
    """Three components that really point up, north and east, in that order.

    Each trace's true orientation comes from ``channel_orientation``. If all three
    already point where their names say, the data are returned untouched (integer
    counts, bit for bit). Otherwise each channel is treated as the projection of the
    ground motion (up, north, east) onto its own direction, that 3x3 system is solved
    for the ground motion, the traces are renamed ``??Z``, ``??N``, ``??E``, and the
    correction is noted in ``tr.stats.processing``. A swap or a sign flip comes out
    exact (the matrix is rounded, so cos 90 is 0, not 6e-17).
    """
    if len(st) != 3:
        raise ValueError(f"orient_zne needs 3 traces, got {len(st)}")
    orient = [channel_orientation(tr) for tr in st]
    if all(_is_nominal(tr.stats.channel, az, dip) for tr, (az, dip, _) in zip(st, orient)):
        return obspy.Stream(sorted(st, key=lambda tr: "ZNE".index(tr.stats.channel[-1])))
    st = st.copy()
    st.trim(max(tr.stats.starttime for tr in st), min(tr.stats.endtime for tr in st))
    az, dip = np.deg2rad([o[0] for o in orient]), np.deg2rad([o[1] for o in orient])
    # row i: unit vector of channel i in (up, north, east); dip is positive down
    to_channels = np.round(np.column_stack([-np.sin(dip), np.cos(dip) * np.cos(az), np.cos(dip) * np.sin(az)]), 12)
    une = np.linalg.solve(to_channels, np.vstack([tr.data.astype(float) for tr in st]))
    note = "orient_zne: " + ", ".join(f"{tr.stats.channel} az={az:g} dip={dip:g} ({src})"
                                       for tr, (az, dip, src) in zip(st, orient))
    out = obspy.Stream()
    for comp, values in zip("ZNE", une):
        tr = st[0].copy()
        tr.data = values
        tr.stats.channel = tr.stats.channel[:2] + comp
        tr.stats.processing = list(tr.stats.get("processing", [])) + [note]
        out += tr
    return out


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
                     allow_download: bool = True, save: bool = True,
                     correct_orientation: bool = True) -> obspy.Stream:
    """Three-component Stream for one catalog row at one station.

    1. Look in the on-disk cache (``data/waveforms/<STATION>/``).
    2. If absent and ``allow_download``, request it from EarthScope and, if
       ``save``, append it to the cache as ``fetched_<evid>.mseed``.
    3. Unless ``correct_orientation=False``, rotate the components to true Z, N, E
       with ``orient_zne`` (this is where STAR's exchanged Z and N are fixed).
       Turn it off only to reproduce results computed before the fix.

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
    z, n, e = st.select(channel="??Z"), st.select(channel="??[N1]"), st.select(channel="??[E2]")
    if not (len(z) == len(n) == len(e) == 1):
        raise ValueError(f"evid {row['evid']} at {station}: expected 1 trace each of Z/N/E, got "
                         f"Z={len(z)} N={len(n)} E={len(e)} (source={source})")
    out = obspy.Stream([z[0], n[0], e[0]])
    if correct_orientation:
        out = orient_zne(out)
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
