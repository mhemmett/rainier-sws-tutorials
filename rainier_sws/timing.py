"""Clock errors: finding them in a catalog, measuring them with ambient noise, fixing them.

Two independent ways to ask "is this station's clock right?":

1. **From the catalog** (`travel_time_residuals`). A clock error shifts every
   arrival at one station by the same amount, so the P *and* S residuals move
   together in time. Location or velocity errors do not do that (they scale
   with slowness, so S moves about Vp/Vs = 1.7 times more than P). This uses
   only the picks already in the repo and is the first thing to try.

2. **From continuous data** (`ccf_hourly`, `peak_lag`, `delta_t_from_lags`,
   `clean_delta_t`, `clock_model`). Cross-correlate ambient noise between the
   station and a neighbour with trusted (GPS) timing; the lag of the
   correlation peak is constant unless one clock moves. This is the method of
   **chronos** (Maleen Kidiwela, UW; github.com/MaleenKidiwela/chronos), with
   its companion **chronfix** applying the measured delta_t(t) to the raw
   miniSEED. The functions below import chronos / chronfix when they are
   installed (see environment.yml) and fall back to short local copies of the
   same numerics otherwise, so the notebook runs either way; the fallback is
   flagged by ``CHRONOS_AVAILABLE``.

Sign convention (chronos): ``delta_t > 0`` means the station's clock is *late*
relative to UTC, i.e. a sample stamped ``T`` was really recorded at ``T - delta_t``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import hilbert

# ----------------------------------------------------------------------------
# chronos / chronfix: import if present, else local copies of the numerics
# ----------------------------------------------------------------------------
import matplotlib

_backend = matplotlib.get_backend()   # chronos' filter module forces "Agg" at import; undo that
try:  # pragma: no cover - depends on the environment
    from chronos.scripts.compute_ccf import cc_segment, cosine_taper_edges, whiten_segment
    from chronos.scripts.compute_peak_lag import envelope_squared
    from chronos.scripts.filter_and_triggers import hampel_mask
    from chronfix.clock_model import ClockModel
    from chronfix.correct import correct_stream
    CHRONOS_AVAILABLE = True
except ImportError:  # pragma: no cover
    CHRONOS_AVAILABLE = False
finally:
    if matplotlib.get_backend() != _backend:
        matplotlib.use(_backend)
if not CHRONOS_AVAILABLE:  # pragma: no cover

    def cosine_taper_edges(x, fs, taper_s=20.0):
        """chronos.scripts.compute_ccf.cosine_taper_edges (local copy)."""
        n = len(x)
        taper_n = int(round(taper_s * fs))
        if taper_n <= 0 or 2 * taper_n >= n:
            return x
        taper = np.ones(n)
        idx = np.arange(taper_n)
        fade_in = 0.5 * (1.0 - np.cos(np.pi * idx / taper_n))
        taper[:taper_n] = fade_in
        taper[-taper_n:] = fade_in[::-1]
        return x * taper

    def whiten_segment(x, fs, fmin, fmax):
        """chronos.scripts.compute_ccf.whiten_segment (local copy): unit amplitude
        spectrum inside a raised-cosine band, zero outside."""
        n = len(x)
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        amp = np.abs(X)
        Xn = X / np.where(amp > 0, amp, 1.0)
        lo1, lo2 = fmin * 0.5, fmin
        hi1 = fmax
        hi2 = min(fmax * 1.2, fs / 2.0 - 1e-6)
        w = np.ones_like(freqs)
        w[freqs < lo1] = 0.0
        w[freqs > hi2] = 0.0
        m = (freqs >= lo1) & (freqs < lo2)
        if (lo2 - lo1) > 0:
            w[m] = 0.5 * (1.0 - np.cos(np.pi * (freqs[m] - lo1) / (lo2 - lo1)))
        m = (freqs > hi1) & (freqs <= hi2)
        if (hi2 - hi1) > 0:
            w[m] = 0.5 * (1.0 + np.cos(np.pi * (freqs[m] - hi1) / (hi2 - hi1)))
        return np.fft.irfft(Xn * w, n)

    def cc_segment(a, b, fs, maxlag):
        """chronos.scripts.compute_ccf.cc_segment (local copy). Positive lag = b lags a."""
        n = len(a)
        nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
        A = np.fft.rfft(a, n=nfft)
        B = np.fft.rfft(b, n=nfft)
        c = np.fft.irfft(np.conj(A) * B, n=nfft)
        half = int(round(maxlag * fs))
        return np.concatenate([c[-half:], c[: half + 1]])

    def envelope_squared(cc):
        """chronos.scripts.compute_peak_lag.envelope_squared (local copy)."""
        return np.abs(hilbert(np.asarray(cc, dtype=np.float64) ** 2, axis=-1))

    def hampel_mask(x, window, threshold_sigma=4.0, min_abs_deviation=0.5):
        """chronos.scripts.filter_and_triggers.hampel_mask (local copy)."""
        s = pd.Series(x)
        mp = max(5, window // 4)
        med = s.rolling(window=window, center=True, min_periods=mp).median()
        resid = (s - med).abs()
        mad = resid.rolling(window=window, center=True, min_periods=mp).median()
        thr = np.maximum(threshold_sigma * 1.4826 * mad, min_abs_deviation)
        return np.isfinite(x) & (resid.to_numpy() > thr.to_numpy())

    ClockModel = None       # chronfix not installed
    correct_stream = None


# ----------------------------------------------------------------------------
# 1. Catalog diagnostic: travel-time residuals per station
# ----------------------------------------------------------------------------
def hypocentral_distance_km(row: pd.Series, sta: pd.DataFrame) -> float:
    """Straight-line source-receiver distance (km), elevation included.

    ``event_depth`` is km below sea level (positive down), station elevation
    is metres above sea level, so the vertical leg is depth + elevation/1000.
    """
    s = sta.loc[row["station"]]
    horiz = gps2dist_azimuth(s["lat"], s["lon"], row["event_lat"], row["event_lon"])[0] / 1000.0
    vert = row["event_depth"] + s["elevation"] / 1000.0
    return float(np.hypot(horiz, vert))


def travel_time_residuals(cat: pd.DataFrame, sta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """P and S residuals after fitting ``t = R / V + c_station`` across the whole catalog.

    This is the crudest possible travel-time model (one velocity, straight
    rays) plus one constant per station that absorbs site effects, telemetry
    delay and any *constant* clock offset. What is left, ``res_p`` and
    ``res_s``, is the part that varies from event to event. A clock that
    wanders in time shows up as a trend that is the *same* in ``res_p`` and
    ``res_s``.

    Parameters
    ----------
    cat : one row per (event, station) with origin_time, p_time, s_time,
        event_lat/lon/depth (e.g. ``rainier_catalog_strict.csv``).
    sta : station table indexed by station code with lat, lon, elevation.

    Returns
    -------
    (df, fit) where df has R_km, tp, ts, res_p, res_s and fit holds the
    fitted velocities and per-station constants.
    """
    df = cat[cat["station"].isin(sta.index)].copy()
    for c in ("origin_time", "p_time", "s_time"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    df["tp"] = (df["p_time"] - df["origin_time"]).dt.total_seconds()
    df["ts"] = (df["s_time"] - df["origin_time"]).dt.total_seconds()
    df["R_km"] = [hypocentral_distance_km(r, sta) for _, r in df.iterrows()]

    stations = sorted(df["station"].unique())
    X = np.zeros((len(df), 1 + len(stations)))
    X[:, 0] = df["R_km"].values
    for i, s in enumerate(stations):
        X[df["station"].values == s, 1 + i] = 1.0
    fit = {}
    for phase, col in (("p", "tp"), ("s", "ts")):
        beta, *_ = np.linalg.lstsq(X, df[col].values, rcond=None)
        const = pd.Series(beta[1:], index=stations)
        df[f"res_{phase}"] = df[col].values - df["R_km"].values * beta[0] - const.loc[df["station"]].values
        fit[f"v{phase}_km_s"] = 1.0 / beta[0]
        fit[f"const_{phase}"] = const
    return df, fit


def daily_residuals(res: pd.DataFrame, station: str, min_events: int = 5) -> pd.DataFrame:
    """Daily median P and S residual (and N) for one station."""
    d = res[res["station"] == station].copy()
    d["day"] = d["origin_time"].dt.floor("D")
    g = d.groupby("day").agg(res_p=("res_p", "median"), res_s=("res_s", "median"), n=("res_p", "size"))
    return g[g["n"] >= min_events]


# ----------------------------------------------------------------------------
# 2. Ambient-noise cross-correlation (the chronos recipe)
# ----------------------------------------------------------------------------
@dataclass
class CCParams:
    """Parameters of the noise cross-correlation.

    chronos defaults are 30-min windows, 7.5-min step, whitening 0.5-3.8 Hz,
    max lag 60 s, for 8 Hz ocean-bottom data. For the 20-Hz synthetic in
    notebook 05b, and for a 5-km summit pair decimated to 50 Hz, we use
    shorter windows and a higher band; the recipe is the same.
    """
    window_s: float = 600.0       # correlation window length
    step_s: float = 150.0         # step between windows (75 % overlap)
    whiten_fmin: float = 1.0      # raised-cosine whitening band
    whiten_fmax: float = 6.0
    taper_s: float = 10.0         # cosine taper at both window ends
    maxlag_s: float = 10.0        # +/- lag kept
    one_bit: bool = True          # sign() amplitude normalisation


def ccf_windows(a: np.ndarray, b: np.ndarray, fs: float, p: CCParams = CCParams()):
    """Cross-correlate every window of two equal-length, co-sampled traces.

    Returns (cc, t_mid_s, lags): cc has one row per window, t_mid_s is the
    window midpoint in seconds from the start of the arrays, lags in seconds.
    Positive lag = b lags a (chronos convention).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    seg_n, step_n = int(p.window_s * fs), int(p.step_s * fs)
    half = int(round(p.maxlag_s * fs))
    lags = np.arange(-half, half + 1) / fs
    rows, mids = [], []
    i = 0
    while i + seg_n <= n:
        za, zb = a[i:i + seg_n].copy(), b[i:i + seg_n].copy()
        if za.std() == 0 or zb.std() == 0:
            i += step_n
            continue
        idx = np.arange(seg_n)
        for arr in (za, zb):
            arr -= arr.mean()
            slope, inter = np.polyfit(idx, arr, 1)
            arr -= slope * idx + inter
        za = whiten_segment(cosine_taper_edges(za, fs, p.taper_s), fs, p.whiten_fmin, p.whiten_fmax)
        zb = whiten_segment(cosine_taper_edges(zb, fs, p.taper_s), fs, p.whiten_fmin, p.whiten_fmax)
        if p.one_bit:
            za, zb = np.sign(za), np.sign(zb)
        rows.append(cc_segment(za, zb, fs, p.maxlag_s))
        mids.append((i + seg_n / 2.0) / fs)
        i += step_n
    return np.array(rows), np.array(mids), lags


def stack_by_hour(cc: np.ndarray, t_mid_s: np.ndarray, t0: UTCDateTime):
    """Median-stack windows into UTC hours (chronos ``stack_hourly``).

    Returns (cc_hourly, hour_times) with hour_times as datetime64[h].
    """
    abs_h = np.floor((t0.timestamp + t_mid_s) / 3600.0).astype(np.int64)
    hours = np.unique(abs_h)
    cc_h = np.full((len(hours), cc.shape[1]), np.nan)
    for i, h in enumerate(hours):
        cc_h[i] = np.median(cc[abs_h == h], axis=0)
    hour_times = (hours.astype("datetime64[h]") - np.datetime64(0, "h")) + np.datetime64("1970-01-01T00", "h")
    return cc_h, hour_times


def peak_lag(cc_h: np.ndarray, lags: np.ndarray, refine: bool = True, side: str = "global") -> np.ndarray:
    """Lag of the maximum of the envelope of CC^2, per row.

    chronos picks the nearest sample. With ``refine=True`` a parabola through
    the three envelope samples around the maximum gives a sub-sample lag
    (chronos' ``uncertainty.py`` uses the same three-point curvature for its
    error bar but does not move the pick); for a 100-Hz pair that takes the
    resolution from 10 ms to ~1 ms. A noise correlation has peaks at both
    +travel time and -travel time; ``side`` = "pos" or "neg" restricts the
    search to one of them (chronos ``--side``), which stops the pick flipping
    between the two when the noise sources move.
    """
    env = envelope_squared(cc_h)
    out = np.full(cc_h.shape[0], np.nan)
    dl = lags[1] - lags[0]
    sel = {"global": np.ones_like(lags, bool), "pos": lags > 0, "neg": lags < 0}[side]
    for i in range(cc_h.shape[0]):
        e = np.where(sel, env[i], -np.inf)
        if not np.all(np.isfinite(env[i])) or np.all(env[i] == 0):
            continue
        k = int(np.argmax(e))
        e = env[i]
        out[i] = lags[k]
        if refine and 0 < k < len(e) - 1:
            denom = e[k - 1] - 2 * e[k] + e[k + 1]
            if denom < 0:
                out[i] = lags[k] + 0.5 * dl * (e[k - 1] - e[k + 1]) / denom
    return out


def delta_t_from_lags(peak: np.ndarray, anchor: float | None = None, target_is_b: bool = True):
    """Convert a peak-lag track to the clock error of the target station.

    ``anchor`` is the lag the pair *should* have (inter-station travel time).
    chronos takes it as the median of a window known to be good; pass None to
    use the median of the whole track. If the target is station b (positive
    lag = b lags a) then ``delta_t = peak - anchor``; for station a the sign
    flips.
    """
    anchor = float(np.nanmedian(peak)) if anchor is None else float(anchor)
    dt = peak - anchor
    return (dt if target_is_b else -dt), anchor


def clean_delta_t(dt: np.ndarray, window: int = 25, threshold_sigma: float = 3.5,
                  min_abs_deviation: float = 0.03, jump_threshold: float = 0.05):
    """Hampel outlier filter, then flag hours where delta_t jumps (a resync).

    chronos runs three Hampel passes (7 d / 3 d / 1 d) tuned for second-scale
    OBS drift. One pass with a window in hours and thresholds in seconds is
    enough here; expose them so you can see what each does. Returns
    (dt_clean, outlier_mask, trigger_mask) where trigger_mask marks both hours
    bracketing a jump larger than ``jump_threshold`` between retained samples
    (a closed interval, as chronos defines its trigger periods).
    """
    dt = np.asarray(dt, dtype=float)
    mask = hampel_mask(dt, window=window, threshold_sigma=threshold_sigma,
                       min_abs_deviation=min_abs_deviation)
    clean = dt.copy()
    clean[mask] = np.nan
    valid = np.where(np.isfinite(clean))[0]
    trig = np.zeros(len(dt), dtype=bool)
    for k in range(1, len(valid)):
        if abs(clean[valid[k]] - clean[valid[k - 1]]) > jump_threshold:
            trig[valid[k - 1]:valid[k] + 1] = True
    return clean, mask, trig


def smooth_segments(dt_clean: np.ndarray, trig: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred rolling median inside each inter-trigger segment (chronos ``model_segments``, simplified).

    chronos found that feeding the raw hourly staircase to the corrector
    injects sub-sample time warp; smoothing each stable segment fixes that.
    Trigger hours stay NaN so the corrector splits there.
    """
    out = np.full_like(dt_clean, np.nan)
    seg_id = np.cumsum(trig & ~np.roll(trig, 1))
    for s in np.unique(seg_id):
        sel = (seg_id == s) & ~trig
        if sel.sum() == 0:
            continue
        ser = pd.Series(dt_clean[sel])
        sm = ser.rolling(window, center=True, min_periods=1).median()
        sm = sm.interpolate(limit_direction="both")
        out[sel] = sm.values
    return out


def clock_model(hour_times: np.ndarray, dt_model: np.ndarray, trig: np.ndarray, station: str):
    """Build a chronfix ``ClockModel`` from the hourly series (requires chronfix)."""
    if ClockModel is None:
        raise ImportError("chronfix is not installed; see environment.yml")
    dt_model = np.asarray(dt_model, dtype=np.float64).copy()
    hour_times = np.asarray(hour_times, dtype="datetime64[h]")
    one_s = np.timedelta64(1, "s")
    starts, ends = [], []
    i = 0
    while i < len(trig):
        if trig[i]:
            j = i
            while j + 1 < len(trig) and trig[j + 1]:
                j += 1
            starts.append(hour_times[i].astype("datetime64[s]"))
            ends.append((hour_times[j] + np.timedelta64(1, "h")).astype("datetime64[s]") - one_s)
            # Fill the trigger hours from each side so the interpolation at the
            # segment boundaries is exact (chronfix returns NaN inside anyway).
            before = dt_model[:i][np.isfinite(dt_model[:i])]
            after = dt_model[j + 1:][np.isfinite(dt_model[j + 1:])]
            mid = (i + j + 1) // 2
            if len(before):
                dt_model[i:mid + 1] = before[-1]
            if len(after):
                dt_model[mid + 1:j + 1] = after[0]
            i = j + 1
        else:
            i += 1
    # Pad one hour at each end so the first and last hours are covered.
    hour_times = np.concatenate([[hour_times[0] - np.timedelta64(1, "h")], hour_times,
                                 [hour_times[-1] + np.timedelta64(1, "h")]])
    dt_model = np.concatenate([[dt_model[0]], dt_model, [dt_model[-1]]])
    return ClockModel(hour_times=hour_times, delta_t=dt_model,
                      trigger_starts=np.array(starts, dtype="datetime64[s]"),
                      trigger_ends=np.array(ends, dtype="datetime64[s]"), station=station)


# ----------------------------------------------------------------------------
# 3. A synthetic station pair with a known clock error
# ----------------------------------------------------------------------------
def synthetic_pair(hours: float = 72.0, fs: float = 20.0, lag_s: float = 1.2,
                   delta_t_fn=None, snr: float = 0.7, seed: int = 0,
                   t0: UTCDateTime = UTCDateTime(2025, 7, 10)) -> tuple[Stream, np.ndarray]:
    """Two stations recording the same noise field; station B has a wrong clock.

    Station A: common signal + independent noise. Station B: the same common
    signal delayed by ``lag_s`` (the inter-station travel time), *re-stamped*
    with the clock error ``delta_t_fn(t_seconds)`` (seconds, positive = clock
    late), plus independent noise. ``snr`` is the ratio of common-signal RMS to
    independent-noise RMS, so 0.7 means the coherent part is weaker than the
    noise in any one window: that is why we stack.

    Returns (Stream[A, B], true_delta_t_hourly) so you can grade the recovery.
    """
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * fs)
    t = np.arange(n) / fs
    if delta_t_fn is None:
        def delta_t_fn(ts):
            return 0.03 * np.sin(2 * np.pi * ts / (hours * 3600))
    # common field: band-limited noise (1-6 Hz) with a little extra margin for the shift
    extra = int(5 * fs)
    common = rng.standard_normal(n + 2 * extra)
    common = _bandpass(common, fs, 1.0, 6.0)
    common /= common.std()
    a = common[extra:extra + n] + rng.standard_normal(n) / snr
    # B records common(t - lag). Its clock stamps true time tau as tau + dt(tau);
    # a sample stamped at t was recorded at true time t - dt(t).
    true_time = t - delta_t_fn(t)
    tc = np.arange(n + 2 * extra) / fs - extra / fs
    b = np.interp(true_time - lag_s, tc, common) + rng.standard_normal(n) / snr
    st = Stream([Trace(a.astype(np.float32)), Trace(b.astype(np.float32))])
    for tr, name in zip(st, ("REFA", "STAB")):
        tr.stats.update({"network": "XX", "station": name, "channel": "HHZ",
                         "sampling_rate": fs, "starttime": t0})
    hour_t = np.arange(int(hours)) * 3600.0 + 1800.0
    return st, delta_t_fn(hour_t)


def _bandpass(x, fs, fmin, fmax, corners=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(corners, [fmin, fmax], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def correct_with_model(st: Stream, model, method: str = "resample") -> Stream:
    """Apply a chronfix ClockModel to a Stream (per-sample resampling onto true UTC).

    One wrinkle worth knowing about (as of chronos commit 8dad676): chronfix
    treats a trigger interval as *closed*, and a stable segment ends exactly
    where the trigger starts, so ``interp_delta_t`` at the segment end is NaN
    and chronfix silently drops the whole segment. We sidestep that by cutting
    each stable segment one sample short of the trigger on both sides before
    handing it to chronfix; because chronfix truncates times to whole seconds
    internally, "one sample" has to be a full second. (A good first
    open-source contribution: report this upstream with a two-line test.)
    """
    if correct_stream is None:
        raise ImportError("chronfix is not installed; see environment.yml")
    from chronfix.correct import correct_trace
    out = Stream()
    for tr in st:
        eps = 1.0
        for s0, s1 in model.stable_intervals(tr.stats.starttime.datetime, tr.stats.endtime.datetime):
            t0 = max(UTCDateTime(str(s0.astype("datetime64[us]"))) + eps, tr.stats.starttime)
            t1 = min(UTCDateTime(str(s1.astype("datetime64[us]"))) - eps, tr.stats.endtime)
            if t1 <= t0:
                continue
            sub = tr.slice(t0, t1, nearest_sample=False)
            out += Stream(correct_trace(sub, model, method=method))
    return out


# ----------------------------------------------------------------------------
# 4. Real data: continuous waveforms for a station pair (needs EarthScope)
# ----------------------------------------------------------------------------
def fetch_continuous(network: str, station: str, t0: UTCDateTime, t1: UTCDateTime,
                     channel: str = "?HZ", location: str = "*", target_fs: float = 50.0,
                     fdsn_url: str = "https://service.earthscope.org") -> Trace:
    """One merged, decimated vertical trace for [t0, t1) from EarthScope.

    Downsampling keeps the correlation cheap; keep ``target_fs`` at least
    twice the top of the whitening band. Gaps are zero-filled (chronos also
    cosine-tapers 100 s on each side of a gap; a window that is all zeros is
    skipped by ``ccf_windows`` either way). If the request matches several
    channel or location codes, the longest trace is kept and the choice printed.
    """
    from obspy.clients.fdsn import Client
    st = Client(fdsn_url).get_waveforms(network, station, location, channel, t0, t1)
    st.merge(method=1, fill_value=0.0)
    if len(st) > 1:
        st.traces.sort(key=lambda t: t.stats.npts, reverse=True)
        print(f"{station}: {len(st)} traces matched; keeping {st[0].id}")
    tr = st[0]
    tr.data = tr.data.astype(np.float64)
    tr.detrend("demean").detrend("linear")
    if tr.stats.sampling_rate > target_fs:
        tr.filter("lowpass", freq=0.4 * target_fs, corners=4, zerophase=True)
        tr.resample(target_fs)
    tr.trim(t0, t1, pad=True, fill_value=0.0, nearest_sample=True)
    return tr


def align_pair(tra: Trace, trb: Trace) -> tuple[np.ndarray, np.ndarray, float, UTCDateTime]:
    """Cut two traces to their common span and equal length (sample 0 = same UTC time)."""
    if abs(tra.stats.sampling_rate - trb.stats.sampling_rate) > 1e-6:
        raise ValueError("sampling rates differ; decimate both to the same rate first")
    fs = float(tra.stats.sampling_rate)
    # Common grid starting on a whole second. A corrected trace starts at an
    # arbitrary fraction of a second, so "trim to the same time" is not enough:
    # if sample 0 of a and sample 0 of b are not the *same* instant, the whole
    # correlation is biased by that offset. (chronos hit exactly this bug when
    # validating its own correction; see its Correction Method doc, section 10.)
    t0 = max(tra.stats.starttime, trb.stats.starttime)
    t0 = UTCDateTime(np.ceil(t0.timestamp))
    t1 = min(tra.stats.endtime, trb.stats.endtime)
    n = int(np.floor((t1 - t0) * fs)) + 1
    out = []
    for tr in (tra, trb):
        c = tr.copy()
        if abs((c.stats.starttime - t0) * fs - round((c.stats.starttime - t0) * fs)) < 1e-6:
            c.trim(t0, t0 + (n - 1) / fs, nearest_sample=True)     # already on the grid
        else:
            c.interpolate(sampling_rate=fs, starttime=t0, npts=n, method="linear")
        out.append(c.data[:n])
    return out[0], out[1], fs, t0


def run_pair(a: np.ndarray, b: np.ndarray, fs: float, t0: UTCDateTime,
             params: CCParams = CCParams(), refine: bool = True) -> pd.DataFrame:
    """Windows -> hourly stacks -> peak lag, in one call. Returns a tidy table."""
    cc, mids, lags = ccf_windows(a, b, fs, params)
    cc_h, hours = stack_by_hour(cc, mids, t0)
    pk = peak_lag(cc_h, lags, refine=refine)
    df = pd.DataFrame({"hour": hours, "peak_lag_s": pk,
                       "n_windows": [int(np.sum(np.floor((t0.timestamp + mids) / 3600) == h))
                                     for h in ((hours - np.datetime64("1970-01-01T00", "h")).astype(np.int64))]})
    df.attrs["lags"] = lags
    df.attrs["cc_hourly"] = cc_h
    return df
