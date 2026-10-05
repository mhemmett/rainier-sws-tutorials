"""Shear-wave splitting (SWS) helper functions, ported from axial-sws
(scripts/splitting_functions.py, scripts/mfast_filters_functions.py) for use on
the Mt. Rainier STAR-strict catalog. Only the pieces actually needed by this
project's pipeline are ported (not the full ~9000-line axial-sws
splitting_functions.py, most of which is Axial/OOI-catalog-format-specific
batch plumbing this project doesn't use):

- cov_eig / calculate_rectilinearity_jurkevics / calculate_incidence_angle_eigenvalue_jurkevics:
  Jurkevics (1988) P-wave polarization analysis, verbatim from axial-sws (fully
  generic 3-component math, no OBS-specific assumptions).
- calculate_back_azimuth: axial-sws's version used a custom small-area flat-earth
  projection (`projection.ll2xy`) tuned to the OOI Axial array's footprint. Our
  station network spans >100 km, so this uses obspy's geodesic
  `gps2dist_azimuth` instead -- more accurate at this scale and avoids porting
  an Axial-specific projection helper.
- MFAST_FILTER_BANDS / try_filters / _s_wave_snr_prefiltered: the 14-band MFAST-like
  filter bank and per-event best-SNR-band selection, verbatim from
  mfast_filters_functions.py.
- get_dominant_period_baillard: NOT re-implemented here -- imported directly from
  swspy.splitting.split_windowcheck (this project's local swspy fork), since that
  module already defines it and duplicating it risks the two copies drifting apart.

Per the axial-sws pipeline map: the old P-wave Jurkevics incidence angle
(calculate_incidence_angle_eigenvalue_jurkevics) is used here for BOTH the
<=35deg QC filter AND the LQT rotation inclination -- unlike axial-sws's
production pipeline, which uses a separate S-wave incidence (Jurkevics-S or
PyKonal ray-tracing through an Axial-specific 3D velocity model) for rotation.
Using P-Jurkevics for both avoids needing a Cascades velocity model.
"""
import numpy as np
from obspy.geodetics import gps2dist_azimuth

from swspy.splitting.split_windowcheck import get_dominant_period_baillard  # swspy lives at the repo root

MFAST_FILTER_BANDS = [
    (1.0, 5.0), (1.0, 8.0), (1.0, 15.0), (1.0, 20.0),
    (3.0, 5.0), (2.0, 8.0), (3.0, 15.0), (3.0, 30.0),
    (5.0, 10.0), (5.0, 15.0), (5.0, 30.0), (5.0, 45.0),
    (10.0, 20.0), (10.0, 45.0),
]
MFAST_FILTER_CORNERS = 2  # two-pole Butterworth


def cov_eig(data_array):
    """Covariance-matrix eigen-decomposition, Jurkevics (1988) convention: uses
    np.linalg.eig (not eigh) and sorts descending. data_array shape (n_samples, 3)."""
    cov_mat = np.cov(np.transpose(data_array))
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    ind_descend = np.argsort(-eig_vals)
    return eig_vals[ind_descend], eig_vecs[:, ind_descend]


def calculate_incidence_angle_eigenvalue_jurkevics(trace_z, trace_n, trace_e, p_arrival_offset,
                                                     analysis_window=0.12, p_window_before=0.02):
    """P-wave incidence angle (degrees from vertical) via Jurkevics (1988) polarization
    analysis. Window [P-p_window_before, P-p_window_before+analysis_window]. Assumes Z-up;
    the leading eigenvector is flipped to point downward before taking arccos."""
    try:
        # NOTE: p_arrival_offset is seconds-from-trace-start, but obspy's Trace.slice()
        # requires absolute UTCDateTime bounds -- passing bare floats gets silently
        # (mis)interpreted as POSIX-epoch UTCDateTimes and produces a wrong-length,
        # wrong-location window with no error (verified against obspy's actual slice
        # behavior; this affected axial-sws's original version of this function too).
        # Anchor explicitly to the trace's own start time instead.
        window_start = trace_z.stats.starttime + (p_arrival_offset - p_window_before)
        window_end = window_start + analysis_window

        z_data = trace_z.slice(window_start, window_end).data
        n_data = trace_n.slice(window_start, window_end).data
        e_data = trace_e.slice(window_start, window_end).data

        data_zne = np.column_stack([z_data, n_data, e_data])
        eigvals, eigvecs = cov_eig(data_zne)
        eigvec1 = eigvecs[:, 0]

        if eigvec1[0] >= 0:
            eigvec1 = -eigvec1

        return np.arccos(np.clip(np.abs(eigvec1[0]), 0, 1)) * 180 / np.pi
    except Exception as e:
        print(f'Jurkevics incidence angle calculation error: {e}')
        return np.nan


def calculate_rectilinearity_jurkevics(trace_z, trace_n, trace_e, p_arrival_offset,
                                        analysis_window=0.12, p_window_before=0.02):
    """P-wave rectilinearity via Jurkevics (1988): rec = 1 - (lambda2+lambda3)/(2*lambda1).
    Same window convention as calculate_incidence_angle_eigenvalue_jurkevics."""
    try:
        sampling_rate = trace_z.stats.sampling_rate
        if not (trace_n.stats.sampling_rate == sampling_rate and
                trace_e.stats.sampling_rate == sampling_rate):
            return np.nan

        window_start = trace_z.stats.starttime + (p_arrival_offset - p_window_before)
        window_end = window_start + analysis_window

        z_data = trace_z.slice(window_start, window_end).data
        n_data = trace_n.slice(window_start, window_end).data
        e_data = trace_e.slice(window_start, window_end).data

        data_zne = np.column_stack([z_data, n_data, e_data])
        eigvals, _ = cov_eig(data_zne)
        lambda1, lambda2, lambda3 = eigvals

        return 1 - (lambda2 + lambda3) / (2 * lambda1)
    except Exception as e:
        print(f'Jurkevics rectilinearity calculation error: {e}')
        return np.nan


def calculate_back_azimuth(eq_lat, eq_lon, sta_lat, sta_lon):
    """Back-azimuth (degrees, event-to-station geodesic) using obspy's geodesic
    gps2dist_azimuth -- appropriate at our >100km station-network scale (axial-sws's
    equivalent used a flat-earth projection tuned to the much smaller OOI Axial array)."""
    _, _, back_azimuth = gps2dist_azimuth(eq_lat, eq_lon, sta_lat, sta_lon)
    return back_azimuth


def _check_index(ind, data):
    if ind < 0:
        return 0
    if ind >= len(data):
        return len(data) - 1
    return ind


def _snr_pick(data, ind_center, n_left, n_right, mode='mean'):
    """SNR around a pick based on absolute values (Baillard-style)."""
    abs_data = np.abs(data)
    ind_left = _check_index(ind_center - n_left, abs_data)
    ind_right = _check_index(ind_center + n_right, abs_data)
    if mode == 'mean':
        right = np.mean(abs_data[ind_center:ind_right])
    elif mode == 'max':
        right = np.max(abs_data[ind_center:ind_right])
    else:
        raise ValueError('mode has to be mean or max')
    left = np.mean(abs_data[ind_left:ind_center + 1])
    return right / left


def _stream2data(st):
    return np.column_stack([tr.data for tr in st])


def _s_wave_snr_prefiltered(event_stream, p_time, s_time):
    """Baillard-style windowed S-wave SNR (mean of E/N), assuming event_stream is
    already filtered to the band being evaluated. Returns (snr, dom_period_samples).
    p_time/s_time are absolute UTCDateTime picks."""
    if not (len(event_stream[0].data) == len(event_stream[1].data) == len(event_stream[2].data)):
        return np.nan, np.nan

    s_window = [0.02, 0.3]
    fs_window = [0.1, 0.3]
    s_snr_window = [0.4, 0.2]

    st_x = event_stream.select(channel='??E')
    st_y = event_stream.select(channel='??N')
    if not (len(st_x) == len(st_y) and len(st_x) > 0):
        return np.nan, np.nan
    xy_array = _stream2data(st_x + st_y)

    trace_start_time = event_stream[0].stats.starttime
    sampling_rate = event_stream[0].stats.sampling_rate

    fs_window_time = [s_time - fs_window[0], s_time + fs_window[1]]
    s_window_time = [s_time - s_window[0], s_time + s_window[1]]

    s_samples = int(round((s_time - trace_start_time) * sampling_rate))
    p_samples = int(round((p_time - trace_start_time) * sampling_rate))
    fs_w1 = int(round((fs_window_time[0] - trace_start_time) * sampling_rate))
    fs_w2 = int(round((fs_window_time[1] - trace_start_time) * sampling_rate))
    mid_samples = int(round(p_samples + (s_samples - p_samples) / 2))
    sw1 = int(round((s_window_time[0] - trace_start_time) * sampling_rate))

    if fs_w1 < mid_samples:
        fs_w1 = mid_samples

    xy_array_dom = xy_array[fs_w1:fs_w2, :]
    if xy_array_dom.shape[0] < 2:
        return np.nan, np.nan

    dom_period_x, dom_freq_x = get_dominant_period_baillard(xy_array_dom[:, 0], sampling_rate)
    dom_period_y, dom_freq_y = get_dominant_period_baillard(xy_array_dom[:, 1], sampling_rate)
    if dom_freq_x <= 0 or dom_freq_y <= 0:
        return np.nan, np.nan

    dom_period = np.mean([dom_period_x, dom_period_y])
    if np.isnan(dom_period) or dom_period <= 0:
        return np.nan, np.nan

    sw2 = int(round(sw1 + 2 * dom_period))

    s_snr_window_time = [s_time - s_snr_window[0], s_time + s_snr_window[1]]
    s_snr_w1 = int(round((s_snr_window_time[0] - trace_start_time) * sampling_rate))
    s_snr_w2 = int(round((s_snr_window_time[1] - trace_start_time) * sampling_rate))

    if s_snr_w1 < mid_samples:
        s_snr_w1 = mid_samples
    if s_snr_w2 > sw2:
        s_snr_w2 = sw2
    if s_snr_w1 >= s_samples or s_snr_w2 <= s_samples:
        return np.nan, float(dom_period)

    snr_x = _snr_pick(xy_array[:, 0], s_samples, s_samples - s_snr_w1, s_snr_w2 - s_samples)
    snr_y = _snr_pick(xy_array[:, 1], s_samples, s_samples - s_snr_w1, s_snr_w2 - s_samples)
    return float(np.mean((snr_x, snr_y))), float(dom_period)


def try_filters(raw_stream, p_time, s_time, bands=MFAST_FILTER_BANDS):
    """Try each MFAST-like candidate bandpass on a copy of the raw (unfiltered) event
    stream, score each by S-wave SNR, and return the band that maximizes it.

    Returns (best_band, best_stream, best_snr, best_dom_period_samples, snr_by_band).
    """
    snr_by_band, dom_period_by_band, streams_by_band = {}, {}, {}
    for (freqmin, freqmax) in bands:
        st = raw_stream.copy()
        st.detrend('linear')
        st.taper(max_percentage=0.05, type='hann')
        st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=MFAST_FILTER_CORNERS)
        snr, dom_period = _s_wave_snr_prefiltered(st, p_time, s_time)
        snr_by_band[(freqmin, freqmax)] = snr
        dom_period_by_band[(freqmin, freqmax)] = dom_period
        streams_by_band[(freqmin, freqmax)] = st

    valid = {band: snr for band, snr in snr_by_band.items() if not np.isnan(snr)}
    if not valid:
        return None, None, np.nan, np.nan, snr_by_band

    best_band = max(valid, key=valid.get)
    return best_band, streams_by_band[best_band], valid[best_band], dom_period_by_band[best_band], snr_by_band
