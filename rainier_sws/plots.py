"""Figures shared by the notebooks: the splitting diagnostic and the shaded-relief map.

``diagnostic_figure`` draws one SWSPy measurement in the layout of the axial-sws methods figures
(``axial-sws/scripts/figure_axec2_mfast_event_80_19957_clean_split_diagnostics.py``, October 2026):

    A  the filtered Z, N, E traces with the P and S picks and the window starts and ends
    B  the phi-dt misfit surface of the best window (normalised eigenvalue ratio lambda2 / lambda1),
       the cycle-skip limit T_dom / 2 and the chosen solution with its error bars
    C  the particle motion across the ray (Q against T) as recorded
    D  the same after the split is removed
    E  the answers of the windows DBSCAN grouped into clusters, and the chosen one
    F  phi window by window in the winning cluster
    G  dt window by window in the winning cluster

phi in B, E and F is measured from Q (the ray frame), as SWSPy searches it; phi from north is phi from Q
plus the back azimuth. The helpers below it return the pieces the figure is made of, so a notebook can
draw one of them on its own.

``splitting_map_figure`` puts one measurement on the map in the style of the project's summary figure
(``mt-rainier/scripts/plot_fast_directions_and_anisotropy_4period_panel_v5.py``): the fast direction as a
red rose on the station, and the percent anisotropy as a blue wedge along the back azimuth.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import obspy
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D

from swspy.splitting import split_windowcheck as swc

from . import data

SUMMIT = (46.8529, -121.7604)                                  # lat, lon of the summit (notebook 02)
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = KM_PER_DEG_LAT * np.cos(np.deg2rad(SUMMIT[0]))


# ----------------------------------------------------------------------------
# Pieces of a measurement
# ----------------------------------------------------------------------------
def _so(m):
    so = m.splitting_object
    if so is None:
        raise ValueError(f"evid {m.evid}: no splitting object (stage {m.stage}); only measured events can be drawn")
    return so


def window_times(m) -> tuple[np.ndarray, np.ndarray]:
    """Start and end times of SWSPy's analysis windows, in s after the S pick (7 starts, 7 ends)."""
    so = _so(m)
    w = so.event_station_win_idxs[m.station]
    t_cut = -so.overall_win_start_pre_fast_S_pick                 # the cut SWSPy measures in starts here
    return t_cut + np.asarray(w["win_start_idxs"]) / so.fs, t_cut + np.asarray(w["win_end_idxs"]) / so.fs


def qt_before_after(m, trail: bool = False) -> tuple[obspy.Stream, obspy.Stream]:
    """The measured S wave in the ray frame (L, Q, T), as recorded and with the split removed.

    Both come from SWSPy's own correction (``remove_splitting``, with the measurement's phi and dt),
    rotated with SWSPy's own ZNE -> LQT. ``trail=False`` keeps only the S pick minus the first window
    start to the earliest window end, which is what the axial-sws figures draw.
    """
    so = _so(m)
    i = so.stations_in.index(m.station)
    baz, inc = so.back_azis_all_stations[i], so.receiver_inc_angles_all_stations[i]
    s_pick = so.S_phase_arrival_times[i]
    uncorr, corr = so._get_uncorr_and_corr_waveforms(m.station)
    before = swc._rotate_ZNE_to_LQT(uncorr.select(channel="??[ZNE]"), baz, inc)
    after = swc._rotate_ZNE_to_LQT(corr.select(channel="??[ZNE]"), baz, inc)
    if not trail:
        t0, t1 = s_pick - so.overall_win_start_pre_fast_S_pick, s_pick + so.overall_win_start_post_fast_S_pick
        before, after = before.trim(t0, t1), after.trim(t0, t1)
    return before, after


def fast_slow(m) -> tuple[obspy.Trace, obspy.Trace, obspy.Trace, obspy.Trace]:
    """(fast, slow) as recorded and (fast, slow) re-aligned, from SWSPy's correction.

    The re-aligned pair is shifted the way ``remove_splitting`` does it: fast later by dt/2, slow
    earlier by dt/2, each rounded down to whole samples.
    """
    so = _so(m)
    _, corr = so._get_uncorr_and_corr_waveforms(m.station)
    fast, slow = corr.select(channel="??F")[0], corr.select(channel="??S")[0]
    half = int((m.dt / 2) * fast.stats.sampling_rate)
    fast_a, slow_a = fast.copy(), slow.copy()
    fast_a.data, slow_a.data = np.roll(fast.data, half), np.roll(slow.data, -half)
    return fast, slow, fast_a, slow_a


def phi_dt_surface(m, best_window: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dt values, phi-from-Q values, lambda2/lambda1 grid of shape (n_dt, n_phi)) of one measurement.

    ``best_window=True`` gives the surface of the window SWSPy's answer comes from (panel B);
    ``False`` the average over all 49 windows.
    """
    so = _so(m)
    grids = getattr(so, "phi_dt_grid_best_window", {}) if best_window else {}
    Z = grids.get(m.station, so.phi_dt_grid_average[m.station])
    lags = so.lags_labels if len(so.lags_labels) == Z.shape[0] else np.arange(Z.shape[0]) / so.fs
    return np.asarray(lags), np.asarray(so.phis_labels), Z


def best_window(m) -> dict:
    """The analysis window SWSPy's answer comes from: the member of the winning cluster with the smallest
    phi_err**2 + dt_err**2 (``_sws_win_clustering``). Returns its number (0 to 48), its start and end as
    sample indices into the cut ``qt_before_after(m, trail=True)`` returns, and as s after the S pick."""
    so, wc = _so(m), winning_cluster(m)
    k = int(wc["sample_idxs"][np.argmin(wc["lag_errs"] ** 2 + wc["phi_errs"] ** 2)])
    w = so.event_station_win_idxs[m.station]
    i0, i1 = int(w["win_start_idxs"][k // so.n_win]), int(w["win_end_idxs"][k % so.n_win])
    t_cut = -so.overall_win_start_pre_fast_S_pick
    return {"number": k, "i0": i0, "i1": i1, "t0": t_cut + i0 / so.fs, "t1": t_cut + i1 / so.fs}


def winning_cluster(m) -> dict:
    """phis, phi_errs, lags, lag_errs and window numbers of the cluster SWSPy's answer comes from."""
    info = _so(m).clustering_info[m.station]
    c = info["clusters_dict"][str(info["min_var_idx"])]
    order = np.argsort(c["sample_idxs"])
    return {k: np.asarray(c[k])[order] for k in ("sample_idxs", "phis", "phi_errs", "lags", "lag_errs")}


def solution(m) -> dict:
    """SWSPy's answer in its own frame: phi from Q, dt and their errors, Q_w, T_dom."""
    so = _so(m)
    r = so.sws_result_df.iloc[0]
    return {"phi_from_Q": float(r["phi_from_Q"]), "dt": float(r["dt"]), "phi_err": float(r["phi_err"]),
            "dt_err": float(r["dt_err"]), "Q_w": float(r["Q_w"]), "T_dom": float(so.Tmid)}


# ----------------------------------------------------------------------------
# The diagnostic figure
# ----------------------------------------------------------------------------
STAR_MARKER = (6, 2, 0)                                         # the six-armed asterisk of the axial-sws figures


def diagnostic_figure(m, row, title: str | None = None, depth_km: float | None = None,
                      window_s: tuple[float, float] = (-1.0, 1.5)) -> plt.Figure:
    """The A-G summary page of one measurement, in the axial-sws methods style.

    ``m`` is a ``measure.Measurement`` that reached SWSPy (``m.splitting_object`` set); ``row`` its
    catalog row (origin, P and S times). ``window_s`` is the stretch of panel A around the picks: from
    ``window_s[0]`` s before P to ``window_s[1]`` s after S (the cached traces are 20 s long).
    """
    so, sol = _so(m), solution(m)
    origin, p_time, s_time = data.to_utc(row["origin_time"]), data.to_utc(row["p_time"]), data.to_utc(row["s_time"])
    st_zne = m.stream_filtered
    before, after = qt_before_after(m)
    w_start, w_end = window_times(m)

    fig = plt.figure(figsize=(15, 13))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.22)
    right = outer[1].subgridspec(3, 1, height_ratios=[1.0, 1.0, 0.85], hspace=0.35)
    left_bb, row0, row1, row2 = (outer[0].get_position(fig), right[0].get_position(fig),
                                 right[1].get_position(fig), right[2].get_position(fig))

    # A: Z, N, E with picks and windows
    zne_gs = fig.add_gridspec(3, 1, left=left_bb.x0, right=left_bb.x1, top=row0.y1, bottom=row0.y0, hspace=0.12)
    ax_Z = fig.add_subplot(zne_gs[0])
    zne_axes = [ax_Z, fig.add_subplot(zne_gs[1], sharex=ax_Z), fig.add_subplot(zne_gs[2], sharex=ax_Z)]
    for k, (ax, comp) in enumerate(zip(zne_axes, "ZNE")):
        tr = st_zne.select(channel=f"??{comp}")[0]
        t = tr.times(reftime=origin)
        keep = (t >= (p_time - origin) + window_s[0]) & (t <= (s_time - origin) + window_s[1])
        ax.plot(t[keep], tr.data[keep], c="k", lw=1.0)
        ax.axvline(p_time - origin, c="tab:blue", ls="--", lw=1.2, label="P-pick")
        ax.axvline(s_time - origin, c="tab:red", ls="--", lw=1.2, label="S-pick")
        for j, ws in enumerate(w_start):
            ax.axvline((s_time - origin) + ws, c="0.6", lw=0.6, alpha=0.6, label="Window start" if j == 0 else None)
        for j, we in enumerate(w_end):
            ax.axvline((s_time - origin) + we, c="0.6", lw=0.6, alpha=0.6, ls="--", label="Window end" if j == 0 else None)
        ax.set_ylabel(f"{tr.stats.channel}\n(counts)", fontsize=9)
        ax.tick_params(labelsize=8)
        if k < 2:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel("Time after origin (s)", fontsize=9)
    zne_axes[0].legend(loc="upper right", fontsize=6)
    band = f", filtered {m.filter_band[0]:g}-{m.filter_band[1]:g} Hz" if m.filter_band else ""
    ax_Z.set_title(f"ZNE S-wave Analysis Window{band}", fontsize=11)

    # C, D: particle motion across the ray, before and after
    fig_w, fig_h = fig.get_size_inches()
    sq_h = row1.height
    sq_w = sq_h * fig_h / fig_w
    ax_C = fig.add_axes([left_bb.x0, row1.y0, sq_w, sq_h])
    ax_D = fig.add_axes([left_bb.x0 + sq_w + 0.03, row1.y0, sq_w, sq_h])
    for ax, st, color, ttl, ylab in [(ax_C, before, "darkred", "Original particle motion", True),
                                     (ax_D, after, "navy", "Corrected particle motion", False)]:
        q, t_ = st.select(channel="??Q")[0].data, st.select(channel="??T")[0].data
        a = 1.1 * max(np.abs(q).max(), np.abs(t_).max())
        ax.plot(t_, q, c=color, lw=2.0)
        ax.set(xlim=(-a, a), ylim=(-a, a), aspect="equal")
        ax.set_xlabel("T amplitude (counts)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        if ylab:
            ax.set_ylabel("Q amplitude (counts)", fontsize=9)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_title(ttl, fontsize=13)

    # B: the misfit surface of the best window
    ax_B = fig.add_subplot(right[0])
    lags, phis, Z = phi_dt_surface(m)
    Y, X = np.meshgrid(phis, lags)
    cs = ax_B.contourf(X, Y, Z, levels=20, cmap="Blues")
    ax_B.contour(cs, levels=cs.levels[1::2], colors="k", alpha=0.25, linewidths=0.5)
    ax_B.axvline(sol["T_dom"] / 2, c="k", ls="--", lw=1.5, label=r"$T_{dom}/2$")
    ax_B.errorbar(sol["dt"], sol["phi_from_Q"], xerr=sol["dt_err"], yerr=sol["phi_err"], c="r", capsize=5,
                  marker=STAR_MARKER, markersize=14, markeredgecolor="r", markeredgewidth=2, label="Optimal solution")
    ax_B.legend(loc="upper right", fontsize=6)
    pos = ax_B.get_position()
    fig.colorbar(cs, cax=fig.add_axes([pos.x1 + 0.015, pos.y0, 0.015, pos.height]), label="Norm. eigenvalue ratio")
    ax_B.set_xlabel(r"$\delta t$ (s)", fontsize=9)
    ax_B.set_ylabel(r"$\phi$ from Q ($^o$)", fontsize=9)
    ax_B.tick_params(labelsize=8)
    ax_B.set_title(r"$\phi-\delta t$ Parameter Space: Best Solution", fontsize=11)

    # E: DBSCAN clusters
    ax_E = fig.add_subplot(right[1])
    clusters = so.clustering_info[m.station]["clusters_dict"]
    for k, c in enumerate(clusters.values()):
        ax_E.errorbar(c["lags"], c["phis"], xerr=c["lag_errs"], yerr=c["phi_errs"], fmt="o", markersize=2, color="k",
                      ecolor="k", elinewidth=0.5, capsize=2, capthick=0.5, zorder=5,
                      label="Cluster solutions" if k == 0 else None)
    ax_E.errorbar(sol["dt"], sol["phi_from_Q"], xerr=sol["dt_err"], yerr=sol["phi_err"], marker=STAR_MARKER,
                  linestyle="none", markersize=14, color="r", markeredgecolor="r", markeredgewidth=2, ecolor="r",
                  elinewidth=0.7, capsize=4, capthick=0.7, label="Optimal (best window)", zorder=10)
    ax_E.set(xlim=(0, np.max(so.lags_labels)), ylim=(-90, 90))
    ax_E.set_xlabel(r"$\delta t$ (s)", fontsize=9)
    ax_E.set_ylabel(r"$\phi$ from Q ($^o$)", fontsize=9)
    ax_E.tick_params(labelsize=8)
    ax_E.legend(loc="best", fontsize=6)
    ax_E.grid(True, alpha=0.3)
    ax_E.set_title(r"DBSCAN Clustering in $\phi-\delta t$ Parameter Space", fontsize=11)

    # F, G: the winning cluster, window by window, across the full width
    wc = winning_cluster(m)
    gap = 0.09
    w = (row2.x1 - left_bb.x0 - gap) / 2
    h = row2.height * 0.95
    ax_F = fig.add_axes([left_bb.x0, row2.y0, w, h])
    ax_G = fig.add_axes([left_bb.x0 + w + gap, row2.y0, w, h])
    for ax, y, yerr, ylab, name in [(ax_F, wc["phis"], wc["phi_errs"], r"$\phi$ from Q ($^o$)", r"$\phi$"),
                                    (ax_G, wc["lags"], wc["lag_errs"], r"$\delta t$ (s)", r"$\delta t$")]:
        ax.errorbar(wc["sample_idxs"], y, yerr=yerr, fmt="o", ms=4, color="k", ecolor="k", elinewidth=0.8, capsize=3, zorder=5)
        ax.set_xlabel("Window number", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(f"Cluster stability: {name} per window ({len(wc['sample_idxs'])} windows)", fontsize=10)
        ax.grid(alpha=0.3)

    for ax, top, label in [(ax_Z, row0.y1, "A"), (ax_B, row0.y1, "B"), (ax_C, row1.y1, "C"), (ax_D, row1.y1, "D"),
                           (ax_E, row1.y1, "E"), (ax_F, row2.y1, "F"), (ax_G, row2.y1, "G")]:
        fig.text(ax.get_position().x0 - 0.02, top + 0.02, label, fontsize=15, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(title or f"{m.station}: event {m.evid}", fontsize=15, fontweight="bold")
    depth = f"     Depth: {depth_km:.2f} km" if depth_km is not None else ""
    fig.text(0.5, 0.06, f"Rectilinearity: {m.rectilinearity:.3f}     SNR: {m.snr_horizontal:.2f}     "
                        f"Quality (Q_w): {sol['Q_w']:.3f}     Incidence angle: {m.incidence_angle:.2f}°{depth}",
             fontsize=10, fontweight="bold", ha="center", va="bottom")
    fig.text(0.5, 0.04, f"phi = {sol['phi_from_Q']:.0f}° from Q + back azimuth {m.back_azimuth:.1f}° = "
                        f"{np.mod(m.phi, 180):.1f}° from north     dt = {m.dt:.2f} ± {m.dt_error:.2f} s     "
                        f"phi error {m.phi_error:.0f}°     T_dom = {sol['T_dom']:.3f} s",
             fontsize=10, ha="center", va="bottom")
    return fig


# ----------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------
def shaded_relief():
    """(RGB shaded relief, extent) of the repo's DEM, lit as in notebooks 02 and 06 (from the NW, 45 deg up)."""
    elev, extent = data.load_topography()
    relief = LightSource(azdeg=315, altdeg=45).shade(
        elev, cmap=plt.cm.gray, vmin=-2000, vmax=5000, blend_mode="soft", vert_exag=2,
        dx=(extent[1] - extent[0]) / elev.shape[1] * KM_PER_DEG_LON * 1e3,
        dy=(extent[3] - extent[2]) / elev.shape[0] * KM_PER_DEG_LAT * 1e3)
    return relief, extent


MAP_LON_RANGE = (SUMMIT[1] - 0.065, SUMMIT[1] + 0.065)          # the size of the summary figure's panels,
MAP_LAT_RANGE = (SUMMIT[0] - 0.0375, SUMMIT[0] + 0.0375)        # centred on the summit
SUMMIT_LABEL_LAT = 46.87                                        # the summit's label sits here, clear of the swarm
ROSE_BIN_DEG = 10.0                                             # width of a rose petal and of a back-azimuth wedge
ROSE_FRAC_OF_PANEL = 0.48                                       # width of a rose, as a fraction of its map panel
PCT_ANISOTROPY_RANGE = (1.0, 4.0)                               # colour scale, about the span of the summary figure


def _bare_polar(fig, ax, lon, lat, frac=ROSE_FRAC_OF_PANEL):
    """A transparent polar axes centred on (lon, lat) of map ``ax``: north up, clockwise, nothing drawn."""
    pos = ax.get_position()
    x, y = fig.transFigure.inverted().transform(ax.transData.transform((lon, lat)))
    w, h = frac * pos.width, frac * pos.height
    rose = fig.add_axes([x - w / 2, y - h / 2, w, h], projection="polar")
    rose.set_theta_zero_location("N"); rose.set_theta_direction(-1)
    rose.patch.set_alpha(0.0); rose.spines["polar"].set_visible(False)
    rose.set_xticks([]); rose.set_yticks([]); rose.grid(False)
    rose.set_ylim(0, 1.15)
    return rose


def splitting_map_figure(m, pct_anisotropy: float, station_lat: float, station_lon: float, events=None,
                         event_lat: float | None = None, event_lon: float | None = None,
                         title: str | None = None, pct_range: tuple[float, float] = PCT_ANISOTROPY_RANGE):
    """One measurement on the map, in two panels, in the style of the project's summary figure.

    Left: the fast direction as a red rose centred on the station (one petal each way, ``ROSE_BIN_DEG`` wide,
    because a fast direction is an axis). Right: a wedge from the station along the back azimuth, coloured by
    ``pct_anisotropy``. The station's triangle is drawn on top in both. ``events`` (columns ``lon``, ``lat``)
    are drawn as dots, and the measured earthquake (``event_lat``, ``event_lon``) as a white star.
    """
    (lon0, lon1), (lat0, lat1) = MAP_LON_RANGE, MAP_LAT_RANGE
    elev, full = data.load_topography()                         # cropped to the map, then lit as in the summary figure
    cols = slice(*np.round((np.array([lon0, lon1]) - full[0]) / (full[1] - full[0]) * elev.shape[1]).astype(int))
    rows = slice(*np.round((full[3] - np.array([lat1, lat0])) / (full[3] - full[2]) * elev.shape[0]).astype(int))
    relief = LightSource(azdeg=315, altdeg=45).shade(elev[rows, cols], cmap=plt.cm.gray, vert_exag=1.5, blend_mode="overlay")
    extent = [lon0, lon1, lat0, lat1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.11, top=0.84, wspace=0.06)
    for ax in axes:
        ax.imshow(relief, extent=extent, aspect="auto", alpha=0.45, zorder=0)
        if events is not None:
            ax.scatter(events.lon, events.lat, s=4, c="black", alpha=0.5, edgecolors="none", zorder=5)
        if event_lat is not None and event_lon is not None:
            ax.plot(event_lon, event_lat, "*", ms=11, mfc="white", mec="k", mew=0.8, zorder=27)
        ax.plot(SUMMIT[1], SUMMIT[0], "^", ms=13, mfc="#C04000", mec="k", mew=0.8, zorder=25)
        ax.text(SUMMIT[1], SUMMIT_LABEL_LAT, "Mt. Rainier", ha="center", va="center", fontsize=12, fontweight="bold",
                zorder=26)
        ax.annotate(f"{m.station}\nN=1", (station_lon, station_lat), xytext=(34, 14), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=30)
        ax.set(xlim=(lon0, lon1), ylim=(lat0, lat1), aspect=1 / np.cos(np.deg2rad(SUMMIT[0])))
        ax.set_xlabel("Longitude (°)", fontsize=13, fontweight="bold")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.grid(False)
    axes[0].set_ylabel("Latitude (°)", fontsize=13, fontweight="bold")
    key = [Line2D([], [], ls="", marker="^", ms=11, mfc="#FFD700", mec="k", mew=0.8, label="Station"),
           Line2D([], [], ls="", marker="^", ms=10, mfc="#C04000", mec="k", mew=0.8, label="Volcano")]
    if event_lat is not None and event_lon is not None:
        key.append(Line2D([], [], ls="", marker="*", ms=11, mfc="white", mec="k", mew=0.8, label="This earthquake"))
    if events is not None:
        key.append(Line2D([], [], ls="", marker="o", ms=3, mfc="black", mec="none", label="All earthquakes"))
    axes[0].legend(handles=key, loc="upper left", fontsize=10, framealpha=0.85)
    axes[0].set_title("Fast Direction - Fracture Orientation", fontsize=14, fontweight="bold")
    axes[1].set_title("Percent Anisotropy - Fracture Density by Back Azimuth", fontsize=14, fontweight="bold")
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.canvas.draw()                                           # fixes the panels' positions for the roses

    width = np.deg2rad(ROSE_BIN_DEG)
    centre = lambda az: np.deg2rad((np.floor(np.mod(az, 360) / ROSE_BIN_DEG) + 0.5) * ROSE_BIN_DEG)   # its bin's centre
    rose = _bare_polar(fig, axes[0], station_lon, station_lat)
    rose.bar([centre(m.phi), centre(m.phi + 180)], [1, 1], width=width, color="red", edgecolor="black", lw=0.4, alpha=0.9)

    norm, cmap = Normalize(*pct_range), plt.get_cmap("Blues")
    wedge = _bare_polar(fig, axes[1], station_lon, station_lat)
    wedge.bar([centre(m.back_azimuth)], [1], width=width, color=cmap(norm(pct_anisotropy)), edgecolor="black", lw=0.4,
              alpha=0.95)
    for polar in (rose, wedge):                                 # the station, on top of the petals and the wedge
        polar.scatter([0], [0], marker="^", s=230, color="#FFD700", edgecolors="black", lw=0.8, zorder=50, clip_on=False)
    axes[1].text(0.97, 0.03, f"A = {pct_anisotropy:.2f}%", transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=14, fontweight="bold", zorder=22,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
    pos = axes[1].get_position()
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([pos.x1 + 0.012, pos.y0, 0.018, pos.height]))
    cbar.set_label("Anisotropy (%)", fontsize=13, fontweight="bold")
    cbar.ax.grid(False)
    return fig
