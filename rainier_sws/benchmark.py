"""The week-8 benchmark: judge every station's splitting results the same way.

``run("STAR")`` loads a results file, applies the post-processing filter and returns a
``BenchmarkReport`` with, in this order:

1. what was measured: the file, whether it is STAR's re-run on corrected components, the
   thresholds, the funnel and the stage counts;
2. the pooled answer and its uncertainty: N, the axial mean fast direction with an event
   bootstrap and a day-block bootstrap interval, R, the circular standard deviation, the Rayleigh p,
   dt (median, and the 0.02 s step it comes in) and percent anisotropy (mean, interval, median);
3. what the answer is made of: the filter bands, with phi and dt per band;
4. whether it could be an artifact: the near-null fraction against the back azimuth (an artifact
   check) and, if the waveforms were re-measured, against the source polarization (the null screen);
5. how much it depends on the choices behind it: each post-processing threshold swept, with the
   swing of mean phi and dt compared with what dropping the same number of random events does; the
   incidence gate; an early/late split with permutation p-values (events, and whole days);
6. optionally, a pick-perturbation test (re-measures the waveforms; STAR only, about 40 s a shift).

Every choice is a named parameter (``BenchmarkParams``, ``quality.PostFilter``). Stations with no
measurements give N = 0 and say so in ``notes``. Notebook 07, section 8, explains each number;
``run_all()`` benchmarks every file in ``results/``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import measure, paths, quality, stats

# One threshold at a time is moved across these values, the others staying at their defaults.
SWEEPS = {
    "qw_min": (0.3, 0.4, 0.5, 0.6, 0.75, 0.9),
    "snr_min": (1.5, 2.0, 2.5, 3.0, 4.0, 6.0),
    "phi_error_max": (5, 10, 15, 20, 30, 45),
    "dt_error_max": (0.01, 0.02, 0.03, 0.05, 0.08, 0.12),
}


@dataclass(frozen=True)
class BenchmarkParams:
    """Every choice the benchmark makes. Change them here, never as literals in a notebook."""
    ci: float = 68.0                     # percent; the interval reported everywhere (about +-1 standard error)
    n_boot: int = 1000                   # bootstrap resamples
    block: str = "D"                     # block for the block bootstrap and permutation (pandas alias; "D" = UTC day)
    min_n: int = 5                       # fewer events than this: a statistic is not computed (NaN, with a note)
    null_width_deg: float = 20.0         # near-null: phi within this of the reference axis or its perpendicular
    sweeps: dict = field(default_factory=lambda: dict(SWEEPS))
    n_random: int = 500                  # random subsets for the threshold-swing baseline
    random_level: float = 95.0           # percentile of the random swings a real swing is compared with
    incidence_cutoffs: tuple = (30, 35, 40, 45, 50, 55, 60, 70)
    split_time: str = "2025-07-11"       # early/late boundary (UTC): the swarm's first three days against the rest
    n_perm: int = 2000                   # permutations per test
    sampling_rate_hz: float = 100.0      # dt step of a station with no delays to infer it from (see dt_step_of)
    perturb_shifts_s: tuple = (-0.01, 0.01)   # S-pick shifts of the perturbation test (one sample each way)
    moved_deg: float = 20.0              # a re-measured phi this far from the stored one counts as moved
    seed: int = 0


DEFAULT_PARAMS = BenchmarkParams()


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def load_results(source, oriented: bool | None = None) -> tuple[pd.DataFrame, Path]:
    """A results table and the file it came from.

    ``source`` is a path or a station name. For a station name, ``oriented=None`` takes the re-run on
    corrected components where one exists (only STAR's does), as the lessons do.
    """
    path = Path(source)
    if not path.suffix:
        station = str(source).upper()
        if oriented is None:
            oriented = paths.station_results_csv(station, oriented=True).exists()
        path = paths.station_results_csv(station, oriented=oriented)
    return pd.read_csv(path), path


def is_oriented_rerun(results: pd.DataFrame) -> bool:
    """True for STAR's re-run on waveforms rotated to true Z/N/E (the only file with an ``orientation`` column)."""
    return "orientation" in results.columns


def station_of(results: pd.DataFrame, path: Path | None = None) -> str:
    """The station a results table belongs to (its ``station`` column, else the file name)."""
    if "station" in results.columns and results["station"].notna().any():
        return str(results["station"].dropna().iloc[0]).upper()
    return Path(path).stem.split("_")[1].upper()


def event_times(results: pd.DataFrame) -> pd.Series:
    """Origin times (UTC) of the rows; NaT where the file has none."""
    if "datetime" not in results.columns:
        return pd.Series(pd.NaT, index=results.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(results["datetime"], utc=True, format="ISO8601")


def column(results: pd.DataFrame, name: str) -> pd.Series:
    """A column, or NaN if this file does not have it (stations with no measurements lack several)."""
    return results[name] if name in results.columns else pd.Series(np.nan, index=results.index)


def has_cached_waveforms(station: str) -> bool:
    """True if ``data/waveforms/<station>/`` holds miniSEED (only STAR's does)."""
    return any(paths.station_waveform_dir(station).glob("*.mseed"))


# ---------------------------------------------------------------------------------------------
# Small statistics the stats module does not have
# ---------------------------------------------------------------------------------------------
def block_bootstrap_mean(values, blocks, n_boot=1000, ci=68.0, seed=0):
    """Mean of ``values`` with a bootstrap interval that resamples whole blocks. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    v, b = np.asarray(values, dtype=float), np.asarray(blocks)
    groups = [v[b == label] for label in np.unique(b)]
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(v.mean()), float(lo), float(hi)


def block_permutation_test(values, blocks, in_a, axial=False, n_perm=2000, seed=0):
    """Two-sample permutation test that relabels whole blocks (days) instead of single events.

    ``in_a`` says which events belong to sample a; every block must lie entirely in a or in b (true for a
    time split at a block boundary). Which blocks are "a" is shuffled, keeping their number. The
    statistic is the difference of means (axial means with ``axial=True``). Returns (difference, p).
    """
    rng = np.random.default_rng(seed)
    v, b, a = np.asarray(values, dtype=float), np.asarray(blocks), np.asarray(in_a, dtype=bool)
    labels = np.unique(b)
    members = [np.flatnonzero(b == label) for label in labels]
    block_in_a = np.array([a[m].all() for m in members])
    if any(a[m].any() != a[m].all() for m in members):
        raise ValueError("a block holds events of both samples; split at a block boundary")

    def diff(blocks_a):
        ia = np.concatenate([members[i] for i in np.flatnonzero(blocks_a)])
        ib = np.concatenate([members[i] for i in np.flatnonzero(~blocks_a)])
        if axial:
            return float(stats.axial_diff(stats.axial_mean(v[ia])[0], stats.axial_mean(v[ib])[0]))
        return float(abs(v[ia].mean() - v[ib].mean()))

    d0 = diff(block_in_a)
    hits = sum(diff(rng.permutation(block_in_a)) >= d0 - 1e-12 for _ in range(n_perm))
    return d0, (hits + 1) / (n_perm + 1)


def random_subset_swings(larger: pd.DataFrame, n_small: int, n_random=500, seed=0):
    """How far the mean phi (axial) and mean dt move when ``n_small`` events are drawn at random from ``larger``.

    This is the baseline for a threshold swing: a threshold that only removed events at random would move
    the mean about this much. Returns two arrays (phi swings in deg, dt swings in s).
    """
    rng = np.random.default_rng(seed)
    phi, dt = larger["phi"].to_numpy(float), larger["dt"].to_numpy(float)
    m_phi, m_dt = stats.axial_mean(phi)[0], dt.mean()
    swings_phi, swings_dt = [], []
    for _ in range(n_random):
        i = rng.choice(len(phi), n_small, replace=False)
        swings_phi.append(float(stats.axial_diff(stats.axial_mean(phi[i])[0], m_phi)))
        swings_dt.append(abs(dt[i].mean() - m_dt))
    return np.array(swings_phi), np.array(swings_dt)


# ---------------------------------------------------------------------------------------------
# The pieces of the report
# ---------------------------------------------------------------------------------------------
def dt_step_of(dt, sampling_rate_hz: float = 100.0) -> float:
    """Spacing of a station's reported delays: their greatest common divisor, to the millisecond.

    SWSPy reports dt in steps of two samples (``measure.dt_step``): 0.02 s at 100 Hz, but 0.04 s at
    MILD, whose data were evidently sampled at 50 Hz. Without delays, two samples at ``sampling_rate_hz``.
    """
    dt = np.asarray(dt, dtype=float)
    ms = np.round(dt[np.isfinite(dt) & (dt > 0)] * 1000).astype(int)
    return float(np.gcd.reduce(ms)) / 1000 if len(ms) else measure.dt_step(sampling_rate_hz)


def pooled_summary(kept: pd.DataFrame, p: BenchmarkParams = DEFAULT_PARAMS) -> pd.Series:
    """N, the mean fast direction with event and block intervals, R, circular SD, Rayleigh p, dt, percent anisotropy."""
    n = len(kept)
    step = dt_step_of(column(kept, "dt"), p.sampling_rate_hz)
    out = {"N": n, "mean_phi": np.nan, "phi_lo": np.nan, "phi_hi": np.nan, "phi_block_lo": np.nan,
           "phi_block_hi": np.nan, "n_blocks": 0, "R": np.nan, "circ_sd": np.nan, "rayleigh_p": np.nan,
           "median_dt": np.nan, "mean_dt": np.nan, "dt_step": step, "dt_on_grid": np.nan,
           "pct_mean": np.nan, "pct_lo": np.nan, "pct_hi": np.nan, "pct_block_lo": np.nan,
           "pct_block_hi": np.nan, "pct_median": np.nan}
    if n < p.min_n:
        return pd.Series(out)
    blocks = event_times(kept).dt.floor(p.block).astype(str).to_numpy()
    m, lo, hi, _ = stats.bootstrap_axial_mean(kept["phi"], n_boot=p.n_boot, ci=p.ci, seed=p.seed)
    _, blo, bhi, _ = stats.block_bootstrap_axial_mean(kept["phi"], blocks, n_boot=p.n_boot, ci=p.ci, seed=p.seed)
    dt = kept["dt"].to_numpy(float)
    out.update(mean_phi=m, phi_lo=lo, phi_hi=hi, phi_block_lo=blo, phi_block_hi=bhi, n_blocks=len(set(blocks)),
               R=stats.axial_mean(kept["phi"])[1], circ_sd=stats.axial_std(kept["phi"]),
               rayleigh_p=stats.rayleigh_test(kept["phi"])[1], median_dt=float(np.median(dt)), mean_dt=float(dt.mean()),
               dt_on_grid=float(np.isclose(dt / step, np.round(dt / step)).mean()))
    pct = column(kept, "pct_anisotropy")
    ok = pct.notna().to_numpy()
    if ok.sum() >= p.min_n:
        pm, plo, phi_, _ = stats.bootstrap(pct[ok].to_numpy(float), np.mean, n_boot=p.n_boot, ci=p.ci, seed=p.seed)
        _, pblo, pbhi = block_bootstrap_mean(pct[ok], blocks[ok], n_boot=p.n_boot, ci=p.ci, seed=p.seed)
        out.update(pct_mean=pm, pct_lo=plo, pct_hi=phi_, pct_block_lo=pblo, pct_block_hi=pbhi,
                   pct_median=float(pct[ok].median()))
    return pd.Series(out)


def band_table(kept: pd.DataFrame, p: BenchmarkParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """One row per filter band: N, share, mean phi and R (if N >= min_n), median and mean dt, percent anisotropy."""
    rows = []
    for band, g in kept.assign(band=column(kept, "filter_band")).dropna(subset=["band"]).groupby("band"):
        big = len(g) >= p.min_n
        rows.append({"band": band, "N": len(g), "share": len(g) / len(kept),
                     "mean_phi": stats.axial_mean(g["phi"])[0] if big else np.nan,
                     "R": stats.axial_mean(g["phi"])[1] if big else np.nan,
                     "median_dt": g["dt"].median(), "mean_dt": g["dt"].mean(),
                     "mean_pct": column(g, "pct_anisotropy").mean(), "median_pct": column(g, "pct_anisotropy").median()})
    cols = ["band", "N", "share", "mean_phi", "R", "median_dt", "mean_dt", "mean_pct", "median_pct"]
    return pd.DataFrame(rows, columns=cols).sort_values("N", ascending=False).set_index("band")


def null_screen(phi, ref, label: str, p: BenchmarkParams = DEFAULT_PARAMS) -> dict:
    """Fraction of fast axes within ``null_width_deg`` of the reference axis or its perpendicular, with the chance level,
    and the mean fast direction with and without them."""
    phi, ref = np.asarray(phi, dtype=float), np.asarray(ref, dtype=float)
    keep = np.isfinite(phi) & np.isfinite(ref)
    phi, ref = phi[keep], ref[keep]
    near = stats.near_axis(phi, ref, p.null_width_deg)
    rest = phi[~near]
    big, big_rest = len(phi) >= p.min_n, len(rest) >= p.min_n
    return {"reference": label, "N": len(phi), "width_deg": p.null_width_deg, "n_near": int(near.sum()),
            "fraction": near.mean() if len(phi) else np.nan, "chance": stats.chance_near_axis(p.null_width_deg),
            "mean_phi_all": stats.axial_mean(phi)[0] if big else np.nan,
            "mean_phi_without": stats.axial_mean(rest)[0] if big_rest else np.nan,
            "R_without": stats.axial_mean(rest)[1] if big_rest else np.nan}


def threshold_sweeps(results: pd.DataFrame, post: quality.PostFilter = quality.DEFAULT_POST,
                     p: BenchmarkParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Move one post-processing threshold at a time; compare each swing of mean phi and dt with random subsets.

    Moving one threshold gives a set that contains, or is contained in, the default set. The baseline is the
    swing you get when the smaller set is a random draw of the same size from the larger one: a swing beyond
    it means the threshold removes (or admits) events that differ from the rest, not just fewer of them.
    """
    kept0 = quality.post_filtered(results, post)
    rows = []
    for param, values in p.sweeps.items():
        for v in values:
            kept = quality.post_filtered(results, replace(post, **{param: v}))
            row = {"param": param, "value": v, "default": bool(np.isclose(v, getattr(post, param))), "N": len(kept),
                   "N_change": len(kept) - len(kept0), "mean_phi": np.nan, "phi_swing": np.nan, "phi_random": np.nan,
                   "mean_dt": np.nan, "dt_swing": np.nan, "dt_random": np.nan, "beyond_random": False}
            if len(kept) >= p.min_n and len(kept0) >= p.min_n:
                larger, n_small = (kept0, len(kept)) if len(kept) <= len(kept0) else (kept, len(kept0))
                rand_phi, rand_dt = random_subset_swings(larger, n_small, p.n_random, p.seed)
                m = stats.axial_mean(kept["phi"])[0]
                row.update(mean_phi=m, phi_swing=float(stats.axial_diff(m, stats.axial_mean(kept0["phi"])[0])),
                           phi_random=float(np.percentile(rand_phi, p.random_level)), mean_dt=kept["dt"].mean(),
                           dt_swing=abs(kept["dt"].mean() - kept0["dt"].mean()),
                           dt_random=float(np.percentile(rand_dt, p.random_level)))
                row["beyond_random"] = bool(row["phi_swing"] > row["phi_random"] + 1e-9)
            rows.append(row)
    return pd.DataFrame(rows)


def incidence_sweep(results: pd.DataFrame, post: quality.PostFilter = quality.DEFAULT_POST,
                    p: BenchmarkParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """For each incidence cutoff: events the gate would admit (stored ray-traced angles), and the filtered events
    at or below it with their mean phi and dt. Events beyond the gate were never measured, so above the gate this
    only counts what a re-run could attempt."""
    inc = column(results, "incidence_angle")
    kept = quality.post_filtered(results, post)
    rows = []
    for c in p.incidence_cutoffs:
        sub = kept[column(kept, "incidence_angle") <= c]
        rows.append({"cutoff_deg": c, "admitted": int((inc <= c).sum()), "filtered": len(sub),
                     "mean_phi": stats.axial_mean(sub["phi"])[0] if len(sub) >= p.min_n else np.nan,
                     "mean_dt": sub["dt"].mean() if len(sub) >= p.min_n else np.nan})
    return pd.DataFrame(rows).set_index("cutoff_deg")


def time_split(kept: pd.DataFrame, p: BenchmarkParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Early against late (``split_time``) for phi and dt, over all bands and within the most common band.

    Two p-values: ``p_event`` shuffles single events (``stats.permutation_test``); ``p_block`` shuffles whole
    blocks (days), which is the honest one when events cluster in time.
    """
    t = event_times(kept)
    split = pd.Timestamp(p.split_time, tz="UTC")
    if split.floor(p.block) != split:
        raise ValueError(f"split_time {p.split_time} is not at a {p.block} boundary")
    early, blocks = (t < split).to_numpy(), t.dt.floor(p.block).astype(str).to_numpy()
    bands = column(kept, "filter_band")
    subsets = [("all bands", np.ones(len(kept), dtype=bool))]
    if bands.notna().any():
        main = bands.value_counts().idxmax()
        subsets.append((f"{main} only", (bands == main).to_numpy()))
    rows = []
    for name, mask in subsets:
        for q in ("phi", "dt"):
            x, e, b = kept[q].to_numpy(float)[mask], early[mask], blocks[mask]
            row = {"subset": name, "quantity": q, "N_early": int(e.sum()), "N_late": int((~e).sum()),
                   "days_early": len(set(b[e])), "days_late": len(set(b[~e])), "early": np.nan, "late": np.nan,
                   "difference": np.nan, "p_event": np.nan, "p_block": np.nan}
            if e.sum() >= p.min_n and (~e).sum() >= p.min_n:
                axial = q == "phi"
                mean = (lambda v: stats.axial_mean(v)[0]) if axial else np.mean
                d, p_event = stats.permutation_test(x[e], x[~e], axial=axial, n_perm=p.n_perm, seed=p.seed)
                _, p_block = block_permutation_test(x, b, e, axial=axial, n_perm=p.n_perm, seed=p.seed)
                row.update(early=mean(x[e]), late=mean(x[~e]), difference=d, p_event=p_event, p_block=p_block)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Re-measuring (needs cached waveforms: STAR only)
# ---------------------------------------------------------------------------------------------
def remeasure_filtered(results: pd.DataFrame, station: str, post: quality.PostFilter = quality.DEFAULT_POST,
                       shifts=(0.0,), oriented: bool | None = None) -> dict:
    """Re-measure the filtered events, once per S-pick shift. Returns {shift: ``measure.remeasure`` table}.

    About 0.2 s per STAR event and shift. Files computed on STAR's channels as labelled are re-measured that
    way, without the clipping gate, so that the zero shift reproduces them.
    """
    if not has_cached_waveforms(station):
        raise FileNotFoundError(f"no cached waveforms for {station}; only STAR's are in data/waveforms/")
    kept = quality.post_filtered(results, post)
    oriented = is_oriented_rerun(results) if oriented is None else oriented
    kw = {} if oriented else {"correct_orientation": False, "gates": quality.PreGates(clip_fraction_max=np.inf)}
    return {s: measure.remeasure(kept, station, s_shift_s=s, **kw) for s in shifts}


def perturbation_test(kept: pd.DataFrame, perturbed: dict, post: quality.PostFilter = quality.DEFAULT_POST,
                      p: BenchmarkParams = DEFAULT_PARAMS) -> pd.Series:
    """How many filtered events leave the filtered set, or change phi by more than ``moved_deg``, when the pick moves.

    ``perturbed`` is ``remeasure_filtered(..., shifts=p.perturb_shifts_s)``. The stable events pass the filter at
    every shift with phi within ``moved_deg`` of the stored value.
    """
    out = {"N_tested": len(kept), "shifts_s": tuple(perturbed)}
    passes_all = pd.Series(True, index=kept.index)
    moved_any = pd.Series(False, index=kept.index)
    for s, m in perturbed.items():
        m = m.reindex(kept.index)
        passes = quality.post_filter_mask(m, post)
        moved = (m["stage"] == "ok") & (stats.axial_diff(m["phi"], kept["phi"]) > p.moved_deg)
        out[f"leave_at_{s:+.2f}s"] = int((~passes).sum())
        passes_all &= passes
        moved_any |= moved
    stable = passes_all & ~moved_any
    out.update(leave_any=int((~passes_all).sum()), moved_any=int(moved_any.sum()),
               moved_but_kept=int((moved_any & passes_all).sum()), N_stable=int(stable.sum()))
    phi = kept.loc[stable, "phi"]
    if len(phi) >= p.min_n:
        m, lo, hi, _ = stats.bootstrap_axial_mean(phi, n_boot=p.n_boot, ci=p.ci, seed=p.seed)
        out.update(mean_phi_stable=m, phi_lo=lo, phi_hi=hi, R_stable=stats.axial_mean(phi)[1])
    return pd.Series(out)


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------
@dataclass
class BenchmarkReport:
    """Everything ``benchmark`` found for one results file. ``row()`` gives the headline numbers, ``str()`` a text report."""
    label: str
    station: str
    source: str
    oriented: bool
    post: quality.PostFilter
    pre: quality.PreGates
    params: BenchmarkParams
    funnel: pd.Series
    stages: pd.Series
    summary: pd.Series
    bands: pd.DataFrame
    nulls: pd.DataFrame
    sweeps: pd.DataFrame
    incidence: pd.DataFrame
    time_split: pd.DataFrame
    perturbation: pd.Series | None = None
    notes: list = field(default_factory=list)

    def row(self) -> dict:
        """One line of the week-8 table."""
        s, nulls = self.summary, self.nulls.set_index("reference")
        swings = self.sweeps.dropna(subset=["phi_swing"])
        worst = swings.loc[swings["phi_swing"].idxmax()] if len(swings) else None
        ts = self.time_split.set_index(["subset", "quantity"])
        phi_all = ts.loc[("all bands", "phi")] if ("all bands", "phi") in ts.index else None
        return {
            "file": self.label, "station": self.station, "oriented": self.oriented,
            "attempted": int(self.funnel["attempted"]), "successful": int(self.funnel["successful"]), "N": int(s["N"]),
            "mean_phi": s["mean_phi"], "phi_68_event": (s["phi_lo"], s["phi_hi"]),
            "phi_68_day": (s["phi_block_lo"], s["phi_block_hi"]), "R": s["R"], "rayleigh_p": s["rayleigh_p"],
            "median_dt": s["median_dt"], "dt_step": s["dt_step"], "pct_mean": s["pct_mean"],
            "pct_68": (s["pct_lo"], s["pct_hi"]), "pct_median": s["pct_median"],
            "main_band": self.bands.index[0] if len(self.bands) else None,
            "main_band_share": self.bands["share"].iloc[0] if len(self.bands) else np.nan,
            "near_baz": nulls.loc["back azimuth", "fraction"], "chance": nulls.loc["back azimuth", "chance"],
            "near_src_pol": nulls.loc["source polarization", "fraction"] if "source polarization" in nulls.index else np.nan,
            "largest_phi_swing": worst["phi_swing"] if worst is not None else np.nan,
            "swing_from": f"{worst['param']} = {worst['value']}" if worst is not None else None,
            "beyond_random": sorted(set(swings.loc[swings["beyond_random"], "param"])),
            "early_late_dphi": phi_all["difference"] if phi_all is not None else np.nan,
            "p_event": phi_all["p_event"] if phi_all is not None else np.nan,
            "p_day": phi_all["p_block"] if phi_all is not None else np.nan,
            "notes": len(self.notes),
        }

    def __str__(self) -> str:
        return report_text(self)


def benchmark(results: pd.DataFrame, station: str, post: quality.PostFilter = quality.DEFAULT_POST,
              params: BenchmarkParams = DEFAULT_PARAMS, pre: quality.PreGates = quality.DEFAULT_PRE,
              source: str = "", remeasured: pd.DataFrame | None = None, perturbed: dict | None = None) -> BenchmarkReport:
    """Benchmark one results table. Uses nothing but its arguments.

    ``remeasured`` (``remeasure_filtered(...)[0.0]``) adds the null screen against the source polarization;
    ``perturbed`` (``remeasure_filtered(..., shifts=params.perturb_shifts_s)``) adds the perturbation test.
    """
    p = params
    kept = quality.post_filtered(results, post)
    notes = []
    oriented = is_oriented_rerun(results)
    if station == "STAR" and not oriented:
        notes.append("STAR's channels as labelled (vertical and north exchanged): not the corrected re-run")
    if len(kept) == 0:
        notes.append("N = 0: no measurement passes the post-processing filter; nothing below is computed")
    elif len(kept) < p.min_n:
        notes.append(f"N = {len(kept)} < min_n = {p.min_n}: statistics are not computed")
    summary = pooled_summary(kept, p)
    if len(kept) and summary["dt_on_grid"] < 1:
        notes.append(f"some dt values are off the {summary['dt_step']:g} s grid: check the sampling rate")
    ok = quality.successful(results)
    beyond = int((column(ok, "incidence_angle") > pre.incidence_max_deg).sum())
    if beyond:
        notes.append(f"{beyond} successful measurements have a stored incidence angle above the "
                     f"{pre.incidence_max_deg:g} deg gate: this file was not gated on those angles")
    nulls = [null_screen(kept["phi"], column(kept, "back_azimuth"), "back azimuth", p)]
    if remeasured is not None:
        nulls.append(null_screen(kept["phi"], column(remeasured.reindex(kept.index), "src_pol"), "source polarization", p))
    if event_times(kept).isna().any():
        notes.append("some events have no origin time: block statistics and the time split are unreliable")
    ts = time_split(kept, p)
    if len(kept) >= p.min_n and ts["p_event"].isna().all():
        notes.append(f"early/late split at {p.split_time}: fewer than {p.min_n} events on one side")
    pert = perturbation_test(kept, perturbed, post, p) if perturbed is not None else None
    return BenchmarkReport(
        label=Path(source).stem.replace("sws_", "").replace("_results", "") if source else station.lower(),
        station=station, source=str(source), oriented=oriented, post=post, pre=pre, params=p,
        funnel=quality.funnel(results, post), stages=quality.stage_counts(results), summary=summary,
        bands=band_table(kept, p), nulls=pd.DataFrame(nulls), sweeps=threshold_sweeps(results, post, p),
        incidence=incidence_sweep(results, post, p), time_split=ts, perturbation=pert, notes=notes)


def run(source, post: quality.PostFilter = quality.DEFAULT_POST, params: BenchmarkParams = DEFAULT_PARAMS,
        oriented: bool | None = None, remeasure: bool = False, perturb: bool = False) -> BenchmarkReport:
    """Load a results file (a path or a station name) and benchmark it.

    ``remeasure=True`` re-measures the filtered events to screen nulls against the source polarization;
    ``perturb=True`` re-measures them with the S pick shifted. Both need cached waveforms (STAR only); for other
    stations they are skipped with a note.
    """
    results, path = load_results(source, oriented)
    station = station_of(results, path)
    remeasured = perturbed = None
    skipped = []
    if (remeasure or perturb) and not has_cached_waveforms(station):
        skipped.append(f"re-measurement skipped: no cached waveforms for {station}")
    elif remeasure or perturb:
        if remeasure:
            remeasured = remeasure_filtered(results, station, post)[0.0]
        if perturb:
            perturbed = remeasure_filtered(results, station, post, shifts=params.perturb_shifts_s)
    report = benchmark(results, station, post, params, source=str(path), remeasured=remeasured, perturbed=perturbed)
    report.notes.extend(skipped)
    return report


def run_all(results_dir: Path = paths.RESULTS, **kwargs) -> list[BenchmarkReport]:
    """Benchmark every ``sws_*_results.csv`` in ``results_dir`` (keyword arguments go to ``run``)."""
    return [run(path, **kwargs) for path in sorted(Path(results_dir).glob("sws_*_results.csv"))]


def summary_table(reports) -> pd.DataFrame:
    """The week-8 table: one row per report."""
    return pd.DataFrame([r.row() for r in reports]).set_index("file")


def report_text(r: BenchmarkReport) -> str:
    """A plain-text report, section by section (what ``print(report)`` shows)."""
    s, f = r.summary, r.funnel
    fmt = lambda x, d=1: "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"  # noqa: E731
    lines = [f"== {r.label}: station {r.station} ==",
             f"file: {r.source}" + ("  (STAR's re-run on corrected components)" if r.oriented else ""),
             f"post-processing filter: {r.post.as_dict()}",
             f"funnel: attempted {f['attempted']}, successful {f['successful']}, filtered {f['filtered']}",
             "stages: " + ", ".join(f"{k} {v}" for k, v in r.stages.items())]
    lines += [f"note: {n}" for n in r.notes]
    if s["N"] >= r.params.min_n:
        ci = r.params.ci
        lines += [
            f"N = {int(s['N'])}: mean phi {fmt(s['mean_phi'])} deg, {ci:g}% interval {fmt(s['phi_lo'])}/+{fmt(s['phi_hi'])} "
            f"(events), {fmt(s['phi_block_lo'])}/+{fmt(s['phi_block_hi'])} ({int(s['n_blocks'])} blocks of {r.params.block})",
            f"R {fmt(s['R'], 2)}, circular SD {fmt(s['circ_sd'])} deg, Rayleigh p {s['rayleigh_p']:.1e}",
            f"dt: median {fmt(s['median_dt'], 2)} s, mean {fmt(s['mean_dt'], 3)} s, in steps of {s['dt_step']:g} s "
            f"({s['dt_on_grid']:.0%} on the grid)",
            f"percent anisotropy: mean {fmt(s['pct_mean'], 2)} ({ci:g}% {fmt(s['pct_lo'], 2)} to {fmt(s['pct_hi'], 2)} events, "
            f"{fmt(s['pct_block_lo'], 2)} to {fmt(s['pct_block_hi'], 2)} days), median {fmt(s['pct_median'], 2)}",
            "", "bands:", r.bands.round(3).to_string(),
            "", "near-null fractions (back azimuth: artifact check; source polarization: null screen):",
            r.nulls.round(3).to_string(index=False),
            "", "threshold sweeps (swing of mean phi against random subsets of the same size):",
            r.sweeps[["param", "value", "N", "phi_swing", "phi_random", "dt_swing", "dt_random", "beyond_random"]]
            .round(3).to_string(index=False),
            "", "incidence gate:", r.incidence.round(2).to_string(),
            "", f"early (before {r.params.split_time}) against late:", r.time_split.round(4).to_string(index=False)]
    if r.perturbation is not None:
        lines += ["", "pick perturbation:", r.perturbation.to_string()]
    return "\n".join(lines)
