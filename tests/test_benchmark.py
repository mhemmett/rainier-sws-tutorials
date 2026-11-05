"""Tests of the week-8 benchmark (rainier_sws/benchmark.py), the numbers notebook 07 quotes from it, and the
statistics it adds. Fast: no waveform is re-measured here (the perturbation and source-polarization parts are
tested on hand-made tables)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rainier_sws import benchmark, paths, quality  # noqa: E402

FAST = benchmark.BenchmarkParams(n_boot=200, n_perm=200, n_random=100)


@pytest.fixture(scope="module")
def reports():
    return {r.label: r for r in benchmark.run_all(params=FAST)}


@pytest.fixture(scope="module")
def star():
    results, path = benchmark.load_results("STAR")
    return results, path


def test_runs_on_every_results_file(reports):
    files = sorted(paths.RESULTS.glob("sws_*_results.csv"))
    assert len(reports) == len(files) == 11
    table = benchmark.summary_table(reports.values())
    assert set(table.station) >= {"STAR", "RCM", "MILD", "LON", "PANH", "OBSR", "PARA", "RER", "COPP"}
    for r in reports.values():
        assert str(r)                                   # the text report renders for every file


def test_empty_stations_say_so(reports):
    # OBSR has one successful measurement and none filtered; it used to come back as silent NaNs
    for label in ("obsr_strict", "panh_strict", "para_strict", "rer_strict", "copp_strict"):
        r = reports[label]
        assert r.summary["N"] == 0 and np.isnan(r.summary["mean_phi"])
        assert any("N = 0" in n for n in r.notes)


def test_star_is_the_oriented_rerun_and_matches_the_readme(reports):
    r = reports["star_strict_oriented"]
    assert r.oriented and r.station == "STAR"
    assert (r.funnel["attempted"], r.funnel["successful"], r.summary["N"]) == (1196, 659, 194)
    assert r.summary["mean_phi"] == pytest.approx(33.7, abs=0.05)
    assert r.summary["dt_step"] == pytest.approx(0.02) and r.summary["dt_on_grid"] == 1.0
    # the day blocks widen the interval: events cluster in time
    assert r.summary["phi_block_hi"] - r.summary["phi_block_lo"] > 1.5 * (r.summary["phi_hi"] - r.summary["phi_lo"])
    assert r.bands.index[0] == "(1.0, 5.0)" and r.bands["N"].iloc[0] == 136
    baz = r.nulls.set_index("reference").loc["back azimuth"]
    assert (baz["N"], baz["n_near"]) == (194, 57)
    assert not any("labelled" in n for n in r.notes)
    assert any("labelled" in n for n in reports["star_strict"].notes)


def test_lon_was_not_gated_on_the_stored_angles(reports):
    assert any("not gated" in n for n in reports["lon_strict"].notes)


def test_time_split_counts(reports):
    ts = reports["star_strict_oriented"].time_split.set_index(["subset", "quantity"])
    assert (ts.loc[("all bands", "phi"), "N_early"], ts.loc[("all bands", "phi"), "N_late"]) == (87, 107)
    assert (ts.loc[("(1.0, 5.0) only", "dt"), "N_early"], ts.loc[("(1.0, 5.0) only", "dt"), "N_late"]) == (51, 85)
    # the VT background lies entirely before the split: reported as missing, not as a number
    assert ts.notna().all().all() and reports["vt_star"].time_split["p_event"].isna().all()


def test_block_permutation_respects_blocks():
    rng = np.random.default_rng(1)
    # ten days, each with its own offset, and no difference between the first three days and the rest
    day_offsets = rng.normal(0, 1.0, 10)
    blocks = np.repeat(np.arange(10), 20)
    values = day_offsets[blocks] + rng.normal(0, 0.1, blocks.size)
    early = blocks < 3
    _, p_event = benchmark.stats.permutation_test(values[early], values[~early], n_perm=500)
    _, p_block = benchmark.block_permutation_test(values, blocks, early, n_perm=500)
    assert p_event < 0.01 < 0.05 < p_block          # single events look significant, days do not
    # a real step between the two halves is found by both
    values2 = rng.normal(0, 0.1, blocks.size) + 3.0 * early
    assert benchmark.block_permutation_test(values2, blocks, early, n_perm=500)[1] < 0.05
    with pytest.raises(ValueError):
        benchmark.block_permutation_test(values, blocks, np.arange(blocks.size) < 50)


def test_random_subset_baseline_flags_a_different_population():
    rng = np.random.default_rng(2)
    same = pd.DataFrame({"phi": rng.normal(30, 10, 200), "dt": 0.06})
    other = pd.DataFrame({"phi": rng.normal(60, 10, 40), "dt": 0.06})
    larger = pd.concat([same, other], ignore_index=True)
    rand_phi, _ = benchmark.random_subset_swings(larger, len(same), n_random=200)
    swing = benchmark.stats.axial_diff(benchmark.stats.axial_mean(same.phi)[0], benchmark.stats.axial_mean(larger.phi)[0])
    assert swing > np.percentile(rand_phi, 95)


def test_sweeps_are_nested_and_star_q_w_is_beyond_random(reports):
    sw = reports["star_strict_oriented"].sweeps
    assert (sw.loc[sw.default, "N"] == 194).all() and (sw.loc[sw.default, "phi_swing"] == 0).all()
    pe = sw[(sw.param == "phi_error_max") & (sw.value == 45)].iloc[0]
    assert pe.N == 281 and pe.beyond_random


def test_perturbation_and_src_pol_bookkeeping(star):
    results, path = star
    kept = quality.post_filtered(results)
    same = kept.copy()
    moved = kept.copy()
    moved.loc[moved.index[0], "phi"] = moved.loc[moved.index[0], "phi"] + 45     # moves, still passes
    failed = kept.copy()
    failed.loc[failed.index[1], "quality"] = 0.1                                  # leaves the filtered set
    rep = benchmark.benchmark(results, "STAR", params=FAST, source=str(path),
                              remeasured=kept.assign(src_pol=kept.back_azimuth),
                              perturbed={-0.01: moved, 0.01: failed})
    pert = rep.perturbation
    assert (pert["N_tested"], pert["leave_any"], pert["moved_any"], pert["moved_but_kept"], pert["N_stable"]) == (194, 1, 1, 1, 192)
    nulls = rep.nulls.set_index("reference")
    # with the back azimuth standing in for the source polarization the two screens must agree
    assert nulls.loc["source polarization", "n_near"] == nulls.loc["back azimuth", "n_near"] == 57
    assert benchmark.perturbation_test(kept, {0.0: same}).loc["N_stable"] == 194


def test_remeasure_needs_the_cache():
    results, _ = benchmark.load_results("RCM")
    with pytest.raises(FileNotFoundError):
        benchmark.remeasure_filtered(results, "RCM")
    r = benchmark.run("RCM", params=FAST, remeasure=True)
    assert any("skipped" in n for n in r.notes) and len(r.nulls) == 1


def test_dt_step_is_inferred_per_station():
    assert benchmark.dt_step_of([0.04, 0.08, 0.04]) == pytest.approx(0.04)
    assert benchmark.dt_step_of([0.04, 0.06]) == pytest.approx(0.02)
    assert benchmark.dt_step_of([]) == pytest.approx(0.02)
    assert benchmark.run("MILD").summary["dt_step"] == pytest.approx(0.04)   # MILD's delays come in 0.04 s steps
