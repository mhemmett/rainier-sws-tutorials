# Rainier shear-wave splitting tutorials

A guided, notebook-based introduction to seismic data and shear-wave splitting (SWS), built around
the Mt. Rainier July 2025 hydrothermal swarm and the preliminary results of the
[mt-rainier-swarm-catalog](https://github.com/mhemmett/mt-rainier-swarm-catalog) research project.

Written for a student with Python experience and no seismology background. The lessons start from
"what is a seismogram" and end with reproducing a published-quality splitting measurement,
putting error bars on the pooled results, benchmarking the thresholds that produced them, and
running that benchmark on every station.

Maintained by Michael Hemmett (UW ESS, Denolle Lab).

## What we are doing and why (read this first)

**The volcano.** Tahoma, or Mount Rainier, is rated a "very high threat" volcano by the USGS, not
because it erupts often (about ten eruptions in the last 2,600 years, the most recent about 1,000
years ago) but because of lahars: fast, far-travelling debris flows. The Osceola Mudflow (about
5,600 years ago) and the Electron Mudflow (about 500 years ago) reached what is now the Puget
Lowland, and both were made of **hydrothermally altered rock**: andesite that hot, slightly acidic
water had turned into weak clay (Finn et al., 2001). That water is still at work. Magma-derived gases
rise from a cooling intrusive complex 8 to 18 km below the summit (a low-velocity body in the
tomography of Moran et al., 1999), mix with meteoric water, and circulate through a network of faults
and fractures to the summit fumaroles and the mineral springs on the flanks (Moran et al., 2000). The
same circulation triggers small earthquakes: a steady background of a few per month at about sea
level (4 km below the summit), and, occasionally, a swarm.

**The two swarms.** On 20 to 22 September 2009 more than a thousand small earthquakes occurred
beneath the summit, the most ever recorded in a day at Rainier at the time; only about 120 could be
located by hand. Shelly, Moran and Thelen (2013) used waveform cross-correlation to locate 726 of
them and found that they defined an 850-m-long, nearly vertical structure striking NNE, with the
activity front migrating outward the way a diffusing pressure pulse would, and with focal mechanisms
oblique to that structure, a pattern they interpreted as **fluid-triggered slip on an en echelon
fault mesh**. On 8 July 2025 at 01:29 PDT a new swarm began. It became the largest ever recorded at
the volcano: a peak of 41 located events per hour on the first morning, about 1,350 located
earthquakes by 25 August, a largest magnitude of 2.4 on 11 July, depths of 2 to 6 km beneath the
summit, and many more events visible on the summit stations than could be located. A gas flight on
4 August measured water vapour and CO2 but no SO2, so the swarm was driven by the **shallow
hydrothermal system, not new magma** (USGS Cascades Volcano Observatory; PNSN). The alert level
never left GREEN. Two things make 2025 special for us: the permanent network grew from 9 to 26
stations between the swarms, and the Denolle Lab deployed **240 temporary nodal seismometers**
(network code Z5, mid-July to mid-August 2025) that caught the end of the swarm and the recovery.
An enhanced PNSN catalog of the 2025 swarm with close to 9,000 events is in preparation (Stevens et
al.).

**The question.** The structure and dynamics of the fault-fracture network that carries these fluids
are poorly known, and seismology has contributed little because there were few earthquakes and few
stations. Now there are many of both. Michael's research asks: *is the extent and orientation of
the hydrothermal fault-fracture network controlled by local geology (old dikes, crustal faults,
layered flows) or by the regional stress field (NE compression from the clockwise rotation of the
Pacific Northwest, with a vertical maximum stress under the summit from the weight of the edifice)?
And does the network change between and during swarms?* The working hypothesis is that local
geology dominates: fast directions should line up with the structures that relocated seismicity
reveals rather than with the regional compression, and the strength of anisotropy should not differ
much between 2009 and 2025.

**The method.** When a shear wave crosses rock with aligned cracks it splits into two orthogonally
polarised waves, a fast one polarised along the cracks and a slow one across them, separated by a
delay. **Shear-wave splitting** measures the fast direction, phi, and the delay time, dt, from each
earthquake-station pair (Silver and Chan, 1991, via the SWSPy package of Hudson et al., 2023, with
the quality control of MFAST, Savage et al., 2010). Phi maps crack orientation; dt divided by the
ray-traced travel time gives **percent anisotropy**, a path-averaged crack density. The research
plan is: (1) rebuild the 2025 catalog with deep-learning picking, association, 3-D absolute location
and double-difference relocation, including the nodes; (2) run splitting on the relocated 2009 and
2025 catalogs; (3) compare fast directions to geological structures and to the regional stress, and
percent anisotropy before, during and after each swarm.

**Where these tutorials fit.** Step 2 has been run once, as a proof of concept, on the unrelocated
PNSN catalog at a handful of permanent stations. Its first version gave about 2.5 percent anisotropy
and fast directions that diverged from the regional stress, which seemed to argue for local control, but
STAR's numbers came from exchanged channels (see *Known facts*). Corrected, STAR's fast directions cluster
at about 34 degrees, close to the regional compression (about NE), and the 2009 structure strikes the same
way, so at STAR the azimuth alone cannot separate stress from structure; differences between stations and
changes with time have to. Its percent anisotropy is about 3, and the apparent changes across the swarm are
at least partly a change in which filter band the pipeline chose (notebook 03). Those numbers are
station-dependent and sensitive to ray path, filtering thresholds and software choices, and that is the
honest state of the science. Your job in these
lessons is to understand the measurement well enough to put error bars on it, to find out which
thresholds it depends on, to check that the data behind it are sound (station clocks included), and
by week 8 to run the whole benchmark on every station from one script. Winter quarter builds on
that, below.

## Setup

You need a terminal, git, and a conda-based Python. If you have never used a terminal or git,
start with the Software Carpentry lessons linked from notebook 00 (an afternoon each), then come
back here. The lessons and tests run on macOS, Windows and Linux.

1. **Install Miniconda** (skip if `conda --version` already works).

   *macOS or Linux*, in a terminal:

   ```bash
   # macOS, Apple silicon
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
   bash Miniconda3-latest-MacOSX-arm64.sh
   # macOS, Intel:  Miniconda3-latest-MacOSX-x86_64.sh
   # Linux:         Miniconda3-latest-Linux-x86_64.sh
   ```

   Accept the licence, keep the default location, say yes to `conda init`, then **open a new
   terminal**.

   *Windows*:

   - Install **Git for Windows** first, from [git-scm.com/download/win](https://git-scm.com/download/win),
     keeping the default options. Besides `git` itself it gives you **Git Bash**, the terminal the
     Software Carpentry shell lesson uses, and the environment build below needs `git` to install
     chronos.
   - Download and run `Miniconda3-latest-Windows-x86_64.exe` from
     [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html):
     "Just Me", default location, leave "add to PATH" unticked.
     On a laptop with a Snapdragon (ARM) processor, use that same x86_64 installer; Windows runs it
     under emulation.
   - From the Start menu open **Anaconda Prompt (miniconda3)** and run every command below there.
     (To use PowerShell or Git Bash instead, run `conda init powershell` or `conda init bash` once
     in the Anaconda Prompt and reopen the terminal.)
   - Clone into a plain folder such as `C:\Users\<you>\rainier-sws-tutorials`, not a OneDrive-synced
     Documents or Desktop folder: OneDrive locks and re-uploads files while git and Jupyter are
     writing them.

   On any system, add the community channel once: `conda config --add channels conda-forge`.

2. **Get the repo and build the environment** (the same commands on every system):

   ```bash
   git clone https://github.com/mhemmett/rainier-sws-tutorials.git
   cd rainier-sws-tutorials
   conda env create -f environment.yml      # about 5 minutes; also pip-installs chronos and chronfix from GitHub
   conda activate rainier-sws
   pytest -q                                # about 30 s; the slowest test reproduces one real measurement
   jupyter lab notebooks/
   ```

`numpy < 2` is required (see `environment.yml` for why). Everything else is standard.

The notebooks run **offline**: the seismograms they need are cached in `data/waveforms/`. With a
network connection, notebook 02 also makes live requests to EarthScope so you can see where the
data come from, and notebook 04b can download continuous data
for the real timing check.

## The lessons

Do them in order. Each starts with a goal and a time estimate and ends with exercises.

| # | Notebook | You will learn to | Time |
| --- | --- | --- | --- |
| 00 | `00_setup_check` | check the environment, find your way around the repo, and where to learn the shell, git and conda | 10 min (+ the linked tutorials) |
| 01 | `01_what_is_a_seismogram` | read miniSEED with ObsPy, plot Z/N/E with P and S picks, read a seismogram | 1 h |
| 02 | `02_data_catalogs_and_maps` | pull station metadata and waveforms from EarthScope; which way the components point; event catalogs and phase picks; map the swarm on Rainier's topography; back azimuth and distance | 2 h |
| 03 | `03_filtering_spectra_snr` | where a seismogram's frequencies come from (source, path, instrument, noise); spectra, spectrograms, bandpass filters; what the SNR gate, filter bank and dominant period measure; the timescales of splitting | 2.5 h |
| 04 | `04_what_is_shear_wave_splitting` | build the splitting measurement from scratch on a synthetic signal; noise, nulls, cycle skips | 2 h |
| 04b | `04b_timing_and_clock_errors` | what a clock error does to a dataset; find one in the catalog (STAR); measure and fix it with chronos on a synthetic pair; the recipe for the real STAR data | 1.5 h |
| 05 | `05_single_event_with_swspy` | reproduce Michael's measurement of one real event to the digit with SWSPy, then redo it with STAR's components the right way round | 1.5 h |
| 06 | `06_exploring_the_results` | read the per-station results, stage bookkeeping, histograms, rose diagrams | 1.5 h |
| 07 | `07_uncertainty_and_thresholds` | bootstrap intervals, circular statistics, threshold sweeps, null screening, a benchmark function | 2 to 3 sessions |

Suggested pacing at 3 hours a week: weeks 1 to 2 for notebooks 00 to 03, week 3 for 04 and 04b,
week 4 for 05, week 5 for 06, weeks 6 to 7 for 07, and **week 8 to scale up**: turn the `benchmark`
function of notebook 07 into a script that runs on every station in `results/` (the nine summit
stations, the VT background at STAR) and produces one table, with an N, an uncertainty and a
threshold-sensitivity for each. The *Winter quarter* section below is what comes after.

## Winter quarter

Week 8 ends with a benchmark that runs on more stations. Winter quarter is about running it on more
*data*, in this order of priority. Each item is a self-contained project with a clear "done".

1. **Timing correction applied to the pipeline.** Notebook 04b finds, from the picks alone, that
   STAR's clock looks to have wandered by about 100 ms peak to peak during the swarm. Measure it
   properly: run the chronos recipe on continuous STAR data against RCM (4.9 km away) and against the
   nearest Z5 nodes (GPS-timed; nodes 227 and 129 near Camp Muir), get an hourly delta_t(t) with
   error bars, and apply it with chronfix to the cached STAR event windows. Done: the before/after
   peak-lag track, a comparison with the catalog residuals, and the splitting measurements of a few
   events re-run on corrected waveforms. If the drift is real and larger than about 20 ms, PNSN and
   CVO need to hear about it before the phase-2 catalog is built.
2. **The 2009 swarm at STAR.** STAR recorded 2009 too (installed September 2008) and has the most
   high-quality picks of any station in the 2009 catalog (127 pairs with P and S quality >= 0.75),
   yet 2009 splitting has only been run at LON. The research repo's
   `download_waveforms_2009_station.py STAR UW EH?` fetches the windows. Done: the 2009 STAR rose and
   percent anisotropy with the week-8 benchmark, side by side with 2025 at the same station; that is
   the proposal's swarm-to-swarm comparison at one site. Mind the clock: in 2009 STAR was an analog
   telemetry station digitized in Seattle (Shelly et al., 2013, section 2.1), which means a fixed
   delay rather than a drift.
3. **All of 2025 at STAR.** The VT-background catalog already uses ComCat phase data for events
   PNSN's export does not cover. Extend it to January to December 2025 so the swarm sits between a
   before and an after at one station. Done: fast direction and percent anisotropy per month, with
   the swarm months marked.
4. **First look at the nodes.** Once picks exist on Z5 stations, run the benchmark on the nodes
   nearest the summit. Done: one rose per node and a map.

## What is in the repo

```
notebooks/        the lessons, executed, with outputs saved so you can see what to expect
rainier_sws/      helper package the notebooks import (read it; it is short)
  paths.py          where things live
  data.py           catalogs, station coordinates, topography, waveforms and instrument responses (cache first,
                    EarthScope second), and component orientation (metadata plus known fixes, rotated to true Z/N/E)
  measure.py        the single-event measurement, mirroring the research pipeline, and re-measuring a results table
  quality.py        pre-processing gates (clipping included) and the post-processing filter, as named parameters
  reference.py      the directions fast directions are compared with (regional compression, 2009 structure)
  stats.py          axial statistics, Rayleigh test, bootstrap (event and block), permutation test, null screens
  timing.py         clock errors: catalog residual diagnostic, the chronos noise-correlation recipe, chronfix correction
  sws_functions.py  P-wave polarization, SNR and filter-bank code copied from the research repo
swspy/            local copy of SWSPy (Hudson et al.), as used by the research repo
data/catalog/     PNSN event list, station coordinates, per-station pick catalogs, channel orientations
data/waveforms/   cached miniSEED for STAR (10 batch files, about 900 events)
data/topography/  USGS 3DEP elevation of the volcano (2 arc-second GeoTIFF)
data/stationxml/  station metadata down to the instrument response (StationXML), STAR during the swarm
results/          Michael's preliminary per-station results CSVs (one row per catalog event); STAR's also re-run on
                  corrected channels (`sws_star_strict_oriented_results.csv`)
tests/            a few tests, including the reproduction of event 61504668 at STAR, the STAR component fix, STAR's instrument response and the synthetic clock recovery
.github/workflows/tests.yml   builds the environment and runs the tests on Windows, macOS and Linux on every push
```

## House rules

- Notebooks call functions in `rainier_sws/`; they do not re-implement the pipeline. Fix a bug once.
- Thresholds are named parameters (`quality.PreGates`, `quality.PostFilter`, `measure.SwsParams`, `timing.CCParams`), never literals in a notebook.
- Every number you report comes with an N and an uncertainty, and a note on which thresholds it depends on.
- Keep a dated lab notebook (a markdown file in this repo is fine) and commit at the end of every session.

## Known facts about the research results (so you do not rediscover them)

- **STAR's vertical and north channels are exchanged**: the channel labelled `EHN` records vertical
  (up-positive) motion and `EHZ` records north; `EHE` is east as labelled, and `ELZ`, the low-gain copy of
  `EHZ`, records north too. Found in October 2026 from the P wave of a regional M4.2 and from neighbouring
  stations (S. Han, "UW.STAR.01 and UW.RCS (Mount Rainier): channel orientation, amplitude, and clipping",
  2026-10-19; the north polarity is the less certain part). An earlier version of this repo treated it as a
  swap of `EHN` and `EHE`; that was wrong. The archived metadata (EarthScope, IRIS MDA) still list `EHZ` as
  the vertical. `data.get_event_stream` corrects it (`data.ORIENTATION_FIXES`), and also rotates any channel
  whose archived azimuth or dip is not nominal (`data/catalog/channel_orientations.csv`). **The STAR results
  in `results/` were computed before the exchange was found**, on the channels as labelled, so the
  pipeline's "horizontals" were the vertical and east records; see notebook 05. The research pipeline has
  since re-run STAR on waveforms rotated to true Z/N/E, with the 33 events whose S wave clipped removed
  (stage `fail_clipped`): `results/sws_star_strict_oriented_results.csv`, read with
  `paths.station_results_csv("STAR", oriented=True)`, which notebook 07 uses. With the post-processing
  filter (Q_w >= 0.75, the research repo's current cut and `quality.PostFilter`'s default; it was 0.5
  before October 2026) it keeps 194 events instead of 110, with 44 in common. The fraction of near-null
  measurements (fast axis within 20 degrees of the back azimuth alone; notebook 07 also counts its
  perpendicular, which gives 93 percent before the fix and 29 percent after, against 44 percent by chance)
  drops from 70 percent to 6 percent, and the fast directions cluster north-north-east: axial mean 34
  degrees (68 percent bootstrap interval about 1 degree either side), R = 0.86, circular standard deviation
  16 degrees, 136 of the 194 between 20 and 40 degrees. The median filtered dt goes from 0.02 to 0.06 s.
  These match the research repo's figures. RCM and MILD reproduce their original results exactly, so only
  STAR's file changed.
- **STAR's amplitudes and clip level** (same report). STAR records 15 to 18 times less than its StationXML
  response implies, on both the EH and EL channels, and its EH channels saturate at about 2.0e4 counts,
  which summit events of M1.1 to 1.6 reach (four of 35 tested). Splitting uses relative amplitudes, so the
  gain deficit does not matter here, but a clipped S wave does: screen the larger events for clipping before
  trusting a measurement. The vertical (`EHN`) has no low-gain backup. RCS (vertical only) is also low, by a
  factor of about 2 to 4, and clips at 2048 counts.
- **The measurement is deterministic but ill-conditioned.** The same input always gives the same output
  (that is why event 61504668 reproduces to the digit), but adding noise of 1e-9 counts to the waveforms
  changes Q_w for about one measured STAR event in nine, flips some fast directions by 90 degrees, and
  moves a few events in or out of the filtered set (on the archived components; on the corrected ones the same test
  changes Q_w for 27 of 677 measured events, about one in 25, moves no fast direction by 90 degrees, and adds
  4 events to the filtered set). Per-event results, and the filtered set itself, carry an
  uncertainty that the formal phi and dt errors do not show; a perturbation test belongs in the benchmark. The
  same tipping shows up between computing environments: re-measuring all 709 of Michael's successful STAR rows
  reproduces 642 in every number, 53 differ in Q_w alone and 14 in phi or dt as well, and the archived filtered
  set comes out as 114 events instead of 110 (8 change membership; notebook 05, section 2).
- **The filter bank filters causally.** `sws_functions.try_filters` calls ObsPy's bandpass with 2 corners and
  the default `zerophase=False`, which delays the S energy by roughly 30 to 50 ms relative to the pick
  (notebook 03, exercise 1). That is the size of the 0.03 s pick shift that changes Q_w (notebook 07,
  section 6). Switching to zero-phase would change every measurement, so it is a decision for the research
  pipeline, not something to change in a lesson.
- **dt comes in steps, and the formal errors are grid-limited 3-sigma bounds.** SWSPy tries a delay by
  shifting each component half of it, rounded down to whole samples, so dt comes in steps of 0.02 s at 100 Hz
  (`measure.dt_step`). phi_error and dt_error are half-widths of a 99.7 percent F-test region on a 2 degree by
  0.02 s grid: dt_error bottoms out at 0.01 s (321 of STAR's 659 successful corrected measurements) and
  phi_error tops out near 90 degrees when the region fills the grid (62). They do not shrink with SNR (rank
  correlation 0.03). Q_w is exactly 0 for 115 of the 659, which looks like a cap value rather than a score.
- **The SNR gate compares S with the P coda.** The pipeline's noise window lies between P and S (never
  before their midpoint), so at STAR, with S-P about 0.6 s, it is all P coda. The median SNR is 7, against 36
  measured against pre-event noise, and it does not depend on magnitude (notebook 03, exercise 2).
- **dt is largely a label of the filter band.** In STAR's corrected filtered set, 108 of the 136 events
  filtered at 1-5 Hz have dt 0.06 s, and 28 of the 31 at 5-10 Hz have 0.02 s. The share of 1-5 Hz events
  changes during the swarm (47 percent on 8 July, 70 on 9-10 July), so the apparent rise of dt through the
  swarm, and its fall with incidence angle, disappear within one band (notebooks 03, 06, 07). The dt <= T/2
  cut depends on the band for the same reason.
- **Nulls of local earthquakes follow the source polarization, not the back azimuth.** For a local S wave
  the initial polarization is set by the focal mechanism; the back azimuth is the teleseismic (SKS) stand-in.
  `measure.Measurement.src_pol` records SWSPy's estimate (the results CSVs do not; `measure.remeasure`
  re-measures a table at about 0.15 s per event). Of STAR's 194 filtered events, 32 lie within 20 degrees of
  their source polarization or its perpendicular and 57 of their back azimuth, only 5 of them in both (44
  percent expected by chance); without the 32 the mean fast direction moves from 33.7 to 33.5 degrees. The
  back-azimuth screen is still worth running as an artifact check: it is what exposed the channel exchange.
- **STAR's P waves look too horizontal on the corrected components.** The incidence angle estimated from
  the P-wave particle motion (Jurkevics) has a median of 61 degrees, against 27 from ray tracing, and 778 of
  901 events come out above 45. On the archived components it was 42. Something about the true vertical
  record (relative gain?) or the P window is off; not yet explained.
- **Clipping check.** `quality.PreGates.clip_fraction_max` (peak counts in the first second after S over
  2.0e4, cut at 0.97; `data.CLIP_COUNTS`) reproduces the research re-run's 33 `fail_clipped` STAR events
  exactly. It was inferred from that file, not taken from the research code. The archived results had no such
  check: reproduce them with `PreGates(clip_fraction_max=np.inf)`.
- **Reference directions.** The regional compression and the 2009 structure both point about NE
  (`rainier_sws.reference`), and STAR's corrected fast direction is about 34 degrees, so azimuth alone cannot
  separate stress from structure at STAR.
- **STAR is still analog telemetry.** Its metadata list a McVCO (analog FM telemetry) in both the 2008 to 2019
  and the current (2014 onward, location `01`) instrument chains, so its samples are timed where they are
  digitized, not on the mountain. Relevant to the clock question (notebook 04b). Other stations' July 2025
  channels are all nominal in the metadata. Older epochs are not: STAR from September 2009 to September
  2010 (the 2009 swarm) has `EHE` at azimuth 270 and `EHZ` at dip +90, which the rotation handles; whether
  the 2009 installation's vertical and north were also exchanged is not known.

- Stations PANH, OBSR, PARA, RER and COPP yield no measurements because essentially every swarm event
  arrives outside the 45-degree shear-wave window under the velocity model used for ray tracing
  (`stage == fail_incidence`). This is understood; the open questions are about the gate and the
  velocity model, not about those stations' data.
- Nearly all rays at STAR arrive from one back azimuth (about 90 degrees) and the measured fast axis is
  also about 90 degrees, so most filtered measurements there are "near-null" by geometry. Notebook 07
  explains why that is the main caveat on the preliminary result. (That fast axis comes from the exchanged
  channels. Corrected, the fast axis is about 35 degrees and under a tenth of the measurements are within 20
  degrees of the back azimuth, below chance. The back-azimuth geometry is unchanged, so one back azimuth still
  cannot by itself separate a real fast axis from a polarization effect, but the near-null concentration is
  gone.)
- The research pipeline's incidence angles come from 3-D ray tracing (PyKonal) through a Vs model
  derived as iMUSH Vp / 1.73. That code is not included here; notebook 05 uses the stored angles.
- STAR's travel-time residuals wander by about 100 ms through the swarm with P and S moving together,
  which is what a clock error looks like and what a velocity-model error does not. Unconfirmed until
  the continuous-data check in notebook 04b has been run. STAR's location code is `01` and its
  channels are short-period `EH?`.

## Sources and credits

Existing tutorials this series points to instead of duplicating:

- The Denolle Lab [cookbook](https://github.com/Denolle-Lab/Group_Utils/blob/main/cookbook.md): the lab's own list of tutorials for the shell, git, Python, conda, Jupyter, ObsPy and seismology; notebook 00 links the ones to do first.
- Software Carpentry (CC BY 4.0): [The Unix Shell](https://swcarpentry.github.io/shell-novice/), [Version Control with Git](https://swcarpentry.github.io/git-novice/), [Plotting and Programming in Python](http://swcarpentry.github.io/python-novice-gapminder/).
- ObsPy documentation and tutorial, The ObsPy Development Team, LGPL v3: [docs.obspy.org/tutorial](https://docs.obspy.org/tutorial/index.html).
  Cite Beyreuther et al. (2010), *SRL* 81(3), 530-533, doi:10.1785/gssrl.81.3.530 and Krischer et al. (2015),
  *Comput. Sci. Disc.* 8, 014003, doi:10.1088/1749-4699/8/1/014003.
- Seismo-Live notebooks, Lion Krischer and contributors, CC BY-NC-SA 4.0: [seismo-live.github.io](https://seismo-live.github.io/).
- EarthScope (NSF National Geophysical Facility) web services: [service.earthscope.org](https://service.earthscope.org/);
  cite data per [earthscope.org/how-to-cite](https://www.earthscope.org/how-to-cite/).
- IRIS/EarthScope education resources, CC BY 4.0: [iris.edu/hq/inclass](https://www.iris.edu/hq/inclass/).
- PNSN Mount Rainier page with the July 2025 swarm summary: [pnsn.org/volcanoes/mount-rainier](https://pnsn.org/volcanoes/mount-rainier); the swarm's event list: [pnsn.org/events/group/2025-mount-rainier-swarm/events](https://pnsn.org/events/group/2025-mount-rainier-swarm/events).
- USGS Cascades Volcano Observatory: [Monitoring stations detect small magnitude earthquakes at Mount Rainier during July and August 2025](https://www.usgs.gov/observatories/cvo/news/monitoring-stations-detect-small-magnitude-earthquakes-mount-rainier-during) and [Gas monitoring helps tell the story at Mount Rainier](https://www.usgs.gov/observatories/cvo/news/gas-monitoring-helps-tell-story-mount-rainier) (the numbers in the background section come from these); Sisson et al. (2017), *Geologic field-trip guide to volcanism and its interaction with snow and ice at Mount Rainier*, USGS SIR 2017-5022-A (geology and hazards background).

Software and methods:

- SWSPy: Hudson, T.S., Asplet, J., Walker, A.M., *Automated shear-wave splitting analysis for single- and
  multi-layer anisotropic media*, Seismica (2023). Code: [github.com/TomSHudson/swspy](https://github.com/TomSHudson/swspy), MIT licence.
  The copy in `swspy/` is the research project's local fork.
- chronos and chronfix: Maleen Kidiwela (UW), *Detect timing errors in seismic data via ambient-noise
  cross-correlation* and *Apply timing corrections to MiniSEED*: [github.com/MaleenKidiwela/chronos](https://github.com/MaleenKidiwela/chronos), MIT licence.
  `rainier_sws/timing.py` imports its functions when installed and carries short local copies of the numerics otherwise.
- Silver, P.G. & Chan, W.W. (1991), *JGR* 96, 16429-16454 (the grid-search method and its errors).
- Crampin, S. & Peacock, S. (2008), *Wave Motion* 45, 675-722 (review of crustal shear-wave splitting).
- Savage, M.K. et al. (2010), *JGR* 115, B12321 (MFAST: automatic measurement and quality grading).
- Baillard, C. et al. (2014), *BSSA* 104, 394-409 (the dominant period used for the cycle-skip cut).
- Brune, J.N. (1970), *JGR* 75, 4997-5009 (earthquake source spectra and the corner frequency); Hanks, T.C. & Kanamori, H. (1979), *JGR* 84, 2348-2350 (moment magnitude).
- Fisher, N.I. (1993), *Statistical Analysis of Circular Data*, Cambridge (axial statistics, bootstrap).
- Ulberg, C.W. et al. (2020), iMUSH local-earthquake tomography (the Vp model behind the research repo's ray tracing).

Background science:

- Shelly, D.R., Moran, S.C. & Thelen, W.A. (2013), *Evidence for fluid-triggered slip in the 2009 Mount Rainier, Washington earthquake swarm*, GRL 40, 1506-1512, doi:10.1002/grl.50354.
- Finn, C., Sisson, T. & Deszcz-Pan, M. (2001), *Aerogeophysical measurements of collapse-prone hydrothermally altered zones at Mount Rainier volcano*, Nature 409, 600-603.
- Chapman, M. (2003), *Frequency-dependent anisotropy due to meso-scale fractures in the presence of equant porosity*, Geophys. Prospect. 51, 369-379, doi:10.1046/j.1365-2478.2003.00384.x; Marson-Pidgeon, K. & Savage, M.K. (1997), *Frequency-dependent anisotropy in Wellington, New Zealand*, GRL 24, doi:10.1029/97GL03274 (splitting that depends on frequency; notebook 03).
- Moran, S.C., Lees, J.M. & Malone, S.D. (1999), *JGR* 104, 10775-10786 (P-wave tomography); Moran, Zimbelman & Malone (2000), *Bull. Volcanol.* 61, 425-436 (the magmatic-hydrothermal model).
- Hemmett, M.A., *Investigating the fracture network inside Mount Rainier's hydrothermal system*, research proposal and preliminary exam (UW ESS, September 2026).

Earthquake catalog and picks: Pacific Northwest Seismic Network. Waveforms: UW network via EarthScope.
