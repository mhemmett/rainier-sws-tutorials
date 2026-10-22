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
layered flows) or by the regional stress field (NNE compression from the clockwise rotation of the
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
PNSN catalog at a handful of permanent stations. It gives about 2.5 percent anisotropy and fast
directions that diverge from the regional stress and vary across the swarm, which already argues for
local control. But those numbers are station-dependent and sensitive to ray path, filtering
thresholds and software choices, and that is the honest state of the science. Your job in these
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
| 03 | `03_filtering_spectra_snr` | spectra, bandpass filters, a precise SNR, the filter bank, dominant period | 1.5 h |
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
  data.py           catalogs, station coordinates, topography, waveforms (cache first, EarthScope second),
                    and component orientation (metadata plus known fixes, rotated to true Z/N/E)
  measure.py        the single-event measurement, mirroring the research pipeline
  quality.py        pre-processing gates and the "grade 3" post-processing filter, as named parameters
  stats.py          axial statistics, Rayleigh test, bootstrap
  timing.py         clock errors: catalog residual diagnostic, the chronos noise-correlation recipe, chronfix correction
  sws_functions.py  P-wave polarization, SNR and filter-bank code copied from the research repo
swspy/            local copy of SWSPy (Hudson et al.), as used by the research repo
data/catalog/     PNSN event list, station coordinates, per-station pick catalogs, channel orientations
data/waveforms/   cached miniSEED for STAR (10 batch files, about 900 events)
data/topography/  USGS 3DEP elevation of the volcano (2 arc-second GeoTIFF)
results/          Michael's preliminary per-station results CSVs (one row per catalog event)
tests/            a few tests, including the reproduction of event 61504668 at STAR, the STAR component fix and the synthetic clock recovery
.github/workflows/tests.yml   builds the environment and runs the tests on Windows, macOS and Linux on every push
```

## House rules

- Notebooks call functions in `rainier_sws/`; they do not re-implement the pipeline. Fix a bug once.
- Thresholds are named parameters (`quality.PreGates`, `quality.Grade3`, `measure.SwsParams`, `timing.CCParams`), never literals in a notebook.
- Every number you report comes with an N and an uncertainty, and a note on which thresholds it depends on.
- Keep a dated lab notebook (a markdown file in this repo is fine) and commit at the end of every session.

## Known facts about the research results (so you do not rediscover them)

- **STAR's horizontal components are swapped**: the channel labelled `EHN` points east and `EHE` points
  north. The archived metadata (EarthScope, IRIS MDA) still say otherwise. `data.get_event_stream`
  corrects it (`data.ORIENTATION_FIXES`), and also rotates any channel whose archived azimuth or dip is
  not nominal (`data/catalog/channel_orientations.csv`). **The STAR results in `results/` were computed
  before the swap was found**, on the components as labelled; see notebook 05. Re-measured with the
  components corrected (same incidence angles and parameters), STAR's grade-3 set is 78 events instead of
  121 with only 12 in common, the fraction of near-null measurements (fast axis within 20 degrees of the back
  azimuth alone; notebook 07 also counts its perpendicular, which gives 90 percent before the fix) drops from
  about two thirds to one third, and the fast directions spread out, mostly between
  0 and 40 degrees with a smaller group near 90 (bimodal, so the axial mean of about 49 degrees is not a
  useful summary). The research results need to be re-run on corrected data before STAR is interpreted.
- **The measurement is deterministic but ill-conditioned.** The same input always gives the same output
  (that is why event 61504668 reproduces to the digit), but adding noise of 1e-9 counts to the waveforms
  changes Q_w for about one measured STAR event in nine, flips some fast directions by 90 degrees, and
  moves a few events in or out of grade 3. Per-event results, and the grade-3 set itself, carry an
  uncertainty that the formal phi and dt errors do not show; a perturbation test belongs in the benchmark. The
  same tipping shows up between computing environments: about one STAR event in ten reproduces Michael's
  row in everything except Q_w (notebook 05, exercise 1), so even the archived grade-3 count can differ by
  an event.
- **The filter bank filters causally.** `sws_functions.try_filters` calls ObsPy's bandpass with 2 corners and
  the default `zerophase=False`, which delays the S energy by roughly 30 to 50 ms relative to the pick
  (notebook 03, exercise 1). That is the size of the 0.03 s pick shift that changes Q_w (notebook 07,
  section 6). Switching to zero-phase would change every measurement, so it is a decision for the research
  pipeline, not something to change in a lesson.
- **STAR is still analog telemetry.** Its metadata list a McVCO (analog FM telemetry) in both the 2008 to 2019
  and the current (2014 onward, location `01`) instrument chains, so its samples are timed where they are
  digitized, not on the mountain. Relevant to the clock question (notebook 04b). Other stations' July 2025
  channels are all nominal in the metadata. Older epochs are not: STAR from September 2009 to September
  2010 (the 2009 swarm) has `EHE` at azimuth 270 and `EHZ` at dip +90, which the rotation handles; whether
  the 2009 horizontals were also swapped is not known.

- Stations PANH, OBSR, PARA, RER and COPP yield no measurements because essentially every swarm event
  arrives outside the 45-degree shear-wave window under the velocity model used for ray tracing
  (`stage == fail_incidence`). This is understood; the open questions are about the gate and the
  velocity model, not about those stations' data.
- Nearly all rays at STAR arrive from one back azimuth (about 90 degrees) and the measured fast axis is
  also about 90 degrees, so most grade-3 measurements there are "near-null" by geometry. Notebook 07
  explains why that is the main caveat on the preliminary result. (That fast axis comes from the swapped
  components; corrected, about a third of the measurements are near-null rather than two thirds. The
  back-azimuth geometry is unchanged, so the caveat still applies to whatever survives.)
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
- Fisher, N.I. (1993), *Statistical Analysis of Circular Data*, Cambridge (axial statistics, bootstrap).
- Ulberg, C.W. et al. (2020), iMUSH local-earthquake tomography (the Vp model behind the research repo's ray tracing).

Background science:

- Shelly, D.R., Moran, S.C. & Thelen, W.A. (2013), *Evidence for fluid-triggered slip in the 2009 Mount Rainier, Washington earthquake swarm*, GRL 40, 1506-1512, doi:10.1002/grl.50354.
- Finn, C., Sisson, T. & Deszcz-Pan, M. (2001), *Aerogeophysical measurements of collapse-prone hydrothermally altered zones at Mount Rainier volcano*, Nature 409, 600-603.
- Moran, S.C., Lees, J.M. & Malone, S.D. (1999), *JGR* 104, 10775-10786 (P-wave tomography); Moran, Zimbelman & Malone (2000), *Bull. Volcanol.* 61, 425-436 (the magmatic-hydrothermal model).
- Hemmett, M.A., *Investigating the fracture network inside Mount Rainier's hydrothermal system*, research proposal and preliminary exam (UW ESS, September 2026).

Earthquake catalog and picks: Pacific Northwest Seismic Network. Waveforms: UW network via EarthScope.
