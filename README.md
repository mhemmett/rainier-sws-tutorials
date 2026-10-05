# Rainier shear-wave splitting tutorials

A guided, notebook-based introduction to seismic data and shear-wave splitting (SWS), built around
the Mt. Rainier July 2025 hydrothermal swarm and the preliminary results of the
[mt-rainier-swarm-catalog](https://github.com/mhemmett/mt-rainier-swarm-catalog) research project.

Written for a student with Python experience and no seismology background. The lessons start from
"what is a seismogram" and end with reproducing a published-quality splitting measurement,
putting error bars on the pooled results, and benchmarking the thresholds that produced them.

Maintained by Michael Hemmett (UW ESS, Denolle Lab).

## Setup

```bash
git clone https://github.com/mhemmett/rainier-sws-tutorials.git
cd rainier-sws-tutorials
conda env create -f environment.yml
conda activate rainier-sws
pytest -q            # about 30 s; the last test reproduces one real measurement
jupyter lab notebooks/
```

`numpy < 2` is required (see `environment.yml` for why). Everything else is standard.

The notebooks run **offline**: the seismograms they need are cached in `data/waveforms/`. With a
network connection, notebook 02 also makes live requests to EarthScope so you can see where the
data come from and fetch events that are not cached.

## The lessons

Do them in order. Each starts with a goal and a time estimate and ends with exercises.

| # | Notebook | You will learn to | Time |
| --- | --- | --- | --- |
| 00 | `00_setup_check` | check the environment and find your way around the repo | 10 min |
| 01 | `01_what_is_a_seismogram` | read miniSEED with ObsPy, plot Z/N/E with P and S picks, read a seismogram | 1 h |
| 02 | `02_getting_data_from_earthscope` | request station metadata and waveforms through FDSN web services; why we cache | 1 h |
| 03 | `03_earthquake_catalogs_and_picks` | work with event catalogs and phase picks; map the swarm; back azimuth and distance | 1.5 h |
| 04 | `04_filtering_spectra_snr` | spectra, bandpass filters, a precise SNR, the filter bank, dominant period | 1.5 h |
| 05 | `05_what_is_shear_wave_splitting` | build the splitting measurement from scratch on a synthetic signal; noise, nulls, cycle skips | 2 h |
| 06 | `06_single_event_with_swspy` | reproduce Michael's measurement of one real event to the digit with SWSPy | 1.5 h |
| 07 | `07_exploring_the_results` | read the per-station results, stage bookkeeping, histograms, rose diagrams | 1.5 h |
| 08 | `08_uncertainty_and_thresholds` | bootstrap intervals, circular statistics, threshold sweeps, null screening, a benchmark function | 2 to 3 sessions |

Suggested pacing at 3 hours a week: weeks 1 to 2 for notebooks 00 to 04, week 3 for 05, week 4 for
06, week 5 for 07, weeks 6 to 7 for 08, week 8 to turn the `benchmark` function of notebook 08 into a
script that runs on every station. The onboarding document has the full plan.

## What is in the repo

```
notebooks/        the lessons, executed, with outputs saved so you can see what to expect
rainier_sws/      helper package the notebooks import (read it; it is short)
  paths.py          where things live
  data.py           catalogs, station coordinates, waveforms (cache first, EarthScope second)
  measure.py        the single-event measurement, mirroring the research pipeline
  quality.py        pre-processing gates and the "grade 3" post-processing filter, as named parameters
  stats.py          axial statistics, Rayleigh test, bootstrap
  sws_functions.py  P-wave polarization, SNR and filter-bank code copied from the research repo
swspy/            local copy of SWSPy (Hudson et al.), as used by the research repo
data/catalog/     PNSN event list, station coordinates, per-station pick catalogs
data/waveforms/   cached miniSEED for STAR (10 batch files, about 900 events)
results/          Michael's preliminary per-station results CSVs (one row per catalog event)
tests/            a few tests, including the reproduction of event 61504668 at STAR
```

## House rules

- Notebooks call functions in `rainier_sws/`; they do not re-implement the pipeline. Fix a bug once.
- Thresholds are named parameters (`quality.PreGates`, `quality.Grade3`, `measure.SwsParams`), never literals in a notebook.
- Every number you report comes with an N and an uncertainty, and a note on which thresholds it depends on.
- Keep a dated lab notebook (a markdown file in this repo is fine) and commit at the end of every session.

## Known facts about the research results (so you do not rediscover them)

- Stations PANH, OBSR, PARA, RER and COPP yield no measurements because essentially every swarm event
  arrives outside the 45-degree shear-wave window under the velocity model used for ray tracing
  (`stage == fail_incidence`). This is understood; the open questions are about the gate and the
  velocity model, not about those stations' data.
- Nearly all rays at STAR arrive from one back azimuth (about 90 degrees) and the measured fast axis is
  also about 90 degrees, so most grade-3 measurements there are "near-null" by geometry. Notebook 08
  explains why that is the main caveat on the preliminary result.
- The research pipeline's incidence angles come from 3-D ray tracing (PyKonal) through a Vs model
  derived as iMUSH Vp / 1.73. That code is not included here; notebook 06 uses the stored angles.

## Sources and credits

Existing tutorials this series points to instead of duplicating:

- ObsPy documentation and tutorial, The ObsPy Development Team, LGPL v3: [docs.obspy.org/tutorial](https://docs.obspy.org/tutorial/index.html).
  Cite Beyreuther et al. (2010), *SRL* 81(3), 530-533, doi:10.1785/gssrl.81.3.530 and Krischer et al. (2015),
  *Comput. Sci. Disc.* 8, 014003, doi:10.1088/1749-4699/8/1/014003.
- Seismo-Live notebooks, Lion Krischer and contributors, CC BY-NC-SA 4.0: [seismo-live.github.io](https://seismo-live.github.io/).
- EarthScope (NSF National Geophysical Facility) web services: [service.earthscope.org](https://service.earthscope.org/);
  cite data per [earthscope.org/how-to-cite](https://www.earthscope.org/how-to-cite/).
- IRIS/EarthScope education resources, CC BY 4.0: [iris.edu/hq/inclass](https://www.iris.edu/hq/inclass/).
- PNSN Mount Rainier page with the July 2025 swarm summary: [pnsn.org/volcanoes/mount-rainier](https://pnsn.org/volcanoes/mount-rainier).

Software and methods:

- SWSPy: Hudson, T.S., Asplet, J., Walker, A.M., *Automated shear-wave splitting analysis for single- and
  multi-layer anisotropic media*, Seismica (2023). Code: [github.com/TomSHudson/swspy](https://github.com/TomSHudson/swspy), MIT licence.
  The copy in `swspy/` is the research project's local fork.
- Silver, P.G. & Chan, W.W. (1991), *JGR* 96, 16429-16454 (the grid-search method and its errors).
- Crampin, S. & Peacock, S. (2008), *Wave Motion* 45, 675-722 (review of crustal shear-wave splitting).
- Savage, M.K. et al. (2010), *JGR* 115, B12321 (MFAST: automatic measurement and quality grading).
- Fisher, N.I. (1993), *Statistical Analysis of Circular Data*, Cambridge (axial statistics, bootstrap).
- Ulberg, C.W. et al. (2020), iMUSH local-earthquake tomography (the Vp model behind the research repo's ray tracing).

Earthquake catalog and picks: Pacific Northwest Seismic Network. Waveforms: UW network via EarthScope.
