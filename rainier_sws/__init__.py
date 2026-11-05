"""rainier_sws: small helper package for the Mt. Rainier shear-wave splitting tutorials.

Everything here is a thin, readable wrapper around ObsPy, SWSPy and the
functions in ``sws_functions.py`` (ported from the mt-rainier research repo).
The notebooks import from this package so that the same code path is used in
every lesson, and so that you can read one short file to see what a step does.

Modules
-------
paths     : where data, results and figures live (relative to the repo root)
data      : loading catalogs, station coordinates and waveforms (cache first,
            EarthScope second)
measure   : the single-event splitting measurement, step by step
quality   : the pre-processing gates, the post-processing filter and the stage bookkeeping
stats     : circular statistics, bootstrap intervals and permutation tests for phi and dt
reference : the directions fast directions are compared with (regional stress, 2009 structure)
timing    : clock errors: catalog residual diagnostic, chronos noise-correlation
            recipe, chronfix correction (notebook 04b)
benchmark : the week-8 benchmark, one report per results file (notebook 07)
plots     : the splitting diagnostic figure (axial-sws methods style) and the shaded-relief map
raytrace  : PyKonal S-wave rays through the Rainier velocity model: travel time and incidence angle (notebook 04)
"""
from . import paths, data, measure, quality, reference, stats, timing, benchmark, plots  # noqa: F401

__version__ = "0.1.0"
