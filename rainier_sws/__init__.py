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
quality   : the "grade 3" acceptance filter and the stage bookkeeping
stats     : circular statistics and bootstrap intervals for phi and dt
"""
from . import paths, data, measure, quality, stats  # noqa: F401

__version__ = "0.1.0"
