"""Repository paths. Import this instead of hard-coding '../data/...' in notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CATALOG = DATA / "catalog"
WAVEFORMS = DATA / "waveforms"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

STATIONS_CSV = CATALOG / "rainier_catalog_stations.csv"
EVENTS_CSV = CATALOG / "rainier_events.csv"
STRICT_CATALOG_CSV = CATALOG / "rainier_catalog_strict.csv"


def station_catalog_csv(station: str) -> Path:
    """Per-station strict catalog, e.g. data/catalog/rainier_catalog_strict_STAR.csv."""
    return CATALOG / f"rainier_catalog_strict_{station.upper()}.csv"


def station_waveform_dir(station: str) -> Path:
    """Directory of cached miniSEED batches for a station."""
    return WAVEFORMS / station.upper()


def station_results_csv(station: str) -> Path:
    """Michael's preliminary results for a station (one row per catalog event)."""
    return RESULTS / f"sws_{station.lower()}_strict_results.csv"


for _d in (WAVEFORMS, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
