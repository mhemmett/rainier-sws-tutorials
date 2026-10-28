"""Repository paths. Import this instead of hard-coding '../data/...' in notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CATALOG = DATA / "catalog"
WAVEFORMS = DATA / "waveforms"
TOPOGRAPHY = DATA / "topography"
STATIONXML = DATA / "stationxml"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

STATIONS_CSV = CATALOG / "rainier_catalog_stations.csv"
EVENTS_CSV = CATALOG / "rainier_events.csv"
STRICT_CATALOG_CSV = CATALOG / "rainier_catalog_strict.csv"
DEM_TIF = TOPOGRAPHY / "rainier_3dep_2arcsec.tif"
ORIENTATIONS_CSV = CATALOG / "channel_orientations.csv"


def station_catalog_csv(station: str) -> Path:
    """Per-station strict catalog, e.g. data/catalog/rainier_catalog_strict_STAR.csv."""
    return CATALOG / f"rainier_catalog_strict_{station.upper()}.csv"


def station_waveform_dir(station: str) -> Path:
    """Directory of cached miniSEED batches for a station."""
    return WAVEFORMS / station.upper()


def station_xml(station: str, network: str = "UW") -> Path:
    """StationXML (channel metadata down to the instrument response), e.g. data/stationxml/UW.STAR.xml."""
    return STATIONXML / f"{network.upper()}.{station.upper()}.xml"


def station_results_csv(station: str, oriented: bool = False) -> Path:
    """Michael's preliminary results for a station (one row per catalog event).

    ``oriented=True`` gives the re-run on waveforms rotated to true Z/N/E, with
    clipped events screened out (stage ``fail_clipped``). Only STAR's differs from
    the original, so only STAR's is in the repo.
    """
    return RESULTS / f"sws_{station.lower()}_strict{'_oriented' if oriented else ''}_results.csv"


for _d in (WAVEFORMS, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
