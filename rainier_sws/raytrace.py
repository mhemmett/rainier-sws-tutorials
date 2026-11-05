"""S-wave rays from an earthquake to a station: PyKonal through the Rainier S-wave velocity model.

The same method as the research pipeline (mt-rainier-swarm-catalog, ``scripts/pykonal_raytracer.py``,
itself ported from axial-sws's PyKonal tracer): for each station the eikonal equation is solved once by
PyKonal's fast-marching method with the station as the source, and every earthquake is then read off
that travel-time field. The field at the hypocenter is the S travel time T_S; following its gradient
back to the station gives the ray, whose last segment gives the incidence angle at the station.

The model is Vs = iMUSH Vp / 1.73 (Ulberg et al., 2020) on a 1.2 km grid, cropped to the summit
(``data/velocity/rainier_vs_imush_vp173_summit.npz``). Coordinates are local km: x east and y north of
the summit, z km below sea level (positive down), so a station at elevation h sits at z = -h. The
station is placed on the nearest grid node, as in the research pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import paths

VS_MODEL = paths.DATA / "velocity" / "rainier_vs_imush_vp173_summit.npz"
REF_LAT, REF_LON = 46.8523, -121.7603          # the summit: the origin of the local x, y
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(REF_LAT))


def ll2xy(lat, lon):
    """Local (x east, y north) km relative to the summit."""
    return (np.asarray(lon) - REF_LON) * KM_PER_DEG_LON, (np.asarray(lat) - REF_LAT) * KM_PER_DEG_LAT


def load_vs_model(path=VS_MODEL):
    """(x_km, y_km, z_km, vs) with vs indexed [x, y, z] (PyKonal's order); the model's few empty nodes are
    filled from their nearest neighbour, as the research pipeline does."""
    m = np.load(path)
    vs = m["vs_km_s"]
    nan = np.isnan(vs)
    if nan.any():
        vs = vs[tuple(distance_transform_edt(nan, return_distances=False, return_indices=True))]
    x_km, _ = ll2xy(REF_LAT, m["longitude"])           # in the file's float32, as the research code computes them
    _, y_km = ll2xy(m["latitude"], REF_LON)
    return (x_km.astype(np.float64), y_km.astype(np.float64), m["depth_km"].astype(np.float64),
            np.transpose(vs, (2, 1, 0)).astype(np.float64))


class StationRayTracer:
    """Solve the eikonal equation once per station (the station is the source), then read any
    earthquake's travel time and ray off the cached field."""

    def __init__(self, model=None):
        import pykonal
        self._pykonal = pykonal
        self.x, self.y, self.z, self.vs = model if model is not None else load_vs_model()
        self._d = (self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0])
        self._o = (self.x[0], self.y[0], self.z[0])
        self._tt, self._sta = {}, {}

    def _node(self, x, y, z):
        return tuple(int(round((v - o) / d)) for v, o, d in zip((x, y, z), self._o, self._d))

    def precompute_station(self, name: str, lat: float, lon: float, elevation_m: float):
        """Solve the travel-time field from a station (a few seconds)."""
        sx, sy = ll2xy(lat, lon)
        solver = self._pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = self._o
        solver.velocity.node_intervals = self._d
        solver.velocity.npts = self.vs.shape
        solver.velocity.values = self.vs.copy()
        ix, iy, iz = self._node(float(sx), float(sy), -elevation_m / 1000.0)
        solver.traveltime.values[:] = np.inf
        solver.unknown[:] = True
        solver.traveltime.values[ix, iy, iz] = 0.0
        solver.unknown[ix, iy, iz] = False
        solver.trial.push(ix, iy, iz)
        solver.solve()
        self._tt[name] = solver.traveltime
        self._sta[name] = np.array([self.x[ix], self.y[iy], self.z[iz]])   # the grid node the rays end at

    def station_xyz(self, name: str) -> np.ndarray:
        """The grid node used as the station's position (km)."""
        return self._sta[name]

    def trace(self, name: str, lat: float, lon: float, depth_km_bsl: float) -> dict:
        """T_S (s), the incidence angle at the station (deg from vertical) and the ray (n x 3, km)."""
        ex, ey = ll2xy(lat, lon)
        eq = np.clip([float(ex), float(ey), float(depth_km_bsl)], [self.x[0], self.y[0], self.z[0]],
                     [self.x[-1], self.y[-1], self.z[-1]])
        field = self._tt[name]
        t_s = float(field.value(np.asarray(eq)))
        ray = field.trace_ray(np.asarray(eq))
        sta = self._sta[name]
        if np.linalg.norm(ray[0] - sta) <= np.linalg.norm(ray[-1] - sta):   # the end of the ray at the station
            p_sta, p_next = ray[0], ray[1]
        else:
            p_sta, p_next = ray[-1], ray[-2]
        d = p_next - p_sta
        return {"travel_time_s": t_s, "incidence_deg": float(np.degrees(np.arctan2(np.hypot(d[0], d[1]), abs(d[2])))),
                "ray": ray}
