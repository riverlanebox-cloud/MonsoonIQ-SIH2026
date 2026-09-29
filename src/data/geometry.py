"""
Dependency-light polygon utilities (numpy only).

Shapely is used when installed; these functions are the fallback and are also what the
real-data builder uses, so the IMD-grid -> district aggregation runs on any machine that has
numpy. GeoJSON Polygon and MultiPolygon (with holes) are supported.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

Ring = np.ndarray            # (n, 2) lon/lat
PolygonRings = List[Ring]    # exterior first, then holes


def geometry_polygons(geom: dict) -> List[PolygonRings]:
    """GeoJSON geometry -> list of polygons, each a list of rings (exterior, holes...)."""
    t = geom["type"]
    if t == "Polygon":
        return [[np.asarray(r, dtype=float)[:, :2] for r in geom["coordinates"]]]
    if t == "MultiPolygon":
        return [[np.asarray(r, dtype=float)[:, :2] for r in poly] for poly in geom["coordinates"]]
    raise ValueError(f"unsupported geometry type {t}")


def _points_in_ring(x: np.ndarray, y: np.ndarray, ring: Ring) -> np.ndarray:
    """Even-odd ray casting, vectorised over points."""
    inside = np.zeros(x.shape, dtype=bool)
    xs, ys = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    for xi_, yi_, xj_, yj_ in zip(xs, ys, xj, yj):
        cond = (yi_ > y) != (yj_ > y)
        if not cond.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj_ - xi_) * (y - yi_) / (yj_ - yi_) + xi_
        inside ^= cond & (x < xint)
    return inside


def contains(geom: dict, x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Boolean mask of points (x=lon, y=lat) inside a GeoJSON geometry."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros(x.shape, dtype=bool)
    for rings in geometry_polygons(geom):
        ext = rings[0]
        bb = (x >= ext[:, 0].min()) & (x <= ext[:, 0].max()) & (y >= ext[:, 1].min()) & (y <= ext[:, 1].max())
        if not bb.any():
            continue
        sub = _points_in_ring(x[bb], y[bb], ext)
        for hole in rings[1:]:
            sub &= ~_points_in_ring(x[bb], y[bb], hole)
        out[bb] |= sub
    return out


def bounds(geom: dict) -> Tuple[float, float, float, float]:
    pts = np.vstack([r for poly in geometry_polygons(geom) for r in poly])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def ring_area(ring: Ring) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def centroid(geom: dict) -> Tuple[float, float]:
    """Area-weighted centroid (lon, lat) of the exterior rings minus holes."""
    ax = ay = at = 0.0
    for rings in geometry_polygons(geom):
        for k, r in enumerate(rings):
            x, y = r[:, 0], r[:, 1]
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cross = x * y1 - x1 * y
            a = 0.5 * cross.sum()
            if abs(a) < 1e-12:
                continue
            cx = ((x + x1) * cross).sum() / (6 * a)
            cy = ((y + y1) * cross).sum() / (6 * a)
            sign = 1.0 if k == 0 else -1.0
            aa = sign * abs(a)
            ax += cx * aa
            ay += cy * aa
            at += aa
    if at == 0:
        b = bounds(geom)
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ax / at, ay / at


def representative_point(geom: dict) -> Tuple[float, float]:
    """A point guaranteed inside the geometry (centroid if inside, else nearest interior sample)."""
    cx, cy = centroid(geom)
    if contains(geom, [cx], [cy])[0]:
        return cx, cy
    x0, y0, x1, y1 = bounds(geom)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 40), np.linspace(y0, y1, 40))
    m = contains(geom, gx.ravel(), gy.ravel())
    if not m.any():
        return cx, cy
    px, py = gx.ravel()[m], gy.ravel()[m]
    k = int(np.argmin((px - cx) ** 2 + (py - cy) ** 2))
    return float(px[k]), float(py[k])


def cell_weights(geom: dict, lats: np.ndarray, lons: np.ndarray, cell: float,
                 sub: int = 5) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Fractional overlap of each grid cell with the polygon, estimated with sub x sub
    sample points per cell (sub=5 -> 4% area resolution). Returns (indices, weights),
    weights normalised to sum to one; nearest cell if the polygon is smaller than a cell.
    """
    x0, y0, x1, y1 = bounds(geom)
    li = np.where((lats >= y0 - cell) & (lats <= y1 + cell))[0]
    lj = np.where((lons >= x0 - cell) & (lons <= x1 + cell))[0]
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    idx, w = [], []
    if li.size and lj.size:
        LI, LJ = np.meshgrid(li, lj, indexing="ij")
        LI, LJ = LI.ravel(), LJ.ravel()
        ox, oy = np.meshgrid(offs * cell, offs * cell)
        px = (lons[LJ][:, None] + ox.ravel()[None, :]).ravel()
        py = (lats[LI][:, None] + oy.ravel()[None, :]).ravel()
        inside = contains(geom, px, py).reshape(LI.size, -1).mean(axis=1)
        keep = inside > 0
        idx = list(zip(LI[keep].tolist(), LJ[keep].tolist()))
        w = inside[keep].tolist()
    if not idx:
        cx, cy = centroid(geom)
        idx = [(int(np.argmin(np.abs(lats - cy))), int(np.argmin(np.abs(lons - cx))))]
        w = [1.0]
    w = np.asarray(w, dtype=np.float32)
    return idx, w / w.sum()


def simplify_ring(ring: Ring, tol: float) -> Ring:
    """Douglas-Peucker on a closed ring (keeps closure, never drops below 4 points)."""
    pts = np.asarray(ring, dtype=float)
    if len(pts) <= 4:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = pts[a + 1:b]
        d = q - p
        n = np.hypot(*d)
        if n == 0:
            dist = np.hypot(*(seg - p).T)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / n
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    out = pts[keep]
    if len(out) < 4:
        step = max(1, len(pts) // 4)
        out = np.vstack([pts[::step][:3], pts[:1]])
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def iter_features(geojson: dict) -> Iterable[dict]:
    return geojson.get("features", [])
