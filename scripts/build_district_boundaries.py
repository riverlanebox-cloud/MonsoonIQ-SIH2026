"""
Build data/geojson/india_districts_census2011.geojson: real boundaries for the 53
modelled districts, taken from the DataMeet Census-2011 district map.

Source : https://github.com/datameet/maps  (Districts/Census_2011/2011_Dist.*)
License: Creative Commons Attribution 2.5 India (attribution in the output file)

Matching: the polygon that contains each district's headquarters point, with
explicit name overrides where the headquarters coordinate falls just outside its
own district in the source map. Rings are simplified (Douglas-Peucker, 0.012 deg
~ 1.3 km) to keep the file small enough for the browser map.

Run:  PYTHONPATH=. python scripts/build_district_boundaries.py <path-to-2011_Dist-without-extension>
"""

import json
import sys

import numpy as np

from src.data.shapefile_lite import read_dbf, read_polygons
from src.compat import points_in_polygon

NAME_OVERRIDES = {  # district_id -> (DISTRICT, ST_NM) in the Census-2011 map
    "GA_NGA": ("North Goa", "Goa"),
    "UK_RUD": ("Rudraprayag", "Uttarakhand"),
    "JK_SRN": ("Srinagar", "Jammu & Kashmir"),
}


def dp_simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) < 5:
        return pts
    if np.allclose(pts[0], pts[-1]):  # closed ring: split at the farthest vertex
        k = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
        a = _dp_open(pts[:k + 1], tol)
        b = _dp_open(pts[k:], tol)
        out = np.vstack([a, b[1:]])
        return out if len(out) >= 4 else pts
    return _dp_open(pts, tol)


def _dp_open(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = q - p
        seg_len = np.hypot(*seg) or 1e-12
        rel = pts[a + 1:b] - p
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / seg_len
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return pts[keep]


def main(src: str, meta_path: str = "data/geojson/india_districts.geojson",
         out_path: str = "data/geojson/india_districts_census2011.geojson") -> None:
    rows = read_dbf(src + ".dbf")
    polys = read_polygons(src + ".shp")
    meta = json.load(open(meta_path, encoding="utf-8"))
    feats = []
    for ft in meta["features"]:
        p = ft["properties"]
        did = p["district_id"]
        if did in NAME_OVERRIDES:
            name, state = NAME_OVERRIDES[did]
            hit = [i for i, r in enumerate(rows) if r["DISTRICT"] == name and r["ST_NM"] == state]
        else:
            hit = [i for i, rings in enumerate(polys) if rings and
                   points_in_polygon(rings, np.array([p["centroid_lon"]]), np.array([p["centroid_lat"]]))[0]]
        if len(hit) != 1:
            raise SystemExit(f"{did}: expected one match, got {[rows[i]['DISTRICT'] for i in hit]}")
        i = hit[0]
        rings = [dp_simplify(np.asarray(r), 0.012) for r in polys[i]]
        rings = [r for r in rings if len(r) >= 4]
        coords = [[[[round(float(x), 3), round(float(y), 3)] for x, y in r]] for r in rings]
        props = dict(p)
        props.update({"census2011_name": rows[i]["DISTRICT"], "census2011_state": rows[i]["ST_NM"],
                      "census2011_code": rows[i]["censuscode"]})
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "MultiPolygon", "coordinates": coords}})
    fc = {"type": "FeatureCollection",
          "attribution": "District boundaries: DataMeet India maps (Census 2011), CC BY 2.5 India. "
                         "https://github.com/datameet/maps",
          "features": feats}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, separators=(",", ":"))
    print(f"wrote {len(feats)} districts to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
