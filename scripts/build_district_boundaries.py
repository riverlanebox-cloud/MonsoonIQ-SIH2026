"""
Build the district boundary layers from real, citable sources.

Inputs (fetched automatically if absent):
  * DataMeet "Districts of India - Census 2011" shapefile (641 districts, names and extent
    from the Census of India Administrative Atlas). CC BY 2.5 IN.
    https://github.com/datameet/maps/tree/master/Districts/Census_2011
  * Natural Earth 1:10m coastline (public domain) - used only for distance-to-coast.
    https://github.com/nvkelso/natural-earth-vector

Outputs (committed, so nobody needs to run this for the demo):
  data/geojson/india_districts_census2011.geojson   all 641 districts, simplified to ~1 km
  data/geojson/india_districts.geojson              the 53-district study set, same polygons,
                                                    original study properties kept
  data/geojson/india_coastline.geojson              coastline clipped to the India box

Usage:  PYTHONPATH=. python scripts/build_district_boundaries.py [--shp path/to/2011_Dist]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import geometry as G  # noqa: E402

STUDY_IN = "data/geojson/synthetic_study_districts.geojson"   # original study set (ids, zones)
STUDY_OUT = "data/geojson/india_districts.geojson"
ALL_OUT = "data/geojson/india_districts_census2011.geojson"
COAST_OUT = "data/geojson/india_coastline.geojson"
BOX = (66.0, 5.0, 99.0, 38.5)
SIMPLIFY_DEG = 0.008   # ~0.9 km: invisible at district zoom, keeps the file < 3 MB

# Study-set names that differ from the Census 2011 spelling.
NAME_MAP = {
    "MH_MUM": ("Mumbai", "Maharashtra"), "MH_SAT": ("Satara", "Maharashtra"),
    "KA_BLR": ("Bangalore", "Karnataka"), "HP_KUL": ("Kullu", "Himachal Pradesh"),
    "OD_BBS": ("Khordha", "Odisha"), "OD_BAL": ("Baleshwar", "Odisha"),
    "GJ_AMD": ("Ahmadabad", "Gujarat"), "GJ_KCH": ("Kachchh", "Gujarat"),
    "AS_GHY": ("Kamrup Metropolitan", "Assam"), "ML_SHL": ("East Khasi Hills", "Meghalaya"),
    "AP_VJA": ("Krishna", "Andhra Pradesh"), "KL_EKM": ("Ernakulam", "Kerala"),
    "UK_RUD": ("Rudraprayag", "Uttarakhand"), "UP_GKP": ("Gorakhpur", "Uttar Pradesh"),
    "WB_24P": ("South 24 Parganas", "West Bengal"), "AR_ITA": ("Papum Pare", "Arunanchal Pradesh"),
    "TS_HYD": ("Hyderabad", "Andhra Pradesh"),   # Census 2011 predates Telangana
}


def _sparse_clone(repo: str, path: str, dest: str) -> str:
    subprocess.check_call(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                           f"https://github.com/{repo}.git", dest])
    subprocess.check_call(["git", "-C", dest, "sparse-checkout", "set", "--no-cone", path])
    return os.path.join(dest, path.lstrip("/"))


def _slug(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", s.upper())[:6]


def _shape_to_geojson(shape) -> dict:
    pts = np.asarray(shape.points, dtype=float)
    parts = list(shape.parts) + [len(pts)]
    rings = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    # Shapefile: exterior rings are clockwise, holes counter-clockwise.
    polys = []
    for r in rings:
        if len(r) < 4:
            continue
        if G.ring_area(r) <= 0 or not polys:      # clockwise (negative shoelace) -> new exterior
            polys.append([r])
        else:
            polys[-1].append(r)
    return {"type": "MultiPolygon", "coordinates": [[ring.tolist() for ring in p] for p in polys]}


def _simplify(geom: dict, tol: float) -> dict:
    out = []
    for poly in G.geometry_polygons(geom):
        ext = G.simplify_ring(poly[0], tol)
        if abs(G.ring_area(ext)) < (tol * tol * 4) and len(G.geometry_polygons(geom)) > 1:
            continue   # drop specks (tiny islets) but never the only polygon
        rings = [np.round(ext, 4).tolist()]
        for hole in poly[1:]:
            h = G.simplify_ring(hole, tol)
            if abs(G.ring_area(h)) > tol * tol * 4:
                rings.append(np.round(h, 4).tolist())
        out.append(rings)
    if not out:
        out = [[np.round(G.geometry_polygons(geom)[0][0], 4).tolist()]]
    return {"type": "MultiPolygon", "coordinates": out} if len(out) > 1 else {"type": "Polygon", "coordinates": out[0]}


def _coast_points(coast_path: str) -> np.ndarray:
    with open(coast_path, "r", encoding="utf-8") as f:
        coast = json.load(f)
    segs = []
    feats = []
    for feat in coast["features"]:
        g = feat["geometry"]
        lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        keep = []
        for line in lines:
            a = np.asarray(line, dtype=float)
            m = (a[:, 0] >= BOX[0]) & (a[:, 0] <= BOX[2]) & (a[:, 1] >= BOX[1]) & (a[:, 1] <= BOX[3])
            if m.sum() >= 2:
                a = a[m]
                keep.append(np.round(a, 3).tolist())
                # densify to <= 2 km spacing so nearest-vertex distance is accurate
                for p, q in zip(a[:-1], a[1:]):
                    n = max(1, int(np.hypot(*(q - p)) / 0.02))
                    t = np.linspace(0, 1, n, endpoint=False)[:, None]
                    segs.append(p + t * (q - p))
        if keep:
            feats.append({"type": "Feature", "properties": {},
                          "geometry": {"type": "MultiLineString", "coordinates": keep}})
    with open(COAST_OUT, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "name": "Natural Earth 10m coastline (India box)",
                   "features": feats}, f, separators=(",", ":"))
    return np.vstack(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", default=None, help="path to 2011_Dist (without extension)")
    ap.add_argument("--coast", default=None, help="path to ne_10m_coastline.geojson")
    args = ap.parse_args()

    import shapefile  # pyshp

    tmp = tempfile.mkdtemp(prefix="miq_bounds_")
    shp = args.shp or os.path.join(_sparse_clone("datameet/maps", "/Districts/Census_2011",
                                                 os.path.join(tmp, "dm")), "2011_Dist")
    coast = args.coast or _sparse_clone("nvkelso/natural-earth-vector",
                                        "/geojson/ne_10m_coastline.geojson", os.path.join(tmp, "ne"))
    coast_pts = _coast_points(coast)

    reader = shapefile.Reader(shp)
    census = []
    for sr in reader.iterShapeRecords():
        name, state, st_code, dt_code, cen = sr.record[:5]
        geom = _shape_to_geojson(sr.shape)
        rlon, rlat = G.representative_point(geom)
        d = G.haversine_km(rlat, rlon, coast_pts[:, 1], coast_pts[:, 0]).min()
        census.append({
            "name": name.strip(), "state": state.strip().replace("Arunanchal", "Arunachal"),
            "state_raw": state.strip(), "census_code": int(cen), "geom": geom,
            "rep": (round(rlat, 4), round(rlon, 4)), "dist_coast_km": round(float(d), 1),
        })

    # 1. all 641 districts
    feats = []
    for c in census:
        cx, cy = G.centroid(c["geom"])
        feats.append({"type": "Feature", "properties": {
            "district_id": f"C11_{c['census_code']}",
            "district_name": c["name"], "state_name": c["state"], "census_code": c["census_code"],
            "centroid_lat": round(cy, 4), "centroid_lon": round(cx, 4),
            "rep_lat": c["rep"][0], "rep_lon": c["rep"][1], "dist_coast_km": c["dist_coast_km"],
        }, "geometry": _simplify(c["geom"], SIMPLIFY_DEG)})
    with open(ALL_OUT, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection",
                   "name": "Districts of India, Census 2011 (DataMeet, CC BY 2.5 IN)",
                   "features": feats}, f, separators=(",", ":"))

    # 2. study set: same polygons, original ids / zones / elevation kept
    with open(STUDY_IN, "r", encoding="utf-8") as f:
        study = json.load(f)
    by_name = {(c["name"].lower(), c["state_raw"].lower()): c for c in census}
    by_only = {}
    for c in census:
        by_only.setdefault(c["name"].lower(), []).append(c)
    out, missing = [], []
    for feat in study["features"]:
        p = dict(feat["properties"])
        name, state = NAME_MAP.get(p["district_id"], (p["district_name"], p["state_name"]))
        c = by_name.get((name.lower(), state.lower()))
        if c is None and len(by_only.get(name.lower(), [])) == 1:
            c = by_only[name.lower()][0]
        if c is None:
            missing.append(p["district_id"])
            continue
        p.update({"census_code": c["census_code"], "census_name": c["name"],
                  "rep_lat": c["rep"][0], "rep_lon": c["rep"][1],
                  "dist_coast_km": c["dist_coast_km"], "geometry_source": "Census 2011 (DataMeet)"})
        out.append({"type": "Feature", "properties": p, "geometry": _simplify(c["geom"], SIMPLIFY_DEG / 2)})
    if missing:
        raise SystemExit(f"unmatched study districts: {missing}")
    with open(STUDY_OUT, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection",
                   "name": "MonsoonIQ study districts - Census 2011 polygons (DataMeet, CC BY 2.5 IN)",
                   "features": out}, f, separators=(",", ":"))
    print(f"wrote {ALL_OUT} ({len(feats)} districts, {os.path.getsize(ALL_OUT)/1e6:.1f} MB), "
          f"{STUDY_OUT} ({len(out)}), {COAST_OUT}")


if __name__ == "__main__":
    main()
