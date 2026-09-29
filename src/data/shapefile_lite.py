"""
Minimal ESRI shapefile + dBASE reader (polygon shapes only), standard library + NumPy.

Used once, offline, to derive real district boundaries for the 53 modelled
districts from the DataMeet Census-2011 district map (CC BY 2.5 India). Kept
dependency-free so the build step runs where geopandas/pyshp are unavailable.
"""

import struct
from typing import Dict, List, Tuple
import numpy as np


def read_dbf(path: str) -> List[Dict[str, str]]:
    with open(path, "rb") as f:
        head = f.read(32)
        nrec, hlen, rlen = struct.unpack("<IHH", head[4:12])
        fields = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            name = d[:11].split(b"\x00")[0].decode("latin-1")
            fields.append((name, chr(d[11]), d[16]))
        f.seek(hlen)
        rows = []
        for _ in range(nrec):
            rec = f.read(rlen)
            pos, row = 1, {}
            for name, _typ, size in fields:
                row[name] = rec[pos:pos + size].decode("latin-1", "replace").strip()
                pos += size
            rows.append(row)
    return rows


def read_polygons(path: str) -> List[List[np.ndarray]]:
    """Return, per record, a list of rings (N x 2 arrays of lon, lat)."""
    out: List[List[np.ndarray]] = []
    with open(path, "rb") as f:
        data = f.read()
    pos = 100
    while pos < len(data):
        _rec, clen = struct.unpack(">ii", data[pos:pos + 8])
        body = data[pos + 8: pos + 8 + clen * 2]
        pos += 8 + clen * 2
        stype = struct.unpack("<i", body[:4])[0]
        if stype == 0:
            out.append([])
            continue
        if stype not in (5, 15, 25):
            raise ValueError(f"unsupported shape type {stype}")
        nparts, npts = struct.unpack("<ii", body[36:44])
        parts = list(struct.unpack(f"<{nparts}i", body[44:44 + 4 * nparts])) + [npts]
        pts = np.frombuffer(body[44 + 4 * nparts: 44 + 4 * nparts + 16 * npts], dtype="<f8").reshape(-1, 2)
        out.append([pts[parts[i]:parts[i + 1]] for i in range(nparts)])
    return out


def rings_contain(rings: List[np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon over all rings (holes handled by parity)."""
    inside = np.zeros(x.shape, dtype=bool)
    for ring in rings:
        xr, yr = ring[:, 0], ring[:, 1]
        x0, y0 = xr, yr
        x1, y1 = np.roll(xr, -1), np.roll(yr, -1)
        for a, b, c, d in zip(x0, y0, x1, y1):
            if b == d:
                continue
            cond = (b > y) != (d > y)
            xint = (c - a) * (y - b) / (d - b) + a
            inside ^= cond & (x < xint)
    return inside
