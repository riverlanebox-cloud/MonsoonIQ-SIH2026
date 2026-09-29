"""
Download the real MonsoonIQ inputs and write the compact bundles the archive
builder reads. Needs only `requests` + NumPy; no GRIB library (src/data/grib2_lite.py).

    IMD   0.25 deg daily gridded rainfall, POST form at imdpune.gov.in (one
          binary per year), cropped to 1 Jun - 30 Sep.
    GFS   NOAA GFS 00 UTC runs from AWS Open Data (s3://noaa-gfs-bdp-pds, from
          Feb 2021). Only the needed GRIB2 messages are downloaded, via the .idx
          byte-range index:
            * APCP 0-F accumulation at F = 3, 27, 51, 75, 99, 123 on the 0.5 deg
              grid -> Day k rainfall on the IMD day = A(24k+3) - A(24k-21)
            * nine predictors at F = 24k-12 (12 UTC of lead day k) on the 1 deg
              grid: u/v/RH 850 hPa, Z/q 500 hPa, MSLP, PWAT, CAPE, TOA OLR

Output (same format the in-browser fetcher produced for the committed results):
    <out>/monsooniq_imd_jjas_<y0>-<y1>.bin.gz
    <out>/monsooniq_gfs_jjas_<year>.bin.gz

Usage:
    python scripts/fetch_real_data.py --out data/raw/bundles --years 2021 2022 2023 2024 2025

Roughly 3.5 GB is transferred for five seasons (about 6 MB per forecast run);
the written bundles total about 200 MB. Re-running skips finished files.
"""

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import struct
import sys
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.grib2_lite import decode  # noqa: E402

S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
IMD_URL = "https://www.imdpune.gov.in/cmpg/Griddata/rainfall.php"
PRED = [("UGRD", "850 mb"), ("VGRD", "850 mb"), ("RH", "850 mb"), ("HGT", "500 mb"),
        ("SPFH", "500 mb"), ("PRMSL", "mean sea level"),
        ("PWAT", "entire atmosphere (considered as a single layer)"), ("CAPE", "surface"),
        ("ULWRF", "top of atmosphere")]
# crops (row/col offsets in the global north->south, 0..360E grids)
R_J0, R_NJ, R_I0, R_NI = 108, 57, 136, 57   # 0.5 deg: 36N..8N, 68E..96E
P_J0, P_NJ, P_I0, P_NI = 50, 41, 50, 51     # 1 deg : 40N..0N, 50E..100E

session = requests.Session()


def _get(url, headers=None, text=False, tries=4):
    for a in range(tries):
        try:
            r = session.get(url, headers=headers, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text if text else r.content
        except requests.RequestException:
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"failed: {url}")


def _messages(url, matchers):
    idx = _get(url + ".idx", text=True)
    if idx is None:
        return None
    lines = [l.split(":") for l in idx.strip().split("\n")]
    out = []
    for m in matchers:
        i = next((k for k, l in enumerate(lines) if m(l)), None)
        if i is None:
            out.append(None)
            continue
        s = int(lines[i][1])
        e = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ""
        raw = _get(url, headers={"Range": f"bytes={s}-{e}"})
        out.append(decode(raw)["values"] if raw else None)
    return out


def _run(ds: str):
    base = f"{S3}gfs.{ds}/00/atmos/gfs.t00z.pgrb2."
    missing = []
    acc = {}
    for F in (3, 27, 51, 75, 99, 123):
        r = _messages(f"{base}0p50.f{F:03d}", [lambda l: l[3] == "APCP" and l[5].startswith("0-")])
        acc[F] = None if not r or r[0] is None else r[0][R_J0:R_J0 + R_NJ, R_I0:R_I0 + R_NI]
        if acc[F] is None:
            missing.append(f"{ds}:apcp:{F}")
    precip = np.full((5, R_NJ, R_NI), np.nan, np.float32)
    for L in range(1, 6):
        a, b = acc[24 * L + 3], acc[24 * L - 21]
        if a is not None and b is not None:
            precip[L - 1] = np.maximum(0.0, a - b)
    pred = np.full((5, len(PRED), P_NJ, P_NI), np.nan, np.float32)
    for L in range(1, 6):
        F = 24 * L - 12
        r = _messages(f"{base}1p00.f{F:03d}",
                      [(lambda v, lev: (lambda l: l[3] == v and l[4] == lev))(v, lev) for v, lev in PRED])
        for k in range(len(PRED)):
            if r and r[k] is not None:
                pred[L - 1, k] = r[k][P_J0:P_J0 + P_NJ, P_I0:P_I0 + P_NI]
            else:
                missing.append(f"{ds}:{PRED[k][0]}:{F}")
    return precip, pred, missing


def fetch_gfs_season(year: int, out: str, workers: int = 8):
    path = os.path.join(out, f"monsooniq_gfs_jjas_{year}.bin.gz")
    if os.path.exists(path):
        print(f"[skip] {path}")
        return
    dates = [(date(year, 6, 1) + timedelta(days=i)).strftime("%Y%m%d") for i in range(122)]
    precip = np.full((len(dates), 5, R_NJ, R_NI), np.nan, np.float32)
    pred = np.full((len(dates), 5, len(PRED), P_NJ, P_NI), np.nan, np.float32)
    missing = []
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_run, d): k for k, d in enumerate(dates)}
        for n, f in enumerate(cf.as_completed(futs), 1):
            k = futs[f]
            try:
                precip[k], pred[k], miss = f.result()
                missing += miss
            except Exception as exc:  # noqa: BLE001
                missing.append(f"{dates[k]}:run:{exc}")
            if n % 10 == 0:
                print(f"  GFS {year}: {n}/{len(dates)} runs")
    header = {"format": "MIQ-GFS-1", "source": "NOAA GFS 00Z, noaa-gfs-bdp-pds (AWS Open Data)",
              "season": year, "dates": dates,
              "precip": {"desc": "IMD-day (03-03 UTC) accumulation, lead k = f(24k+3)-f(24k-21), mm",
                         "shape": list(precip.shape), "lat_first": 36, "lat_step": -0.5,
                         "lon_first": 68, "lon_step": 0.5, "dtype": "float32"},
              "pred": {"desc": "valid 12Z of lead day (f=24k-12), 1deg",
                       "vars": [f"{v}@{lev}" for v, lev in PRED], "shape": list(pred.shape),
                       "lat_first": 40, "lat_step": -1, "lon_first": 50, "lon_step": 1,
                       "dtype": "float32"},
              "missing": missing, "fetched_utc": datetime.now(timezone.utc).isoformat()}
    _write(path, b"MIQ1", header, [precip.astype("<f4").tobytes(), pred.astype("<f4").tobytes()])


def fetch_imd(years, out: str):
    path = os.path.join(out, f"monsooniq_imd_jjas_{min(years)}-{max(years)}.bin.gz")
    if os.path.exists(path):
        print(f"[skip] {path}")
        return
    npt = 135 * 129
    parts, status = [], {}
    for y in years:
        r = session.post(IMD_URL, data={"rain": str(y)}, timeout=300)
        r.raise_for_status()
        buf = r.content
        leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
        d0 = 152 if leap else 151
        parts.append(buf[d0 * npt * 4:(d0 + 122) * npt * 4])
        status[y] = {"bytes": len(buf), "ndays": len(buf) // (npt * 4), "leap": leap,
                     "jjas_start_index": d0}
        print(f"  IMD {y}: {len(buf)} bytes")
    # GFS orography + land mask (0.5 deg, 40N..0N, 60E..100E) for slope / coast distance
    st = _messages(f"{S3}gfs.20230601/00/atmos/gfs.t00z.pgrb2.0p50.f000",
                   [lambda l: l[3] == "HGT" and l[4] == "surface",
                    lambda l: l[3] == "LAND" and l[4] == "surface"])
    crop = lambda a: [round(float(x), 2) for x in a[100:181, 120:201].ravel()]
    header = {"format": "MIQ-IMD-1",
              "source": "IMD Pune 0.25deg daily gridded rainfall (Pai et al. 2014)",
              "years": list(years), "season": "JJAS day 1 = 1 June, 122 days",
              "grid": {"nlat": 129, "nlon": 135, "lat_first": 6.5, "lon_first": 66.5,
                       "res": 0.25, "order": "lat-major, lon fastest"},
              "raw_status": status, "missing_value": -999,
              "static": {"source": "GFS 0.5deg f000 2023-06-01 00Z, HGT:surface and LAND:surface",
                         "hgt": crop(st[0]), "land": crop(st[1]),
                         "grid": {"lat_first": 40, "lat_step": -0.5, "lon_first": 60,
                                  "lon_step": 0.5, "n": 81}},
              "fetched_utc": datetime.now(timezone.utc).isoformat()}
    _write(path, b"MIQI", header, parts)


def _write(path, magic, header, parts):
    hb = json.dumps(header).encode("utf-8")
    tmp = path + ".part"
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        f.write(magic)
        f.write(struct.pack(">I", len(hb)))
        f.write(hb)
        for p in parts:
            f.write(p)
    os.replace(tmp, path)
    print(f"[ok] {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/bundles")
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fetch_imd(a.years, a.out)
    for y in a.years:
        fetch_gfs_season(y, a.out, a.workers)
