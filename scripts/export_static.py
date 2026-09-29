"""
Export a static snapshot of the MonsoonIQ API for serverless hosting (Vercel).

The live API runs the trained models on every request. Hosts such as Vercel cap a
serverless function well below the size of numpy + pandas + scipy + scikit-learn
+ the models, so for static hosting we run every request the console can make
once, at build time, and write the answers as files:

    <out>/index.json                  dates in the snapshot + default date
    <out>/health.json, timeline.json, events.json, model-card.json, ...
    <out>/day/<date>.json.gz          console, district detail, bulletin (en/hi)
                                      and CSV for leads 1-5, one file per day

`frontend/src/api.js` reads these files when built with VITE_STATIC=1. The
payloads are produced by the same handler functions the live API serves, so a
static page and the live API show identical numbers.

    PYTHONPATH=. python scripts/export_static.py --out frontend/public/snapshot
"""

import argparse
import gzip
import json
import logging
import os
import shutil
import sys
import time

# One thread per process: the export parallelises over days, and OpenMP thread
# pools inherited across fork() spin against each other.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api import main as api  # noqa: E402

LEADS = (1, 2, 3, 4, 5)


def _plain(obj):
    """FastAPI handlers may return pydantic models or numpy scalars."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict") and not isinstance(obj, dict):
        return obj.dict()
    return obj


def _dump(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_plain(payload), f, ensure_ascii=False, separators=(",", ":"), default=_default)


def _default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _try(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as exc:  # a missing artifact must not fail the whole build
        logging.warning("skipped %s: %s", getattr(fn, "__name__", fn), exc)
        return None


def _export_day(job):
    """Every per-day request the console can make, for leads 1-5, as one gzip file."""
    date, path = job
    day_ids = api.STATE["df"].loc[api.STATE["df"]["date"] == date, "district_id"].tolist()
    day = {"date": date, "leads": {}}
    for lead in LEADS:
        entry = {
            "console": _try(api.console, date=date, lead=lead, horizon=True),
            "district": {did: _try(api.district_detail, did, date=date, lead=lead)
                         for did in day_ids},
            "bulletin": {lang: _try(api.bulletin, date=date, lead=lead, lang=lang,
                                    only_warned=True) for lang in ("en", "hi")},
        }
        csv_resp = _try(api.export_districts, date=date, lead=lead)
        body = getattr(csv_resp, "body", None)
        entry["csv"] = body.decode("utf-8") if isinstance(body, bytes) else body
        day["leads"][str(lead)] = entry
    raw = json.dumps(_plain(day), ensure_ascii=False, separators=(",", ":"),
                     default=_default).encode("utf-8")
    with open(path, "wb") as f:
        f.write(gzip.compress(raw, 9, mtime=0))
    return date


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frontend/public/snapshot")
    ap.add_argument("--dates", default="all", choices=["all", "test"],
                    help="all timeline days, or only the held-out test seasons")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: CPUs, max 8)")
    ap.add_argument("--limit", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    t0 = time.time()
    api.load_artifacts()
    out = args.out
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # ---- shared payloads (one file each)
    shared = {
        "health.json": api.health,
        "model-card.json": api.model_card,
        "timeline.json": api.timeline,
        "events.json": lambda: api.events(limit=40),
        "grid.json": lambda: api.grid_fields(date=None),
        "verification/summary.json": api.verification_summary,
        "verification/heavy-events.json": api.verification_heavy_events,
        "verification/regime-value.json": api.verification_regime_value,
        "case-replays.json": api.case_replays,
        "districts.json": api.list_districts,
        "districts/geojson.json": api.districts_geojson,
    }
    for name, fn in shared.items():
        payload = _try(fn)
        if payload is not None:
            _dump(os.path.join(out, name), payload)
    if os.path.exists(api.PDF_PATH):
        os.makedirs(os.path.join(out, "verification"), exist_ok=True)
        shutil.copy(api.PDF_PATH, os.path.join(out, "verification", "report.pdf"))
    spec = _try(api.app.openapi) if hasattr(api.app, "openapi") else None
    if spec:
        _dump(os.path.join(out, "openapi.json"), spec)

    # ---- per-day payloads
    timeline_dates = [d["date"] for d in (api.STATE["timeline"] or [])]
    available = set(api._available_dates())
    dates = [d for d in timeline_dates if d in available] or sorted(available)
    if args.dates == "test":
        from src import config
        dates = [d for d in dates if int(d[:4]) in config.TEST_YEARS]
    default_date = api._resolve_date(None)
    meta_default = (api.STATE["console_meta"] or {}).get("default_date")
    for d in (default_date, meta_default):
        if d and d in available and d not in dates:
            dates.append(d)
    dates = sorted(set(dates))

    district_ids = sorted(api.STATE["df"]["district_id"].unique().tolist())
    if args.limit:
        dates = dates[:args.limit]
    os.makedirs(os.path.join(out, "day"), exist_ok=True)
    jobs = [(date, os.path.join(out, "day", f"{date}.json.gz")) for date in dates]
    workers = max(1, min(args.workers or (os.cpu_count() or 1), 8))
    if workers > 1:
        import multiprocessing as mp
        # fork after load_artifacts so every worker shares the loaded models
        with mp.get_context("fork").Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_export_day, jobs, chunksize=4)):
                if (i + 1) % 50 == 0 or i == len(jobs) - 1:
                    print(f"  {i + 1}/{len(jobs)} days  ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, job in enumerate(jobs):
            _export_day(job)
            if (i + 1) % 50 == 0 or i == len(jobs) - 1:
                print(f"  {i + 1}/{len(jobs)} days  ({time.time() - t0:.0f}s)", flush=True)

    _dump(os.path.join(out, "index.json"), {
        "snapshot": True,
        "dates": dates,
        "default_date": default_date,
        "built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "districts": len(district_ids),
    })
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
    print(f"static snapshot: {len(dates)} days, {size / 1e6:.1f} MB in {out} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
