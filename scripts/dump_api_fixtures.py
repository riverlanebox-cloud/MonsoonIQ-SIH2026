"""
Dump API responses to JSON fixtures.

The frontend smoke test (`frontend/scripts/smoke.mjs`) renders the console in
jsdom against recorded API responses, so it can run in CI and on a laptop with
no backend process. Record the fixtures with:

    PYTHONPATH=. python scripts/dump_api_fixtures.py --date 2019-07-26 --lead 1

The fixtures are development aids and are written to `frontend/fixtures/`.
"""

import argparse
import json
import logging
import os

from fastapi.testclient import TestClient

from src.api.main import app, load_artifacts

# Fields that change on every run. They are replaced with a fixed placeholder so a
# re-recording only produces a diff when the payload actually changed, and so the
# committed fixtures do not churn on every `make fixtures`.
VOLATILE = {"built_utc", "build_seconds", "issued_utc", "loaded_utc", "elapsed_seconds",
            "sent", "effective", "expires", "onset"}  # the last four are CAP alert timestamps


def sanitize(value):
    if isinstance(value, dict):
        return {k: ("<volatile>" if k in VOLATILE else sanitize(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


ROUTES = [
    ("health", "/health"),
    ("timeline", "/timeline"),
    ("events", "/events?limit=30"),
    ("geojson", "/districts/geojson"),
    ("grid", "/grid"),
    ("verification", "/verification/summary"),
    ("heavy_events", "/verification/heavy-events"),
    ("regime_value", "/verification/regime-value"),
    ("model_card", "/model-card"),
    ("cases", "/case-replays"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="date for the per-date fixtures")
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--district", default=None, help="district id for the detail fixture")
    ap.add_argument("--out", default="frontend/fixtures")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    load_artifacts()
    client = TestClient(app)
    os.makedirs(args.out, exist_ok=True)

    console = client.get("/console", params={"date": args.date, "lead": args.lead}).json()
    date = console["date"]
    district = args.district or console["districts"][0]["district_id"]

    routes = list(ROUTES) + [
        ("console", f"/console?date={date}&lead={args.lead}"),
        ("district", f"/district/{district}?date={date}&lead={args.lead}"),
        ("bulletin", f"/bulletin?date={date}&lead={args.lead}"),
    ]

    index = {}
    for name, path in routes:
        res = client.get(path)
        if res.status_code != 200:
            print(f"skip {path}: HTTP {res.status_code}")
            continue
        with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(sanitize(res.json()), f, separators=(",", ":"))
        index[path] = name
        print(f"wrote {name}.json ({len(res.content) / 1024:.1f} KiB)")

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"paths": index, "date": date, "lead": args.lead,
                   "district": district}, f, indent=2)
    print(f"fixtures recorded for {date}, Day {args.lead}, district {district}")


if __name__ == "__main__":
    main()
