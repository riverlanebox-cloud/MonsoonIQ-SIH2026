"""
Which archived NWP models actually have Day 1-5 precipitation for which monsoon seasons?

Open-Meteo's Previous Runs archive depth differs per model and per variable (most models
start in 2024). This asks for three July days per year at one point (Mumbai) for each
candidate model and reports the fraction of non-missing hourly values, so the fetch can
pick the model with the longest real forecast archive. Output: data/real/model_coverage.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import REAL_DIR  # noqa: E402
from src.data.real.openmeteo import PREVIOUS_RUNS_URL, OpenMeteoClient  # noqa: E402

MODELS = ["gfs_seamless", "jma_gsm", "ecmwf_ifs025", "ukmo_global_deterministic_10km",
          "icon_global", "gem_global", "cma_grapes_global"]
YEARS = list(range(2018, 2026))


def main():
    client = OpenMeteoClient(cache_dir=os.path.join(REAL_DIR, "raw", "openmeteo_cache"))
    out = {}
    for m in MODELS:
        out[m] = {}
        for y in YEARS:
            try:
                d = client.get(PREVIOUS_RUNS_URL, {
                    "latitude": "19.0760", "longitude": "72.8777",
                    "hourly": "precipitation_previous_day1,precipitation_previous_day5",
                    "start_date": f"{y}-07-01", "end_date": f"{y}-07-03", "timezone": "GMT",
                    "models": m})
                h = d.get("hourly", {})
                v1 = h.get("precipitation_previous_day1") or []
                v5 = h.get("precipitation_previous_day5") or []
                out[m][y] = {"d1": round(sum(x is not None for x in v1) / max(1, len(v1)), 2),
                             "d5": round(sum(x is not None for x in v5) / max(1, len(v5)), 2)}
            except Exception as exc:  # unknown model id, etc.
                out[m][y] = {"error": str(exc)[:120]}
        print(m, out[m], flush=True)
    os.makedirs(REAL_DIR, exist_ok=True)
    with open(os.path.join(REAL_DIR, "model_coverage.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
