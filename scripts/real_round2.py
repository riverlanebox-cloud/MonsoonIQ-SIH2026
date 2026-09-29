"""
Round 2 of the real-data fetch: probe which archived model has the longest Day 1-5 precipitation
record, then fetch IMD + that model + dynamics for every season it covers (>= 2021, where the
analysis-dynamics archive starts) and rebuild the archive. Everything already downloaded is cached.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.config import REAL_DIR  # noqa: E402

PREFERENCE = ["gfs_seamless", "ecmwf_ifs025", "ukmo_global_deterministic_10km", "jma_gsm",
              "icon_global", "gem_global", "cma_grapes_global"]


def run(args):
    print(">>", " ".join(args), flush=True)
    return subprocess.call([sys.executable] + args, cwd=ROOT)


def main():
    run([os.path.join(HERE, "probe_forecast_archive.py")])
    cov = json.load(open(os.path.join(REAL_DIR, "model_coverage.json"), encoding="utf-8"))
    best, best_years = None, []
    for m in PREFERENCE:
        yrs = [int(y) for y, v in cov.get(m, {}).items()
               if int(y) >= 2021 and v.get("d1", 0) >= 0.9 and v.get("d5", 0) >= 0.9]
        print(m, "covers", yrs, flush=True)
        if len(yrs) > len(best_years):
            best, best_years = m, sorted(yrs)
    if not best or len(best_years) < 3:
        print("No model covers >= 3 seasons; using what exists:", best, best_years, flush=True)
    years = [str(y) for y in best_years]
    with open(os.path.join(REAL_DIR, "chosen_model.json"), "w", encoding="utf-8") as f:
        json.dump({"model": best, "years": best_years, "coverage": cov}, f, indent=1)
    fetch = os.path.join(HERE, "fetch_real_data.py")
    run([fetch, "imd", "--realtime", "--years", *years])
    run([fetch, "forecast", "--model", best, "--years", *years])
    run([fetch, "dynamics", "--years", *years])
    run([fetch, "olr", "--years", *years])
    run([fetch, "terrain", "--years", *years])
    run([fetch, "build", "--model", best, "--years", *years])


if __name__ == "__main__":
    main()
