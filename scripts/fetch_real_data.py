"""
Fetch real observations + forecasts and build the real MonsoonIQ archive.

    PYTHONPATH=. python scripts/fetch_real_data.py all            # every stage, from config
    PYTHONPATH=. python scripts/fetch_real_data.py imd            # IMD 0.25 deg rainfall only
    PYTHONPATH=. python scripts/fetch_real_data.py forecast       # Open-Meteo Previous Runs
    PYTHONPATH=. python scripts/fetch_real_data.py dynamics       # Open-Meteo (or --era5)
    PYTHONPATH=. python scripts/fetch_real_data.py olr terrain
    PYTHONPATH=. python scripts/fetch_real_data.py build          # assemble from interim CSVs

Then train/evaluate on it:
    MONSOONIQ_PROFILE=real make train evaluate console serve
    (PowerShell: $env:MONSOONIQ_PROFILE="real"; .\\scripts\\real_data.ps1)

Options override configs/data_sources.yaml: --years 2022 2023 2024 --districts all
--model ecmwf_ifs025 --points-per-district 5 --gridded "data/real/raw/ncum/*.nc".
Every network response is cached under data/real/raw/, so re-running resumes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import REAL_DIR  # noqa: E402
from src.data.real import build_archive as B  # noqa: E402
from src.data.real import imd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_real_data")

STAGES = ["imd", "forecast", "dynamics", "olr", "terrain", "build"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", default=["all"], help=f"any of {STAGES} or 'all'")
    ap.add_argument("--config", default="configs/data_sources.yaml")
    ap.add_argument("--years", nargs="+", type=int)
    ap.add_argument("--months", nargs="+", type=int)
    ap.add_argument("--districts", choices=["study", "all"])
    ap.add_argument("--model", help="Open-Meteo model id, e.g. gfs_seamless, ecmwf_ifs025, "
                                    "ukmo_global_deterministic_10km")
    ap.add_argument("--points-per-district", type=int)
    ap.add_argument("--era5", action="store_true", help="dynamics from ERA5 (CDS key needed)")
    ap.add_argument("--gridded", help="glob of NCUM/GFS NetCDF forecasts instead of Open-Meteo")
    ap.add_argument("--gridded-var", default=None)
    ap.add_argument("--realtime", action="store_true",
                    help="also pull IMD real-time daily files for years without a final yearly file")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    years = args.years or cfg["seasons"]["years"]
    months = args.months or cfg["seasons"]["months"]
    scope = args.districts or cfg.get("districts", "study")
    model = args.model or cfg["forecast"]["models"][0]
    leads = cfg["forecast"].get("leads", [1, 2, 3, 4, 5])
    ppd = args.points_per_district or cfg["forecast"].get("points_per_district", 1)
    start_hour = cfg["observations"].get("day_window_start_utc_hour", 3)
    offset = cfg["observations"].get("day_offset", 0)

    stages = STAGES if "all" in args.stages else args.stages
    raw = os.path.join(REAL_DIR, "raw")
    interim = os.path.join(REAL_DIR, "interim")
    os.makedirs(interim, exist_ok=True)

    feats = B.load_districts(scope)
    days = imd.season_days(years, months)
    start, end = days[0], days[-1]
    # Contiguous date ranges (one per season) so the off-season months between JJAS
    # seasons are never requested - that alone cuts API use by ~2.5x.
    ranges = []
    for d in days:
        if ranges and (d - ranges[-1][1]).days == 1:
            ranges[-1][1] = d
        else:
            ranges.append([d, d])
    pts = [(f["properties"]["rep_lat"], f["properties"]["rep_lon"]) for f in feats]
    log.info("%d districts (%s), %d days %s..%s, model %s", len(feats), scope, len(days), start, end, model)
    sources = {}

    from src.data.real.openmeteo import DailyLimitReached, OpenMeteoClient
    client = OpenMeteoClient(cache_dir=os.path.join(raw, "openmeteo_cache"))

    try:
        if "imd" in stages:
            for y in years:
                if imd.download_year(y, os.path.join(raw, "imd")) is None:
                    if args.realtime:
                        log.info("IMD %d final file unavailable - pulling real-time daily files", y)
                        imd.download_realtime([d for d in days if d.year == y], os.path.join(raw, "imd"))
                    else:
                        log.warning("IMD %d not available (use --realtime for the current season)", y)
            B.stage_observations(raw, feats, days, interim)

        if "forecast" in stages:
            if args.gridded:
                from src.data.real.gridded_nwp import load_gridded_forecasts
                g = load_gridded_forecasts(args.gridded, args.gridded_var or cfg["forecast"]["gridded_var"],
                                           feats, leads)
                g["date"] = g["date"].astype(str)
                g["nwp_model"] = "gridded"
                g.to_csv(os.path.join(interim, "forecast_gridded.csv"), index=False)
                model = "gridded"
            else:
                import pandas as pd
                parts = [B.stage_forecast_openmeteo(client, feats, a, b, model, leads, ppd, interim,
                                                    start_hour, 0) for a, b in ranges]
                pd.concat(parts, ignore_index=True).to_csv(os.path.join(interim, f"forecast_{model}.csv"),
                                                          index=False)

        if "dynamics" in stages:
            if args.era5 or cfg["dynamics"]["source"] == "era5":
                from src.data.real import era5
                era5.download(years, months, os.path.join(raw, "era5"))
                frames = era5.sample_points(os.path.join(raw, "era5"), years, pts, start, end)
                B.stage_dynamics(frames, feats, interim, "ERA5 (Copernicus CDS)")
            else:
                import pandas as pd
                from src.data.real.openmeteo import fetch_dynamics
                per_range = [fetch_dynamics(client, pts, a, b, cfg["dynamics"]["openmeteo_model"],
                                            cfg["dynamics"].get("stencil_deg", 0.5),
                                            stencil=cfg["dynamics"].get("stencil", "forward"))
                             for a, b in ranges]
                frames = [pd.concat([r[k] for r in per_range if not r[k].empty]) if any(
                    not r[k].empty for r in per_range) else pd.DataFrame() for k in range(len(pts))]
                B.stage_dynamics(frames, feats, interim, f"Open-Meteo Historical Forecast ({cfg['dynamics']['openmeteo_model']})")

        if "olr" in stages:
            from src.data.real.olr import fetch_olr
            import pandas as pd
            per_range = [fetch_olr(pts, a, b, cache_dir=os.path.join(raw, "olr_cache")) for a, b in ranges]
            tag = per_range[0][1] if all(t == per_range[0][1] for _, t in per_range) else "partial"
            frames = [pd.concat([r[0][k] for r in per_range if not r[0][k].empty]) if any(
                not r[0][k].empty for r in per_range) else pd.DataFrame() for k in range(len(pts))]
            B.stage_olr(frames, feats, interim, tag)

        if "terrain" in stages:
            from src.data.real.openmeteo import terrain
            elev, slope = terrain(client, pts, cfg["terrain"].get("stencil_deg", 0.1))
            B.stage_terrain(elev, slope, feats, interim)

    except DailyLimitReached as exc:
        log.error("Open-Meteo daily limit reached (%s). Everything fetched so far is cached - "
                  "re-run the same command tomorrow to continue.", exc)
        sys.exit(2)

    if "build" in stages:
        for name, key in (("dynamics.csv", "dynamics_source"), ("olr.csv", "olr_source"),
                          ("terrain.csv", "terrain_source")):
            p = os.path.join(interim, name)
            if os.path.exists(p):
                import pandas as pd
                v = pd.read_csv(p, usecols=[key])[key].dropna()
                sources[key.replace("_source", "")] = str(v.iloc[0]) if len(v) else "missing"
        sources.update({
            "observations": "IMD 0.25 deg daily gridded rainfall (Pai et al. 2014), imdpune.gov.in",
            "forecast": f"Open-Meteo Previous Runs API, model={model}" if model != "gridded" else args.gridded,
            "boundaries": "Census 2011 districts (DataMeet, CC BY 2.5 IN)",
            "coastline": "Natural Earth 1:10m",
        })
        path = B.assemble(feats, interim, REAL_DIR, cfg["split"], cfg.get("regime_labels"), sources,
                          model=model if not args.gridded else "gridded", day_offset=offset)
        log.info("Done: %s. Next: set MONSOONIQ_PROFILE=real and run train / evaluate / console.", path)


if __name__ == "__main__":
    main()
