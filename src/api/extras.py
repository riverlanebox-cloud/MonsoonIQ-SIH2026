"""
Console extras: the data-source catalogue, the all-India district layer, and "Ask MonsoonIQ" -
a deterministic natural-language query bar over the live forecast (no LLM, no API key, every
answer is traceable to the same numbers the map shows).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src import config

router = APIRouter()

CATEGORY_RANK = {"green": 0, "yellow": 1, "orange": 2, "red": 3}

DATA_SOURCES: List[Dict[str, Any]] = [
    {
        "id": "imd_gridded", "role": "Observed truth (verification + training target)",
        "name": "IMD 0.25° daily gridded rainfall", "provider": "India Meteorological Department, Pune",
        "resolution": "0.25° × 0.25°, daily (24 h ending 03 UTC)", "period": "1901 – present (real-time for the current season)",
        "access": "imdpune.gov.in/cmpg/Griddata (no login) · scripts/fetch_real_data.py imd",
        "licence": "IMD open data policy", "citation": "Pai et al. (2014), Mausam 65(1):1–18",
        "why": "The gauge-based analysis IMD and NCMRWF verify against; ~6,900 stations.",
        "profiles": ["real"],
    },
    {
        "id": "ncum", "role": "Raw NWP forecast (operational target system)",
        "name": "NCMRWF NCUM-G / NEPS-G", "provider": "National Centre for Medium Range Weather Forecasting",
        "resolution": "12 km deterministic / 12 km 23-member ensemble, Day 1–10",
        "period": "operational archive via NCMRWF data portal",
        "access": "NetCDF → src/data/real/gridded_nwp.py (fetch_real_data.py forecast --gridded)",
        "licence": "NCMRWF data policy (registration)", "citation": "Mittal et al. (2019); Sarkar et al. (2021)",
        "why": "The forecast this post-processor is designed to correct in operations.",
        "profiles": ["real"],
    },
    {
        "id": "openmeteo_prev", "role": "Raw NWP forecast (open archive, Day 1–5 leads)",
        "name": "Archived GFS / ECMWF IFS / UK Met Office UM forecasts",
        "provider": "Open-Meteo Previous Runs API (NOAA, ECMWF, UKMO open data)",
        "resolution": "native model grid (9–25 km), hourly → IMD day", "period": "2021 → (GFS), 2024 → (most models)",
        "access": "previous-runs-api.open-meteo.com (no key) · fetch_real_data.py forecast",
        "licence": "CC BY 4.0", "citation": "Zippenfenig (2023), Open-Meteo.com, doi:10.5281/zenodo.7970649",
        "why": "What each model actually predicted N days ahead — the only free lead-time archive. "
               "UKMO UM is the same model family as NCUM-G.",
        "profiles": ["real"],
    },
    {
        "id": "era5", "role": "Regime predictors (reanalysis)",
        "name": "ERA5 reanalysis — 850/500 hPa wind, vorticity, humidity, z500, MSLP, CAPE",
        "provider": "ECMWF / Copernicus Climate Data Store", "resolution": "0.25°, hourly",
        "period": "1940 – present (5-day latency)",
        "access": "CDS API key · fetch_real_data.py dynamics --era5 (Open-Meteo fallback, no key)",
        "licence": "Copernicus licence", "citation": "Hersbach et al. (2020), QJRMS 146:1999–2049",
        "why": "Reference analysis for the dynamical state that defines a regime.",
        "profiles": ["real"],
    },
    {
        "id": "olr", "role": "Convection proxy (active/break signal)",
        "name": "NOAA Interpolated OLR", "provider": "NOAA PSL", "resolution": "2.5°, daily",
        "period": "1974 – present", "access": "PSL THREDDS NetCDF Subset (CSV, no key)",
        "licence": "Public domain", "citation": "Liebmann & Smith (1996), BAMS 77:1275–1277",
        "why": "Standard index of monsoon intraseasonal convection.", "profiles": ["real"],
    },
    {
        "id": "rajeevan", "role": "Regime labels — active / break spells",
        "name": "Core-monsoon-zone index (Rajeevan, Bhate & Jaswal 2010)",
        "provider": "IMD criterion, computed from the IMD grid", "resolution": "daily, CMZ area mean",
        "period": "same as IMD", "access": "src/regime/real_labeller.py",
        "licence": "method", "citation": "Rajeevan et al. (2010), J. Earth Syst. Sci. 119:229–247",
        "why": "IMD's own operational definition of active and break spells.", "profiles": ["real"],
    },
    {
        "id": "census_districts", "role": "District product geometry",
        "name": "Districts of India — Census 2011 (641 districts)", "provider": "DataMeet (Census of India atlas)",
        "resolution": "vector, simplified to ~1 km", "period": "2011 boundaries",
        "access": "github.com/datameet/maps · scripts/build_district_boundaries.py",
        "licence": "CC BY 2.5 IN", "citation": "DataMeet community maps",
        "why": "Official administrative extent used for area-weighted district rainfall.",
        "profiles": ["real", "synthetic"],
    },
    {
        "id": "dem", "role": "Terrain (orographic regime)", "name": "Copernicus GLO-90 DEM",
        "provider": "ESA / Airbus via Open-Meteo Elevation API", "resolution": "90 m",
        "period": "static", "access": "api.open-meteo.com/v1/elevation (no key)",
        "licence": "Copernicus DEM licence", "citation": "ESA (2021) Copernicus DEM",
        "why": "Elevation and slope for the windward-slope criterion.", "profiles": ["real"],
    },
    {
        "id": "coastline", "role": "Distance to coast (coastal regime)", "name": "Natural Earth 1:10m coastline",
        "provider": "Natural Earth", "resolution": "1:10 million", "period": "static",
        "access": "github.com/nvkelso/natural-earth-vector", "licence": "Public domain",
        "citation": "naturalearthdata.com", "why": "The 65 km coastal-convergence criterion.",
        "profiles": ["real", "synthetic"],
    },
    {
        "id": "synthetic", "role": "Benchmark archive (reproducible demo)",
        "name": "MonsoonIQ physically-parameterised synthetic archive", "provider": "this repository",
        "resolution": "0.5° grid → 53 districts, daily", "period": "2016 – 2023 (simulated)",
        "access": "make data (seed 42, bit-reproducible)", "licence": "MIT",
        "citation": "src/data/synthetic_generator.py",
        "why": "Lets anyone reproduce every number offline; errors are simulated, so skill here is "
               "internal consistency, not operational skill.",
        "profiles": ["synthetic"],
    },
]


def _meta() -> Dict[str, Any]:
    p = config.P("data/synthetic/dataset_metadata.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@router.get("/data-sources")
def data_sources():
    """What the system is built on, where to get it, and what is active right now."""
    real_meta_path = os.path.join(config.REAL_DIR, "dataset_metadata.json")
    real_meta = json.load(open(real_meta_path, encoding="utf-8")) if os.path.exists(real_meta_path) else None
    return {
        "active_profile": config.PROFILE,
        "active_archive": config.DATA_PATH,
        "split": config.SPLIT,
        "archive_metadata": _meta(),
        "real_archive_built": real_meta is not None,
        "real_archive_metadata": real_meta,
        "sources": [{**s, "active": config.PROFILE in s["profiles"]} for s in DATA_SOURCES],
        "how_to_switch": "python scripts/fetch_real_data.py all  →  MONSOONIQ_PROFILE=real make train evaluate console serve",
    }


# --------------------------------------------------------------------- Ask MonsoonIQ
_LEAD_WORDS = {"today": 1, "tomorrow": 1, "day after": 2, "day-after": 2}
_ZONES = ["west coast", "east coast", "central india", "northwest", "northeast", "gangetic plain",
          "south peninsula"]


def _parse(q: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    s = " " + q.lower().strip() + " "
    out: Dict[str, Any] = {"lead": None, "category": None, "prob": None, "places": [], "intent": "list"}
    m = re.search(r"\b(?:day|d)\s*-?\s*([1-5])\b", s) or re.search(r"\b([1-5])\s*(?:day|days)\s*(?:ahead|out|lead)", s)
    if m:
        out["lead"] = int(m.group(1))
    else:
        for w, v in _LEAD_WORDS.items():
            if w in s:
                out["lead"] = v
                break
    if "extreme" in s or "204" in s:
        out["prob"] = "p_extremely_heavy"
    elif "very heavy" in s or "115" in s:
        out["prob"] = "p_very_heavy"
    elif "heavy" in s or "64.5" in s:
        out["prob"] = "p_heavy"
    for cat, words in (("red", ["red", "warning"]), ("orange", ["orange", "alert"]),
                       ("yellow", ["yellow", "watch"])):
        if any(re.search(rf"\b{w}\b", s) for w in words):
            out["category"] = cat
            break
    names: Dict[str, List[tuple]] = {}

    def add(key: str, item: tuple):
        key = key.strip().lower()
        if len(key) >= 3 and item not in names.setdefault(key, []):
            names[key].append(item)

    for r in rows:
        add(r["state_name"], ("state", r["state_name"]))
        base = re.sub(r"\s*\(.*?\)", "", r["district_name"])
        add(base, ("district", r["district_id"]))
        add(base.split()[0], ("district", r["district_id"]))      # "Mumbai" -> City + Suburban
        for alias in re.findall(r"\((.*?)\)", r["district_name"]):
            for a in re.split(r"[/,]", alias):
                add(a, ("district", r["district_id"]))
    for z in _ZONES:
        add(z, ("zone", z.title()))
    for name in sorted(names, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", s):
            for item in names[name]:
                if item not in out["places"]:
                    out["places"].append(item)
            s = re.sub(rf"(?<![a-z]){re.escape(name)}(?![a-z])", " ", s)
    if "regime" in s or "situation" in s or "synoptic" in s:
        out["intent"] = "regime"
    elif any(w in s for w in ("raw", "correct", "bias", "adjust", "compare")):
        out["intent"] = "adjustment"
    elif out["places"] and all(k == "district" for k, _ in out["places"]) and (
            any(w in s for w in ("how much", "forecast for", "what about"))
            or (any(w in s for w in ("rainfall in", "rain in")) and not out["prob"] and not out["category"])):
        out["intent"] = "detail"
    return out


@router.get("/ask")
def ask(q: str = Query(..., min_length=2, max_length=300), date: Optional[str] = Query(None),
        lead: int = Query(1, ge=1, le=5)):
    """
    Plain-language questions over the current forecast, e.g.
      "heavy rain in Kerala tomorrow", "red warnings day 3", "what regime is driving today",
      "how much rain in Mumbai", "where did the correction add most rain on the west coast".
    """
    from src.api import main as M

    resolved = M._resolve_date(date)
    probe = M._district_rows(M.infer(resolved, lead))
    parsed = _parse(q, probe)
    ld = parsed["lead"] or lead
    rows = probe if ld == lead else M._district_rows(M.infer(resolved, ld))

    sel = rows
    if parsed["places"]:
        keep = []
        for r in rows:
            for kind, val in parsed["places"]:
                if (kind == "state" and r["state_name"] == val) or (kind == "district" and r["district_id"] == val) \
                        or (kind == "zone" and r["zone"].lower() == val.lower()):
                    keep.append(r)
                    break
        sel = keep
    where = ", ".join(v if k != "district" else next(r["district_name"] for r in rows if r["district_id"] == v)
                      for k, v in parsed["places"]) or "all districts"
    when = f"Day {ld} (valid {resolved})"

    if parsed["intent"] == "regime":
        c = M.console(resolved, ld, horizon=False)
        post = sorted(c["regime"]["posterior"].items(), key=lambda kv: -kv[1])[:3]
        local = {}
        for r in sel:
            local[r["regime"]] = local.get(r["regime"], 0) + 1
        top_local = sorted(local.items(), key=lambda kv: -kv[1])[:3]
        answer = (f"{when}: the dominant regime is {c['regime']['name']} "
                  f"({round(100 * c['regime']['confidence'])}% mean posterior). Runner-up: "
                  + ", ".join(f"{n} {round(100 * p)}%" for n, p in post[1:])
                  + (f". In {where}: " + ", ".join(f"{n} ({k} districts)" for n, k in top_local) if parsed["places"] else "")
                  + ".")
        return {"answer": answer, "intent": "regime", "date": resolved, "lead": ld,
                "parsed": parsed, "districts": [r["district_id"] for r in sel], "rows": sel[:12]}

    if parsed["category"]:
        sel = [r for r in sel if CATEGORY_RANK[r["category"]] >= CATEGORY_RANK[parsed["category"]]]
    if parsed["prob"]:
        sel = [r for r in sel if r[parsed["prob"]] >= 0.3]
        sel.sort(key=lambda r: -r[parsed["prob"]])
    elif parsed["intent"] == "adjustment":
        ql = q.lower()
        if any(w in ql for w in ("add", "increase", "raise", "under")):
            sel.sort(key=lambda r: -r["adjustment_mm"])
        elif any(w in ql for w in ("reduce", "cut", "lower", "over")):
            sel.sort(key=lambda r: r["adjustment_mm"])
        else:
            sel.sort(key=lambda r: -abs(r["adjustment_mm"]))
    else:
        sel.sort(key=lambda r: (-CATEGORY_RANK[r["category"]], -r["corrected_mm"]))

    if parsed["intent"] == "detail" and sel:
        r = sel[0]
        answer = (f"{r['district_name']}, {r['state_name']} — {when}: corrected {r['corrected_mm']} mm "
                  f"(raw NWP {r['raw_mm']} mm, P10–P90 {r['p10_mm']}–{r['p90_mm']} mm). "
                  f"P(≥64.5 mm) {round(100 * r['p_heavy'])}%, P(≥115.6 mm) {round(100 * r['p_very_heavy'])}%. "
                  f"Regime: {r['regime']}. IMD category: {r['category_label']}.")
    elif parsed["intent"] == "adjustment" and sel:
        top = sel[:5]
        answer = (f"{when}, {where}: largest regime-aware corrections — "
                  + "; ".join(f"{r['district_name']} {r['raw_mm']}→{r['corrected_mm']} mm ({r['regime']})" for r in top) + ".")
    elif sel:
        label = {"p_heavy": "P(heavy ≥64.5 mm) ≥ 30%", "p_very_heavy": "P(very heavy ≥115.6 mm) ≥ 30%",
                 "p_extremely_heavy": "P(extremely heavy ≥204.5 mm) ≥ 30%"}.get(parsed["prob"] or "", "")
        crit = label or (f"{parsed['category']} or higher" if parsed["category"] else "ranked by warning level")
        answer = (f"{when}, {where} — {len(sel)} district(s) {crit}: "
                  + ", ".join(f"{r['district_name']} ({r['corrected_mm']} mm, {round(100 * r['p_heavy'])}% heavy)"
                              for r in sel[:6]) + ("…" if len(sel) > 6 else "."))
    else:
        pool = keep if parsed["places"] else rows
        near = sorted(pool, key=lambda r: (-CATEGORY_RANK[r["category"]], -r["p_heavy"], -r["corrected_mm"]))[:3]
        answer = (f"{when}, {where}: no district meets that criterion. Closest: "
                  + ", ".join(f"{r['district_name']} ({r['corrected_mm']} mm, {round(100 * r['p_heavy'])}% heavy, "
                              f"{r['category_label'].lower()})" for r in near) + ".") if near else \
            f"{when}, {where}: no district meets that criterion."
    shown = sel if sel else locals().get("near", [])      # nothing matched: show the closest
    return {"answer": answer, "intent": parsed["intent"], "date": resolved, "lead": ld,
            "parsed": {**parsed, "places": [list(p) for p in parsed["places"]]},
            "districts": [r["district_id"] for r in sel], "rows": shown[:12]}
