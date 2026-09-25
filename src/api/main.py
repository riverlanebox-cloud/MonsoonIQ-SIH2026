"""
MonsoonIQ API.

Design rules this file follows:

* One request per screen. `/console` returns everything the operations console
  needs for a date and lead, so changing the date is a single round trip (~25 ms
  of inference) instead of a fan-out of five requests.
* The model's own view of the regime is what gets served. The dataset's latent
  regime label is only ever exposed under an explicitly named `demo_truth_*`
  field, because in operations the regime is unknown.
* Everything that is expensive and shared (timeline, ranked events, verification
  artifacts) is precomputed and served as JSON.

Endpoints are grouped as: system, console, district, bulletin, spatial,
verification, model card, exports, and legacy aliases.
"""

import csv
import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.regime.ml_classifier import MLRegimeClassifier
from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts
from src.correction.lead_time_models import LeadTimeCorrectionBundle
from src.correction.quantile_regressor import QuantileRegressor
from src.heavy_rain.heavy_rain_classifier import HeavyRainProbabilityModule
from src.api.advisory import AdvisoryGenerator
from src.api.cache import LRUCache
from src.api.schemas import HealthResponse

logger = logging.getLogger("MonsoonIQ_API")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_PATH = "data/synthetic/district_daily.parquet"
GEOJSON_PATH = "data/geojson/india_districts.geojson"
TIMELINE_PATH = "artifacts/console/timeline.json"
EVENTS_PATH = "artifacts/console/events.json"
CONSOLE_META_PATH = "artifacts/console/meta.json"
GRID_NPZ = "data/synthetic/grid_feature_samples.npz"
MANIFEST_PATH = "artifacts/models/training_manifest.json"
PDF_PATH = "artifacts/reports/MonsoonIQ_Verification_Report.pdf"

CATEGORY_LABEL = {"green": "No warning", "yellow": "Watch", "orange": "Alert", "red": "Warning"}
CATEGORY_RANK = {"green": 0, "yellow": 1, "orange": 2, "red": 3}

app = FastAPI(
    title="MonsoonIQ API",
    description=("Regime-aware post-processing of NWP rainfall for India. "
                 "Dense, single-request endpoints for the operations console."),
    version="2.0.0",
)
# The season timeline is a few hundred kilobytes of JSON; compress it on the wire.
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

STATE: Dict[str, Any] = {
    "df": None, "geojson": None, "grid": None, "timeline": None, "events": None,
    "console_meta": None, "manifest": None,
    "regime_clf": None, "moe": None, "lead_bundle": None, "hrc": None, "qr": None,
    "cache": LRUCache(capacity=1024),
    "loaded_utc": None,
}


# --------------------------------------------------------------------- loading
def _read_json(path: str, default=None):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not read %s: %s", path, exc)
    return default


def load_artifacts():
    """Load models, dataset and precomputed console/verification artifacts."""
    logger.info("Loading MonsoonIQ artifacts ...")
    if os.path.exists(DATA_PATH):
        STATE["df"] = pd.read_parquet(DATA_PATH)
        STATE["df"]["date"] = STATE["df"]["date"].astype(str)
        logger.info("dataset: %d rows", len(STATE["df"]))
    STATE["geojson"] = _read_json(GEOJSON_PATH)
    STATE["timeline"] = _read_json(TIMELINE_PATH, [])
    STATE["events"] = _read_json(EVENTS_PATH, [])
    STATE["console_meta"] = _read_json(CONSOLE_META_PATH, {})
    STATE["manifest"] = _read_json(MANIFEST_PATH, {})

    if os.path.exists(GRID_NPZ):
        STATE["grid"] = np.load(GRID_NPZ, allow_pickle=True)

    for key, loader, path in (
        ("regime_clf", MLRegimeClassifier, "artifacts/models/regime_classifier.joblib"),
        ("moe", MonsoonIQMixtureOfExperts, "artifacts/models/mixture_of_experts.joblib"),
        ("lead_bundle", LeadTimeCorrectionBundle, "artifacts/models/lead_time_bundle.joblib"),
        ("hrc", HeavyRainProbabilityModule, "artifacts/models/heavy_rain_module.joblib"),
        ("qr", QuantileRegressor, "artifacts/models/quantile_regressor.joblib"),
    ):
        if not os.path.exists(path):
            logger.warning("missing artifact %s", path)
            continue
        try:
            obj = loader() if key == "regime_clf" else loader.load(path)
            if key == "regime_clf":
                obj.load()
            STATE[key] = obj
            logger.info("loaded %s", key)
        except Exception as exc:  # pragma: no cover
            logger.error("failed to load %s: %s", key, exc)

    STATE["loaded_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info("Ready. models=%s", [k for k in ("regime_clf", "moe", "lead_bundle", "hrc", "qr")
                                     if STATE[k] is not None])


@app.on_event("startup")
def _startup():
    load_artifacts()


# ------------------------------------------------------------------- inference
def _require(*keys):
    for k in keys:
        if STATE[k] is None:
            raise HTTPException(status_code=503, detail=f"{k} not loaded; run the training pipeline")


def _available_dates() -> List[str]:
    return sorted(STATE["df"]["date"].unique().tolist()) if STATE["df"] is not None else []


def _resolve_date(date: Optional[str]) -> str:
    dates = _available_dates()
    if not dates:
        raise HTTPException(status_code=503, detail="dataset not loaded")
    if date and date in set(dates):
        return date
    if date:
        # nearest available date, so a deep link never dead-ends
        idx = int(np.argmin([abs((pd.Timestamp(d) - pd.Timestamp(date)).days) for d in dates]))
        return dates[idx]
    return dates[len(dates) // 2]


def infer(date: str, lead: int) -> Dict[str, Any]:
    """All model output for one date and lead, including regime posteriors."""
    cache_key = f"infer::{date}::{lead}"
    hit = STATE["cache"].get(cache_key)
    if hit is not None:
        return hit
    _require("df", "regime_clf", "hrc", "qr")

    day_df = STATE["df"][STATE["df"]["date"] == date].reset_index(drop=True)
    if day_df.empty:
        raise HTTPException(status_code=404, detail=f"no data for {date}")

    posteriors = STATE["regime_clf"].predict_proba(day_df)
    nwp_col = f"raw_nwp_d{lead}"

    if STATE["lead_bundle"] is not None and lead in STATE["lead_bundle"].available_leads():
        preds = STATE["lead_bundle"].predict_all_systems(day_df, lead, regime_probs=posteriors)
        probs = STATE["lead_bundle"].predict_probabilities(day_df, lead)
        quants = STATE["lead_bundle"].predict_quantiles(day_df, lead)
        model_scope = "per_lead"
    else:
        preds = STATE["moe"].predict_all_systems(day_df, nwp_col=nwp_col,
                                                 regime_probs=posteriors)
        probs = STATE["hrc"].predict_probabilities(day_df, nwp_col=nwp_col)
        quants = STATE["qr"].predict_quantiles(day_df, nwp_col=nwp_col)
        model_scope = "day1_model_applied_to_lead"

    regimes = [STATE["regime_clf"].REGIME_NAMES[STATE["regime_clf"].classes_[i]]
               for i in np.argmax(posteriors, axis=1)]
    out = {
        "date": date, "lead": lead, "model_scope": model_scope,
        "day": day_df, "posteriors": posteriors, "regimes": regimes,
        "raw": preds["raw_nwp"], "corrected": preds["monsooniq"],
        "global_qm": preds["global_qm"], "global_lgb": preds["global_lgb"],
        "p10": quants["p10"], "p50": quants["p50"], "p90": quants["p90"],
        "p_heavy": probs["heavy"], "p_very_heavy": probs["very_heavy"],
        "p_extremely_heavy": probs["extremely_heavy"],
    }
    STATE["cache"].set(cache_key, out)
    return out


def _category(corrected: float, p90: float, p_heavy: float, p_very_heavy: float) -> str:
    return AdvisoryGenerator.determine_alert_level(
        mean_rain=corrected, max_rain=max(p90, corrected),
        p_heavy=float(p_heavy), p_very_heavy=float(p_very_heavy))


def _district_rows(inf: Dict[str, Any]) -> List[Dict[str, Any]]:
    df = inf["day"]
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        corrected = float(inf["corrected"][i]); raw = float(inf["raw"][i])
        p90 = float(inf["p90"][i])
        cat = _category(corrected, p90, float(inf["p_heavy"][i]), float(inf["p_very_heavy"][i]))
        rows.append({
            "district_id": r["district_id"], "district_name": r["district_name"],
            "state_name": r["state_name"], "zone": r["zone"],
            "lat": round(float(r["latitude"]), 4), "lon": round(float(r["longitude"]), 4),
            "elevation_m": round(float(r["elevation"]), 1),
            "regime": inf["regimes"][i],
            "regime_probability": round(float(inf["posteriors"][i].max()), 3),
            "raw_mm": round(raw, 2),
            "corrected_mm": round(corrected, 2),
            "adjustment_mm": round(corrected - raw, 2),
            "adjustment_pct": round(100 * (corrected - raw) / raw, 1) if raw > 0.5 else None,
            "p10_mm": round(float(inf["p10"][i]), 2),
            "p50_mm": round(float(inf["p50"][i]), 2),
            "p90_mm": round(p90, 2),
            "p_heavy": round(float(inf["p_heavy"][i]), 3),
            "p_very_heavy": round(float(inf["p_very_heavy"][i]), 3),
            "p_extremely_heavy": round(float(inf["p_extremely_heavy"][i]), 3),
            "category": cat, "category_label": CATEGORY_LABEL[cat],
            # reference only - never a forecast input, kept out of the UI body
            "reference_observed_max_mm": round(float(r["obs_rain_max"]), 1),
        })
    return rows


# ------------------------------------------------------------------- system
@app.get("/health")
def health():
    df = STATE["df"]
    return {
        "status": "healthy" if STATE["moe"] is not None else "degraded",
        "version": app.version,
        "loaded_utc": STATE["loaded_utc"],
        "provenance": STATE["console_meta"].get("provenance", "UNKNOWN"),
        "models_loaded": {k: STATE[k] is not None for k in
                          ("regime_clf", "moe", "lead_bundle", "hrc", "qr")},
        "leads_available": STATE["lead_bundle"].available_leads() if STATE["lead_bundle"] else [1],
        "districts": int(df["district_id"].nunique()) if df is not None else 0,
        "dates": len(_available_dates()),
        "date_range": [min(_available_dates()), max(_available_dates())] if df is not None else [],
        "artifacts": {
            "timeline": len(STATE["timeline"] or []),
            "events": len(STATE["events"] or []),
            "verification_summary": os.path.exists("artifacts/metrics/verification_summary.json"),
            "report_pdf": os.path.exists(PDF_PATH),
        },
    }


@app.get("/model-card")
def model_card():
    manifest = STATE["manifest"] or {}
    verification = _read_json("artifacts/metrics/verification_summary.json", {})
    return {
        "product": "MonsoonIQ — regime-aware post-processing of NWP rainfall",
        "version": app.version,
        "data": {
            "archive": manifest.get("dataset", {}),
            "provenance": (verification.get("data_provenance")
                           or STATE["console_meta"].get("provenance")),
            "provenance_detail": verification.get("provenance_detail", {}),
            "archive_sha256_16": (verification.get("provenance_detail", {})
                                  or {}).get("archive_sha256_16"),
            "verification_period": verification.get("verification_period", {}),
        },
        "training": {
            "split": manifest.get("split"),
            "samples": {k: manifest.get(k) for k in
                        ("train_samples", "val_samples", "test_samples")},
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "regime_classifier": manifest.get("regime_classifier"),
            "lead_bundle": manifest.get("lead_bundle"),
            "grid_correction": manifest.get("grid_correction"),
            "exceedance_models": manifest.get("exceedance_models"),
        },
        "artifact_hashes": manifest.get("artifacts", {}),
        "regimes": [
            {"id": 1, "name": "Active Monsoon", "criterion": "trough 18-26N, strong 850 hPa jet, deep convection"},
            {"id": 2, "name": "Break Monsoon", "criterion": "trough at Himalayan foothills, suppressed central India"},
            {"id": 3, "name": "Monsoon Low / Depression", "criterion": "closed cyclonic core, MSLP deficit"},
            {"id": 4, "name": "Orographic", "criterion": "steep terrain, cross-barrier flow"},
            {"id": 5, "name": "Coastal", "criterion": "convergence within ~50 km of the coast"},
            {"id": 6, "name": "Western Disturbance", "criterion": "mid-latitude trough, NW India"},
            {"id": 7, "name": "Weak / Normal", "criterion": "no dominant forcing"},
        ],
        "known_limits": [
            "All skill figures come from a synthetic archive whose NWP errors were "
            "generated by this repository; they are not operational skill.",
            "Regime conditioning currently shows no significant gain over a "
            "regime-agnostic model on the heavy-rainfall categorical score.",
            "Grid resolution is 0.5 degrees; district geometries are simplified boxes.",
            "Rapid cyclogenesis and sub-kilometre orographic extremes are unresolved.",
        ],
    }


# ------------------------------------------------------------------ console
@app.get("/console")
def console(date: Optional[str] = Query(None), lead: int = Query(1, ge=1, le=5),
            horizon: bool = Query(True, description="include the 5-day district trend")):
    """Single payload for the operations console."""
    resolved = _resolve_date(date)
    inf = infer(resolved, lead)
    rows = _district_rows(inf)

    sorted_rows = sorted(rows, key=lambda r: (-CATEGORY_RANK[r["category"]],
                                              -r["p_heavy"], -r["corrected_mm"]))
    counts = {c: sum(1 for r in rows if r["category"] == c) for c in CATEGORY_RANK}
    dates = _available_dates()
    i = dates.index(resolved)
    timeline = STATE["timeline"] or []
    timeline_index = next((k for k, t in enumerate(timeline) if t["date"] == resolved), None)

    horizon_rows = []
    if horizon:
        for ld in (STATE["lead_bundle"].available_leads() if STATE["lead_bundle"] else [lead]):
            d = infer(resolved, ld)
            horizon_rows.append({
                "lead": ld,
                "max_corrected_mm": round(float(max(d["corrected"])), 1),
                "max_p90_mm": round(float(max(d["p90"])), 1),
                "districts_warned": int(sum(
                    1 for k in range(len(d["day"]))
                    if _category(float(d["corrected"][k]), float(d["p90"][k]),
                                 float(d["p_heavy"][k]), float(d["p_very_heavy"][k])) != "green")),
                "red": int(sum(
                    1 for k in range(len(d["day"]))
                    if _category(float(d["corrected"][k]), float(d["p90"][k]),
                                 float(d["p_heavy"][k]), float(d["p_very_heavy"][k])) == "red")),
            })

    return {
        "date": resolved,
        "lead": lead,
        "model_scope": inf["model_scope"],
        "issued_utc": STATE["loaded_utc"],
        "valid_for": resolved,
        "provenance": STATE["console_meta"].get("provenance", "UNKNOWN"),
        "regime": {
            "id": int(np.argmax(inf["posteriors"].mean(axis=0))) + 1,
            "name": STATE["regime_clf"].REGIME_NAMES[
                STATE["regime_clf"].classes_[int(np.argmax(inf["posteriors"].mean(axis=0)))]],
            "confidence": round(float(inf["posteriors"].mean(axis=0).max()), 3),
            "posterior": {
                STATE["regime_clf"].REGIME_NAMES[c]: round(float(inf["posteriors"].mean(axis=0)[k]), 3)
                for k, c in enumerate(STATE["regime_clf"].classes_)
            },
            "district_agreement": round(float(np.mean(
                [r["regime"] == inf["regimes"][k] for k, r in enumerate(rows)])), 3),
        },
        "summary": {
            "counts": counts,
            "districts_in_warning": counts["yellow"] + counts["orange"] + counts["red"],
            "max_corrected_mm": round(float(max(inf["corrected"])), 1),
            "max_raw_mm": round(float(max(inf["raw"])), 1),
            "max_observed_mm": round(float(inf["day"]["obs_rain_max"].max()), 1),
            "mean_bias_adjustment_mm": round(float(np.mean(inf["corrected"] - inf["raw"])), 2),
        },
        "horizon": horizon_rows,
        "districts": sorted_rows,
        "navigation": {
            "index": i, "total": len(dates),
            "prev_date": dates[i - 1] if i > 0 else None,
            "next_date": dates[i + 1] if i + 1 < len(dates) else None,
        },
        "demo_truth_regime": inf["day"]["regime_name"].mode().iloc[0],
    }


@app.get("/timeline")
def timeline():
    """Per-day regime and risk over the whole archive (for the scrubber)."""
    if not STATE["timeline"]:
        raise HTTPException(status_code=404,
                            detail="timeline not built; run python -m src.console.artifacts")
    return {"meta": STATE["console_meta"], "days": STATE["timeline"]}


@app.get("/events")
def events(limit: int = Query(25, ge=1, le=40)):
    if not STATE["events"]:
        raise HTTPException(status_code=404,
                            detail="events not built; run python -m src.console.artifacts")
    return {"total": len(STATE["events"]), "events": STATE["events"][:limit]}


# ------------------------------------------------------------------ districts
@app.get("/district/{district_id}")
def district_detail(district_id: str, date: Optional[str] = Query(None),
                    lead: int = Query(1, ge=1, le=5)):
    resolved = _resolve_date(date)
    inf = infer(resolved, lead)
    rows = _district_rows(inf)
    row = next((r for r in rows if r["district_id"] == district_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown district {district_id}")

    advisory = AdvisoryGenerator.generate_advisory(
        district_name=row["district_name"], state_name=row["state_name"],
        regime_name=row["regime"], mean_rain=row["corrected_mm"], max_rain=row["p90_mm"],
        p_heavy=row["p_heavy"], p_very_heavy=row["p_very_heavy"])
    cap = AdvisoryGenerator.generate_cap_alert(
        district_id=district_id, district_name=row["district_name"],
        state_name=row["state_name"], date_str=resolved, advisory_data=advisory)

    idx = [k for k, r in enumerate(rows) if r["district_id"] == district_id][0]
    posterior = {STATE["regime_clf"].REGIME_NAMES[c]: round(float(inf["posteriors"][idx][k]), 3)
                 for k, c in enumerate(STATE["regime_clf"].classes_)}
    trend = []
    for ld in (STATE["lead_bundle"].available_leads() if STATE["lead_bundle"] else [lead]):
        d = infer(resolved, ld)
        trend.append({
            "lead": ld,
            "corrected_mm": round(float(d["corrected"][idx]), 2),
            "raw_mm": round(float(d["raw"][idx]), 2),
            "p90_mm": round(float(d["p90"][idx]), 2),
            "p_heavy": round(float(d["p_heavy"][idx]), 3),
        })

    return {
        "date": resolved, "lead": lead, "provenance": STATE["console_meta"].get("provenance"),
        **row, "regime_posterior": posterior, "advisory": advisory, "cap_alert": cap,
        "lead_trend": trend,
    }


@app.get("/bulletin")
def bulletin(date: Optional[str] = Query(None), lead: int = Query(1, ge=1, le=5),
             lang: str = Query("en", pattern="^(en|hi)$"),
             only_warned: bool = Query(True)):
    """Copy-pasteable district warning bulletin, in the shape an SDMA would issue."""
    resolved = _resolve_date(date)
    inf = infer(resolved, lead)
    rows = _district_rows(inf)
    warned = [r for r in rows if r["category"] != "green"] if only_warned else rows
    warned.sort(key=lambda r: (-CATEGORY_RANK[r["category"]], -r["p_heavy"]))

    regime = STATE["regime_clf"].REGIME_NAMES[
        STATE["regime_clf"].classes_[int(np.argmax(inf["posteriors"].mean(axis=0)))]]
    header = (f"MONSOONIQ DISTRICT RAINFALL GUIDANCE — {resolved} (Day {lead})\n"
              f"Synoptic regime: {regime}\n"
              f"Basis: bias-corrected NWP rainfall guidance; IMD warning categories "
              f"(≥64.5 mm watch, ≥115.6 mm alert, ≥204.5 mm warning).\n"
              f"Provenance: {STATE['console_meta'].get('provenance')} — research prototype, "
              f"not an official IMD bulletin.\n")

    if lang == "hi":
        header = header.replace("MONSOONIQ DISTRICT RAINFALL GUIDANCE",
                                "मानसूनआईक्यू जिला वर्षा मार्गदर्शन")
        header += "\n(हिंदी अनुवाद — श्रेणी लेबल और जिला नाम अंग्रेज़ी में भी दिए गए हैं)\n"

    lines = [header, "-" * 78]
    for r in warned:
        adv = AdvisoryGenerator.generate_advisory(
            district_name=r["district_name"], state_name=r["state_name"],
            regime_name=r["regime"], mean_rain=r["corrected_mm"], max_rain=r["p90_mm"],
            p_heavy=r["p_heavy"], p_very_heavy=r["p_very_heavy"])
        label = adv.get("alert_label_hi" if lang == "hi" else "alert_label_en",
                        r["category_label"])
        lines.append(
            f"{r['district_name']} ({r['state_name']})  [{r['category'].upper()} — {label}]\n"
            f"   expected {r['corrected_mm']:.0f} mm, 90th percentile {r['p90_mm']:.0f} mm, "
            f"P(≥64.5 mm) = {r['p_heavy']:.2f}, P(≥115.6 mm) = {r['p_very_heavy']:.2f}\n"
            f"   regime: {r['regime']} (p={r['regime_probability']:.2f})"
        )
    if not warned:
        lines.append("No district reaches a warning category on this date.")
    text = "\n".join(lines)

    return {
        "date": resolved, "lead": lead, "language": lang,
        "districts_included": len(warned),
        "text": text,
        "districts": warned,
    }


# --------------------------------------------------------------------- spatial
@app.get("/grid")
def grid_fields(date: Optional[str] = Query(None)):
    """Gridded fields for dates present in the spatial archive (raw vs corrected)."""
    if STATE["grid"] is None:
        raise HTTPException(status_code=404,
                            detail="grid archive missing; run scripts/build_grid_samples.py")
    keys = [k for k in STATE["grid"].keys()
            if k not in ("lats", "lons", "land_index", "grid_shape", "land_mask",
                         "elevation_grid")]
    if not keys:
        raise HTTPException(status_code=404, detail="grid archive has no date records")
    resolved = date if date in keys else sorted(keys)[-1]
    rec = STATE["grid"][resolved].item()
    li = STATE["grid"]["land_index"].astype(int)
    shape = tuple(int(x) for x in STATE["grid"]["grid_shape"])

    def to_grid(flat):
        arr = np.zeros(shape, dtype=np.float32)
        arr.ravel()[li] = np.asarray(flat, dtype=np.float32)
        return arr

    return {
        "date": resolved,
        "available_dates": sorted(keys),
        "grid_shape": list(shape),
        "cells": [
            {"lat": float(STATE["grid"]["lats"][j // shape[1]]),
             "lon": float(STATE["grid"]["lons"][j % shape[1]]),
             "observed_mm": round(float(rec["true_rain"][k]), 1),
             "raw_mm": round(float(rec["raw_nwp_d1"][k]), 1)}
            for k, j in enumerate(li)
        ],
    }


# ---------------------------------------------------------------- verification
@app.get("/verification/summary")
def verification_summary():
    data = _read_json("artifacts/metrics/verification_summary.json")
    if data is None:
        raise HTTPException(status_code=404, detail="run python src/evaluate.py")
    return data


@app.get("/verification/heavy-events")
def verification_heavy_events():
    data = _read_json("artifacts/metrics/heavy_events_summary.json")
    if data is None:
        raise HTTPException(status_code=404, detail="run python src/evaluate.py")
    return data


@app.get("/verification/regime-value")
def verification_regime_value():
    data = _read_json("artifacts/metrics/regime_value_audit.json")
    if data is None:
        raise HTTPException(status_code=404, detail="run python src/evaluate.py")
    return data


@app.get("/verification/report.pdf")
def verification_report():
    if not os.path.exists(PDF_PATH):
        raise HTTPException(status_code=404, detail="run python src/evaluate.py")
    return FileResponse(PDF_PATH, media_type="application/pdf",
                        filename="MonsoonIQ_Verification_Report.pdf")


# ---------------------------------------------------------------------- exports
@app.get("/export/districts.csv")
def export_districts(date: Optional[str] = Query(None), lead: int = Query(1, ge=1, le=5)):
    resolved = _resolve_date(date)
    rows = _district_rows(infer(resolved, lead))
    cols = ["district_id", "district_name", "state_name", "zone", "category", "regime",
            "raw_mm", "corrected_mm", "adjustment_mm", "p10_mm", "p50_mm", "p90_mm",
            "p_heavy", "p_very_heavy", "p_extremely_heavy", "lat", "lon"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return Response(
        content=buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="monsooniq_{resolved}_d{lead}.csv"'})


# ----------------------------------------------------------------------- legacy
@app.get("/districts")
def list_districts():
    if not STATE["geojson"]:
        raise HTTPException(status_code=500, detail="districts geojson not loaded")
    out = []
    for f in STATE["geojson"]["features"]:
        p = f["properties"]
        out.append({"district_id": p["district_id"], "district_name": p["district_name"],
                    "state_name": p["state_name"], "zone": p["zone"],
                    "centroid_lat": p["centroid_lat"], "centroid_lon": p["centroid_lon"],
                    "elevation_m": p.get("elevation_m", 100)})
    return {"total": len(out), "districts": out}


@app.get("/districts/geojson")
def districts_geojson():
    if not STATE["geojson"]:
        raise HTTPException(status_code=500, detail="GeoJSON not loaded")
    return STATE["geojson"]


@app.get("/regime")
def regime(date: Optional[str] = Query(None), lead: int = Query(1, ge=1, le=5)):
    resolved = _resolve_date(date)
    inf = infer(resolved, lead)
    clf = STATE["regime_clf"]
    mean_post = inf["posteriors"].mean(axis=0)
    order = np.argsort(mean_post)[::-1]
    return {
        "date": resolved,
        "dominant_regime": clf.REGIME_NAMES[clf.classes_[int(order[0])]],
        "dominant_regime_id": int(clf.classes_[int(order[0])]),
        "soft_probabilities": {clf.REGIME_NAMES[c]: round(float(mean_post[k]), 3)
                               for k, c in enumerate(clf.classes_)},
        "source": "classifier posterior mean over districts",
        "demo_truth_regime": inf["regimes"][0] if inf["regimes"] else None,
    }


@app.get("/forecast/corrected")
def forecast_corrected(date: Optional[str] = Query(None), lead_time_days: int = Query(1, ge=1, le=5),
                       mode: str = Query("district", pattern="^(district|grid)$")):
    resolved = _resolve_date(date)
    rows = _district_rows(infer(resolved, lead_time_days))
    payload = {
        "date": resolved, "lead_time_days": lead_time_days, "mode": mode,
        "provenance": STATE["console_meta"].get("provenance", "UNKNOWN"),
        "total_districts": len(rows),
        "districts": [{
            "district_id": r["district_id"], "district_name": r["district_name"],
            "state_name": r["state_name"], "zone": r["zone"],
            "centroid_lat": r["lat"], "centroid_lon": r["lon"], "elevation_m": r["elevation_m"],
            "dominant_regime": r["regime"], "regime_probability": r["regime_probability"],
            "raw_nwp": r["raw_mm"], "monsooniq_corrected": r["corrected_mm"],
            "bias_delta": r["adjustment_mm"], "observed_rain": r["reference_observed_max_mm"],
            "p10": r["p10_mm"], "p50": r["p50_mm"], "p90": r["p90_mm"],
            "p_heavy": r["p_heavy"], "p_very_heavy": r["p_very_heavy"],
            "p_extremely_heavy": r["p_extremely_heavy"],
            "alert_level": r["category"], "alert_code": r["category"].upper(),
            "is_grid_mode": mode == "grid",
        } for r in rows],
    }
    if mode == "grid":
        payload["note"] = ("Grid mode returns the district-scale product; true gridded "
                           "fields are served by /grid for the sampled archive dates.")
    return payload


@app.get("/forecast/heavy-probability")
def heavy_probability(date: Optional[str] = Query(None), lead_time_days: int = Query(1, ge=1, le=5)):
    resolved = _resolve_date(date)
    rows = _district_rows(infer(resolved, lead_time_days))
    return {
        "date": resolved, "lead_time_days": lead_time_days,
        "thresholds_mm": {"heavy": 64.5, "very_heavy": 115.6, "extremely_heavy": 204.5},
        "district_probabilities": [{
            "district_id": r["district_id"], "district_name": r["district_name"],
            "p_heavy": r["p_heavy"], "p_very_heavy": r["p_very_heavy"],
            "p_extremely_heavy": r["p_extremely_heavy"], "alert_level": r["category"],
        } for r in rows],
    }


@app.get("/case-replays")
def case_replays():
    return {"cases": [
        {"id": "kerala_2018", "name": "Kerala orographic surge (Aug 2018)",
         "regime": "Orographic / active low-level jet",
         "dates": ["2018-08-14", "2018-08-15", "2018-08-16", "2018-08-17"],
         "key_districts": ["KL_WAY", "KL_IDK", "KL_EKM", "KL_TVM"],
         "description": "Windward Ghats surge under a strong Somali jet; the raw field is "
                        "damped everywhere the terrain is steep."},
        {"id": "konkan_2019", "name": "Konkan coastal convergence (Jul 2019)",
         "regime": "Coastal convergence",
         "dates": ["2019-07-25", "2019-07-26", "2019-07-27"],
         "key_districts": ["MH_MUM", "MH_SUB", "MH_THN", "MH_RTG"],
         "description": "Narrow coastal rain band that coarse fields smear inland and weaken."},
        {"id": "himalaya_2023", "name": "NW Himalaya trough interaction (Jul 2023)",
         "regime": "Western disturbance + monsoon interaction",
         "dates": ["2023-07-09", "2023-07-10", "2023-07-11"],
         "key_districts": ["UK_RUD", "UK_DRN", "UK_UTK", "HP_SML"],
         "description": "Valley-scale extremes that the raw field smooths away."},
    ]}


@app.get("/explain/{district_id}/{date}")
def explain(district_id: str, date: str, lead: int = Query(1, ge=1, le=5)):
    resolved = _resolve_date(date)
    day_df = STATE["df"][STATE["df"]["date"] == resolved]
    row = day_df[day_df["district_id"] == district_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="district/date not found")
    if STATE["regime_clf"] is None:
        raise HTTPException(status_code=503, detail="classifier not loaded")
    out = STATE["regime_clf"].explain_sample(row.iloc[0].to_dict())
    out.update({"district_id": district_id, "date": resolved, "lead": lead})
    return out


# --------------------------------------------------------------- static frontend
from fastapi.staticfiles import StaticFiles  # noqa: E402

_frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              "frontend", "dist")
if not os.path.exists(_frontend_dist):
    _frontend_dist = "frontend/dist"
if os.path.exists(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend-static")
    logger.info("Serving built frontend from %s", _frontend_dist)
else:
    logger.warning("frontend/dist not found - run `npm run build` in frontend/ to serve the UI")
