"""
Console data products.

The operations console needs two things the verification pipeline does not
produce: a per-day timeline of the whole archive (so a user can scrub a season
and see regime and risk move together) and a ranked list of significant days (so
they can jump straight to what matters instead of hunting for a date).

Both are precomputed here from the fitted models, in bulk, and written to
`artifacts/console/`. The API serves them directly; a date that is not in the
timeline is still served live from the models in ~25 ms.

Run:  PYTHONPATH=. python -m src.console.artifacts
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA = "data/synthetic/district_daily.parquet"
OUT_DIR = "artifacts/console"
LEADS = (1, 2, 3, 4, 5)
IMD_CATEGORIES = {"green": 0, "yellow": 1, "orange": 2, "red": 3}


def _alert_category(day_df: pd.DataFrame, corrected: np.ndarray,
                    p90: np.ndarray, p_heavy: np.ndarray,
                    p_very_heavy: np.ndarray) -> np.ndarray:
    """IMD warning category per district: green / yellow / orange / red."""
    cat = np.zeros(len(day_df), dtype=np.int8)
    cat = np.where(corrected >= 64.5, 1, cat)
    cat = np.where((corrected >= 115.6) | (p_heavy >= 0.7), 2, cat)
    cat = np.where((corrected >= 204.5) | (p90 >= 204.5) | (p_very_heavy >= 0.6), 3, cat)
    return cat


def build_console_artifacts(data_path: str = DATA, out_dir: str = OUT_DIR,
                            leads: tuple = LEADS, force: bool = False) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    timeline_path = os.path.join(out_dir, "timeline.json")
    events_path = os.path.join(out_dir, "events.json")
    meta_path = os.path.join(out_dir, "meta.json")

    if os.path.exists(timeline_path) and not force:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        logger.info("Console artifacts already present (built %s); use force=True to rebuild.",
                    meta.get("built_utc"))
        return meta

    from src.regime.ml_classifier import MLRegimeClassifier
    from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts
    from src.correction.quantile_regressor import QuantileRegressor
    from src.heavy_rain.heavy_rain_classifier import HeavyRainProbabilityModule

    t0 = time.time()
    df = pd.read_parquet(data_path).reset_index(drop=True)
    clf = MLRegimeClassifier(); clf.load()
    moe = MonsoonIQMixtureOfExperts.load()
    qr = QuantileRegressor.load()
    hrc = HeavyRainProbabilityModule.load()

    logger.info("Classifying regimes over %d district-days ...", len(df))
    posteriors = clf.predict_proba(df)
    dominant = np.array([clf.classes_[i] for i in np.argmax(posteriors, axis=1)])
    confidence = posteriors.max(axis=1)

    dates = df["date"].to_numpy()
    unique_dates = sorted(set(dates.tolist()))
    date_index = {d: i for i, d in enumerate(unique_dates)}
    by_date = df.groupby("date", sort=True).indices

    per_day: Dict[str, Dict[str, Any]] = {
        d: {
            "date": d,
            "regime_id": int(df["regime"].iloc[by_date[d][0]]),
            "leads": {},
        } for d in unique_dates
    }

    observed_max: Dict[str, float] = {}
    for d in unique_dates:
        idx = by_date[d]
        observed_max[d] = round(float(df["obs_rain_max"].iloc[idx].max()), 1)
        dominant_day = np.bincount(dominant[idx], minlength=8)[1:].argmax() + 1
        per_day[d]["regime_id"] = int(dominant_day)
        per_day[d]["regime_name"] = clf.REGIME_NAMES[int(dominant_day)]
        per_day[d]["regime_confidence"] = round(float(confidence[idx].mean()), 3)
        per_day[d]["observed_heavy_districts"] = int((df["obs_rain_max"].iloc[idx] >= 64.5).sum())
        per_day[d]["observed_very_heavy_districts"] = int(
            (df["obs_rain_max"].iloc[idx] >= 115.6).sum())

    for lead in leads:
        nwp_col = f"raw_nwp_d{lead}"
        if nwp_col not in df.columns:
            continue
        preds = moe.predict_all_systems(df, nwp_col=nwp_col, regime_probs=posteriors)
        quants = qr.predict_quantiles(df, nwp_col=nwp_col)
        probs = hrc.predict_probabilities(df, nwp_col=nwp_col)
        corrected = preds["monsooniq"]
        raw = preds["raw_nwp"]
        p90 = quants["p90"]
        cat = _alert_category(df, corrected, p90, probs["heavy"], probs["very_heavy"])

        for d in unique_dates:
            idx = by_date[d]
            counts = np.bincount(cat[idx], minlength=4)
            per_day[d]["leads"][f"d{lead}"] = {
                "corrected_max_mm": round(float(corrected[idx].max()), 1),
                "raw_max_mm": round(float(raw[idx].max()), 1),
                "yellow": int(counts[1]), "orange": int(counts[2]), "red": int(counts[3]),
                "p_heavy_max": round(float(probs["heavy"][idx].max()), 3),
            }
        logger.info("lead D%d aggregated in %.1fs", lead, time.time() - t0)

    timeline = [per_day[d] for d in unique_dates]

    # Ranked significant days: warnings at the shortest lead, then observed extremes.
    scored = []
    for d in unique_dates:
        entry = per_day[d]
        d1 = entry["leads"].get("d1", {})
        severity = 3 * d1.get("red", 0) + 2 * d1.get("orange", 0) + d1.get("yellow", 0)
        scored.append({
            "date": d,
            "regime_name": entry["regime_name"],
            "regime_id": entry["regime_id"],
            "severity_score": severity,
            "warned_districts": d1.get("yellow", 0) + d1.get("orange", 0) + d1.get("red", 0),
            "red": d1.get("red", 0), "orange": d1.get("orange", 0), "yellow": d1.get("yellow", 0),
            "corrected_max_mm": d1.get("corrected_max_mm"),
            "observed_max_mm": observed_max[d],
            "observed_heavy_districts": entry["observed_heavy_districts"],
            "observed_very_heavy_districts": entry["observed_very_heavy_districts"],
        })
    scored.sort(key=lambda e: (e["severity_score"], e["observed_very_heavy_districts"],
                               e["observed_heavy_districts"], e["corrected_max_mm"] or 0),
                reverse=True)
    events = scored[:40]

    # Regime spell summary: useful for the "break vs active" story.
    spells = []
    current = None
    for d in unique_dates:
        rid = per_day[d]["regime_id"]
        if current and current["regime_id"] == rid:
            current["end"] = d
            current["days"] += 1
            current["observed_heavy_districts"] += per_day[d]["observed_heavy_districts"]
        else:
            if current:
                spells.append(current)
            current = {"regime_id": rid, "regime_name": per_day[d]["regime_name"],
                       "start": d, "end": d, "days": 1,
                       "observed_heavy_districts": per_day[d]["observed_heavy_districts"]}
    if current:
        spells.append(current)
    spells.sort(key=lambda s: s["days"], reverse=True)

    # Which day should the console open on? The most warning-heavy day of the
    # monsoon season (JJAS), not the most warning-heavy day of the year: the
    # archive's winter Western Disturbance days out-rank monsoon days on raw
    # warning count, and opening a monsoon product on a November day would
    # misrepresent what it is for.
    jjas = [e for e in scored if e["date"][5:7] in ("06", "07", "08", "09")]
    featured = [e for e in jjas if e["severity_score"] >= np.percentile(
        [e["severity_score"] for e in jjas], 95)] if jjas else []
    default_date = (featured[0]["date"] if featured
                    else (events[0]["date"] if events else unique_dates[-1]))

    meta = {
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "build_seconds": round(time.time() - t0, 1),
        "provenance": "SYNTHETIC_PHYSICALLY_PLAUSIBLE",
        "dates": len(timeline),
        "date_range": [unique_dates[0], unique_dates[-1]],
        "default_date": default_date,
        "featured_monsoon_days": [e["date"] for e in featured[:10]],
        "leads": [f"d{l}" for l in leads if f"d{l}" in per_day[unique_dates[0]]["leads"]],
        "districts": int(df["district_id"].nunique()),
        "alert_categories": {"green": "no warning", "yellow": "64.5-115.5 mm",
                             "orange": "115.6-204.4 mm", "red": ">=204.5 mm"},
        "year_summary": _year_summary(per_day, unique_dates),
        "spells": spells[:12],
        "note": ("Timeline is model output over a synthetic archive; observed columns are "
                 "shown only as reference for demos and are never used as forecast input."),
    }

    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, separators=(",", ":"))
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=1)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Console artifacts written: %d days, %d events in %.1fs",
                len(timeline), len(events), time.time() - t0)
    return meta


def _year_summary(per_day: Dict[str, Any], dates: List[str]) -> List[Dict[str, Any]]:
    years: Dict[int, Dict[str, Any]] = {}
    for d in dates:
        entry = per_day[d]
        y = int(d[:4])
        rec = years.setdefault(y, {
            "year": y, "days": 0, "warning_district_days": 0,
            "red": 0, "orange": 0, "yellow": 0, "observed_heavy_districts": 0,
            "regimes": {},
        })
        d1 = entry["leads"].get("d1", {})
        rec["days"] += 1
        rec["red"] += d1.get("red", 0)
        rec["orange"] += d1.get("orange", 0)
        rec["yellow"] += d1.get("yellow", 0)
        rec["warning_district_days"] += d1.get("yellow", 0) + d1.get("orange", 0) + d1.get("red", 0)
        rec["observed_heavy_districts"] += entry["observed_heavy_districts"]
        name = entry["regime_name"]
        rec["regimes"][name] = rec["regimes"].get(name, 0) + 1
    return [years[y] for y in sorted(years)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    info = build_console_artifacts(force=True)
    print(json.dumps({k: v for k, v in info.items() if k != "spells"}, indent=2)[:2000])
