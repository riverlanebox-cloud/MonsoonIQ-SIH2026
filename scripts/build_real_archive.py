"""
Build the real-data archive (data/real/) from the fetched IMD + GFS bundles.

    PYTHONPATH=. python scripts/build_real_archive.py --bundles data/raw/bundles

Also writes data/real/case_replays.json: the held-out days with the largest
observed district rainfall, chosen from the data (not from memory), so the
console's case strip always points at real events that are in the archive.
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from src.data.real_archive import RealArchiveBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def pick_cases(df: pd.DataFrame, test_years, n: int = 4, gap_days: int = 5) -> dict:
    """Largest observed district-maximum rainfall in the held-out seasons, one per event."""
    t = df[df["year"].isin(test_years)].copy()
    t["date_ts"] = pd.to_datetime(t["date"])
    ranked = t.sort_values("obs_rain_max", ascending=False)
    cases, used = [], []
    for _, r in ranked.iterrows():
        if any(abs((r["date_ts"] - u).days) <= gap_days for u in used):
            continue
        used.append(r["date_ts"])
        day = t[t["date"] == r["date"]].sort_values("obs_rain_max", ascending=False)
        key = day.head(4)
        span = [(r["date_ts"] + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in (-1, 0, 1)]
        span = [d for d in span if d in set(t["date"])]
        cases.append({
            "id": f"{r['district_id'].lower()}_{r['date']}",
            "name": f"{r['district_name'].split(' (')[0]}, {r['state_name']} "
                    f"({pd.Timestamp(r['date']).strftime('%d %b %Y')})",
            "regime": r["regime_name"],
            "dates": span,
            "key_districts": key["district_id"].tolist(),
            "description": (f"IMD observed {r['obs_rain_max']:.0f} mm (heaviest 0.25 deg cell) and "
                            f"{r['obs_rain_mean']:.0f} mm district mean; GFS Day-1 forecast "
                            f"{r['raw_nwp_d1']:.0f} mm district mean."),
            "observed_max_mm": round(float(r["obs_rain_max"]), 1),
        })
        if len(cases) == n:
            break
    return {"cases": cases,
            "selection": f"top {n} held-out district-days by observed rainfall, >{gap_days} days apart"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default="data/raw/bundles")
    ap.add_argument("--out", default="data/real")
    a = ap.parse_args()
    df = RealArchiveBuilder(a.bundles, a.out).build()
    from src.config import SPLITS
    cases = pick_cases(df, SPLITS["real"]["test"])
    with open(os.path.join(a.out, "case_replays.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=1)
    print(f"archive: {len(df)} rows, {df['date'].nunique()} days; cases: "
          + "; ".join(c["name"] for c in cases["cases"]))
