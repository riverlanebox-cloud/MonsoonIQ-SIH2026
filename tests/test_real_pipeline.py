"""
Real-data pipeline tests.

These run offline: IMD files are written in the exact IMD binary layout and the Open-Meteo
client is replaced by a fake that answers in the real JSON shape. They check the plumbing
(formats, day windows, joins, labels, schema, the train chain on the real profile), not skill.
"""

import json
import os
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.real import imd
from src.data.real import build_archive as B
from src.data.real.openmeteo import daily_accumulation, imd_day_key, specific_humidity, _uv
from src.regime import real_labeller as RL

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_imd_grd_roundtrip(tmp_path):
    arr = np.random.RandomState(0).gamma(0.6, 12, size=(3, imd.NLAT, imd.NLON)).astype(np.float32)
    arr[:, 0, 0] = np.nan
    p = tmp_path / "x.grd"
    imd.write_grd(str(p), arr)
    assert os.path.getsize(p) == 3 * 129 * 135 * 4
    back = imd.read_grd(str(p))
    assert back.shape == (3, 129, 135)
    assert np.isnan(back[:, 0, 0]).all()
    np.testing.assert_allclose(back[:, 5:, 5:], arr[:, 5:, 5:])
    assert imd.LATS[0] == 6.5 and imd.LATS[-1] == 38.5 and imd.LONS[-1] == 100.0


def test_imd_day_window_is_03utc_to_03utc():
    t = pd.date_range("2024-07-01 00:00", "2024-07-03 23:00", freq="h", tz="UTC")
    s = pd.Series(1.0, index=t)
    daily = daily_accumulation(s, start_hour_utc=3, min_hours=24)
    # 2024-07-01 = (Jul 1 03 UTC, Jul 2 03 UTC] -> exactly 24 hourly accumulations
    assert daily.loc[date(2024, 7, 1)] == 24.0
    assert daily.loc[date(2024, 7, 2)] == 24.0
    k = imd_day_key(pd.DatetimeIndex(["2024-07-02 03:00", "2024-07-02 04:00"], tz="UTC"))
    assert list(k) == [date(2024, 7, 1), date(2024, 7, 2)]


def test_wind_and_humidity_conversions():
    u, v = _uv(np.array([10.0]), np.array([270.0]))       # from the west -> blowing east
    assert u[0] == pytest.approx(10.0) and abs(v[0]) < 1e-9
    q = specific_humidity(np.array([100.0]), np.array([20.0]), 1000.0)
    assert 0.014 < q[0] < 0.015                             # ~14.7 g/kg at saturation, 20 C


def test_rajeevan_spells_need_three_days():
    idx = pd.Index([f"2024-07-{d:02d}" for d in range(1, 11)])
    z = pd.Series([1.2, 1.5, 1.1, 0.2, -1.3, -1.4, 0.0, -1.2, -1.1, -1.5], index=idx)
    sp = RL.spells(z, 1.0, 3)
    assert list(sp) == [1, 1, 1, 0, 0, 0, 0, -1, -1, -1]


def test_cmz_mask_covers_central_india():
    m = RL.cmz_mask(imd.LATS, imd.LONS)
    i, j = np.argmin(abs(imd.LATS - 23)), np.argmin(abs(imd.LONS - 79))
    assert m[i, j]                                          # Madhya Pradesh is in the CMZ
    i, j = np.argmin(abs(imd.LATS - 10)), np.argmin(abs(imd.LONS - 76.5))
    assert not m[i, j]                                      # Kerala is not


class FakeClient:
    """Answers like Open-Meteo: a list of per-location objects with an `hourly` block."""

    def get(self, url, params):
        if "elevation" in url:
            n = len(params["latitude"].split(","))
            return {"elevation": [100.0 + 10 * i for i in range(n)]}
        lats = [float(x) for x in params["latitude"].split(",")]
        t = pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="h")
        out = []
        for k, lat in enumerate(lats):
            rs = np.random.RandomState(int(abs(lat) * 100) + k)
            h = {"time": [x.strftime("%Y-%m-%dT%H:%M") for x in t]}
            for var in params["hourly"].split(","):
                if var.startswith("precipitation"):
                    h[var] = np.round(rs.gamma(0.3, 1.5, len(t)), 2).tolist()
                elif "direction" in var:
                    h[var] = (240 + rs.randn(len(t)) * 20).tolist()
                elif "wind_speed" in var:
                    h[var] = (9 + rs.randn(len(t)) * 3).clip(0).tolist()
                elif "relative_humidity" in var:
                    h[var] = (75 + rs.randn(len(t)) * 10).clip(0, 100).tolist()
                elif var.startswith("temperature_850"):
                    h[var] = (20 + rs.randn(len(t))).tolist()
                elif var.startswith("temperature_500"):
                    h[var] = (-6 + rs.randn(len(t))).tolist()
                elif "geopotential" in var:
                    h[var] = (5860 + rs.randn(len(t)) * 20).tolist()
                elif var == "pressure_msl":
                    h[var] = (1000 + rs.randn(len(t)) * 2).tolist()
                elif var == "cape":
                    h[var] = (800 + rs.randn(len(t)) * 300).clip(0).tolist()
            out.append({"latitude": lat, "hourly": h})
        return out


@pytest.fixture(scope="module")
def real_archive(tmp_path_factory):
    from src.data.real.openmeteo import fetch_dynamics
    root = tmp_path_factory.mktemp("real")
    raw, interim = root / "raw", root / "interim"
    years, months = [2021, 2022, 2023], [7]
    rs = np.random.RandomState(1)
    for y in years:
        n = 365
        f = rs.gamma(0.5, 14, size=(n, imd.NLAT, imd.NLON)).astype(np.float32)
        imd.write_grd(imd.yearly_path(str(raw / "imd"), y), f)
    feats = B.load_districts("study")
    days = imd.season_days(years, months)
    os.makedirs(interim, exist_ok=True)
    B.stage_observations(str(raw), feats, days, str(interim))
    client = FakeClient()
    B.stage_forecast_openmeteo(client, feats, days[0], days[-1], "gfs_seamless", [1, 2, 3, 4, 5], 1, str(interim))
    pts = [(f["properties"]["rep_lat"], f["properties"]["rep_lon"]) for f in feats]
    B.stage_dynamics(fetch_dynamics(client, pts, days[0], days[-1]), feats, str(interim), "fake")
    B.stage_olr([pd.DataFrame() for _ in feats], feats, str(interim), "unavailable")
    B.stage_terrain(None, None, feats, str(interim))
    path = B.assemble(feats, str(interim), str(root), {"train": [2021], "val": [2022], "test": [2023]},
                      sources={"test": "fixture"}, model="gfs_seamless")
    return root, path


def test_real_archive_schema_matches_models(real_archive):
    from src.data.io import read_table
    from src.regime.ml_classifier import MLRegimeClassifier
    from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts
    root, path = real_archive
    df = read_table(path)
    need = set(MLRegimeClassifier.FEATURE_COLS) | set(MonsoonIQMixtureOfExperts.PREDICTOR_COLS) | {
        "date", "year", "month", "district_id", "district_name", "state_name", "zone", "regime",
        "regime_name", "obs_rain_mean", "obs_rain_max", "obs_heavy", "obs_very_heavy",
        "raw_nwp_d1", "raw_nwp_d5"}
    assert need <= set(df.columns), need - set(df.columns)
    assert df["district_id"].nunique() == 53
    assert {"cmz_index_lag2", "cmz_spell_lag2"} <= set(df.columns)   # issue-time persistence
    assert df[list(MLRegimeClassifier.FEATURE_COLS)].isna().sum().sum() == 0
    assert set(df["regime"].unique()) <= set(range(1, 8))
    assert (df["obs_rain_max"] >= df["obs_rain_mean"] - 1e-6).all()
    meta = json.load(open(os.path.join(root, "dataset_metadata.json")))
    assert meta["dataset_type"] == "REAL_OBSERVED_IMD_NWP"
    assert meta["split"] == {"train": [2021], "val": [2022], "test": [2023]}
    assert set(meta["day_alignment_check"]) == {"shift_-1", "shift_+0", "shift_+1"}


def test_issue_time_predictors_lag_one_day(real_archive):
    """u850 on D must equal the 00 UTC analysis of D-1 (no peeking at the valid day)."""
    root, _ = real_archive
    dyn = pd.read_csv(os.path.join(root, "interim", "dynamics.csv"))
    from src.data.io import read_table
    df = read_table(os.path.join(root, "district_daily.parquet"))
    did = df["district_id"].iloc[0]
    a = dyn[dyn["district_id"] == did].set_index("date")["u850"]
    b = df[df["district_id"] == did].set_index("date")["u850"]
    d = "2021-07-10"
    assert b.loc[d] == pytest.approx(a.loc["2021-07-09"], rel=1e-6)


def test_train_chain_runs_on_real_profile(real_archive):
    root, _ = real_archive
    env = dict(os.environ, MONSOONIQ_PROFILE="real", MONSOONIQ_REAL_DIR=str(root),
               PYTHONPATH=os.pathsep.join([ROOT, os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c",
                        "import src.config as c; assert c.PROFILE=='real' and c.TEST_YEARS==[2023];"
                        "from src.train import run_training_pipeline as t; t(fit_lead_bundle=False)"],
                       cwd=str(root), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert os.path.exists(os.path.join(str(root), "artifacts", "real", "models", "mixture_of_experts.joblib"))


def test_fetch_cli_end_to_end_offline(tmp_path, monkeypatch):
    """The user-facing orchestrator: every stage + build, with network replaced by fakes."""
    import importlib
    import src.data.real.openmeteo as om
    import src.data.real.olr as olrmod
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    cli = importlib.import_module("fetch_real_data")

    rs = np.random.RandomState(3)

    def fake_download(year, raw_dir):
        path = imd.yearly_path(raw_dir, year)
        imd.write_grd(path, rs.gamma(0.5, 14, size=(365, imd.NLAT, imd.NLON)).astype(np.float32))
        return path

    monkeypatch.setattr(cli, "REAL_DIR", str(tmp_path))
    monkeypatch.setattr(imd, "download_year", fake_download)
    monkeypatch.setattr(om, "OpenMeteoClient", lambda **kw: FakeClient())
    monkeypatch.setattr(olrmod, "fetch_olr", lambda pts, a, b, cache_dir=None: ([pd.DataFrame() for _ in pts], "unavailable"))
    monkeypatch.setattr(sys, "argv", ["fetch_real_data.py", "all", "--years", "2021", "2022", "2023",
                                      "--months", "7", "8"])
    cli.main()

    meta = json.load(open(os.path.join(tmp_path, "dataset_metadata.json")))
    assert meta["years"] == [2021, 2022, 2023]
    assert meta["districts_count"] == 53
    import glob
    fc = pd.read_csv(glob.glob(os.path.join(tmp_path, "interim", "forecast_*.csv"))[0])
    # only JJAS-style season months were requested, never the months in between
    assert set(pd.to_datetime(fc["date"]).dt.month.unique()) <= {7, 8}
    assert meta["sources"]["observations"].startswith("IMD 0.25")
