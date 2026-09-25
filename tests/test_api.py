"""
MonsoonIQ API contract tests.

These lock the schemas the console depends on. When a payload changes shape, the
test fails here rather than in the browser.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app, load_artifacts


@pytest.fixture(scope="module")
def client():
    load_artifacts()
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_date(client):
    """A date that exists in the archive, taken from the API rather than hardcoded."""
    health = client.get("/health").json()
    return health["date_range"][1]


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert isinstance(data["models_loaded"], dict) and all(data["models_loaded"].values())
    assert data["districts"] > 0
    assert data["provenance"] == "SYNTHETIC_PHYSICALLY_PLAUSIBLE"
    assert data["artifacts"]["timeline"] > 0


def test_model_card_endpoint(client):
    data = client.get("/model-card").json()
    assert data["data"]["provenance"]
    assert len(data["regimes"]) == 7
    assert data["known_limits"], "the model card must state its limitations"


def test_console_payload_is_one_screen(client, sample_date):
    """The console contract: one request must carry the whole screen."""
    data = client.get(f"/console?date={sample_date}&lead=1").json()
    for key in ("date", "lead", "regime", "summary", "districts", "navigation", "horizon"):
        assert key in data, f"console payload missing {key}"
    assert data["date"] == sample_date
    assert len(data["districts"]) == data["summary"]["counts"]["green"] + \
        data["summary"]["districts_in_warning"]
    counts = data["summary"]["counts"]
    assert sum(counts.values()) == len(data["districts"])
    top = data["districts"][0]
    assert {"district_id", "corrected_mm", "raw_mm", "p_heavy", "category", "regime"} <= set(top)
    assert 0.0 <= top["p_heavy"] <= 1.0
    # the ranking must actually be a ranking: most exposed district first
    rank = {"red": 3, "orange": 2, "yellow": 1, "green": 0}
    assert all(rank[data["districts"][i]["category"]] >= rank[data["districts"][i + 1]["category"]]
               for i in range(len(data["districts"]) - 1))


def test_console_accepts_every_lead(client, sample_date):
    for lead in range(1, 6):
        data = client.get(f"/console?date={sample_date}&lead={lead}").json()
        assert data["lead"] == lead
        assert len(data["horizon"]) >= 1


def test_console_navigation_is_walkable(client, sample_date):
    """The ← / → keys depend on navigation always pointing at a real date."""
    current = sample_date
    seen = {current}
    for _ in range(6):
        data = client.get(f"/console?date={current}&lead=1").json()
        nxt = data["navigation"]["prev_date"]
        if not nxt:
            break
        assert nxt not in seen
        seen.add(nxt)
        current = nxt
    assert len(seen) > 1


def test_timeline_and_events(client):
    tl = client.get("/timeline").json()
    assert tl["meta"]["dates"] == len(tl["days"])
    assert tl["meta"]["default_date"] in [d["date"] for d in tl["days"]]
    day = tl["days"][0]
    assert {"date", "regime_id", "regime_name", "leads"} <= set(day)
    assert "d1" in day["leads"] and {"red", "orange", "yellow"} <= set(day["leads"]["d1"])

    ev = client.get("/events?limit=5").json()
    assert 0 < len(ev["events"]) <= 5
    assert ev["events"][0]["severity_score"] >= ev["events"][-1]["severity_score"]


def test_bulletin_is_a_filed_document(client, sample_date):
    data = client.get(f"/bulletin?date={sample_date}&lead=1").json()
    assert "MONSOONIQ" in data["text"].upper()
    assert "not an official IMD bulletin" in data["text"]
    hi = client.get(f"/bulletin?date={sample_date}&lead=1&lang=hi").json()
    assert hi["text"] != data["text"]


def test_district_detail_endpoint(client, sample_date):
    listing = client.get(f"/console?date={sample_date}&lead=1").json()["districts"]
    district_id = listing[0]["district_id"]
    data = client.get(f"/district/{district_id}?date={sample_date}&lead=1").json()
    assert data["district_id"] == district_id
    assert "advisory" in data and "cap_alert" in data
    assert data["p10_mm"] <= data["p90_mm"]
    assert len(data["lead_trend"]) == 5
    assert abs(sum(data["regime_posterior"].values()) - 1.0) < 0.01
    assert data["reference_observed_max_mm"] >= 0


def test_export_csv(client, sample_date):
    res = client.get(f"/export/districts.csv?date={sample_date}&lead=1")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    lines = res.text.strip().splitlines()
    assert len(lines) > 10
    assert "district_id" in lines[0] and "corrected_mm" in lines[0]


def test_grid_fields(client):
    data = client.get("/grid").json()
    assert data["grid_shape"] == [57, 57]
    assert len(data["cells"]) > 1000
    assert {"lat", "lon", "observed_mm", "raw_mm"} <= set(data["cells"][0])


def test_regime_endpoint(client, sample_date):
    data = client.get(f"/regime?date={sample_date}").json()
    assert "dominant_regime" in data
    assert abs(sum(data["soft_probabilities"].values()) - 1.0) < 0.01


def test_corrected_forecast_endpoint(client, sample_date):
    data = client.get(f"/forecast/corrected?date={sample_date}&lead_time_days=1").json()
    assert data["total_districts"] == len(data["districts"])
    first = data["districts"][0]
    assert "raw_nwp" in first and "monsooniq_corrected" in first
    assert "p10" in first and "p50" in first and "p90" in first
    assert "alert_level" in first


def test_heavy_probability_endpoint(client, sample_date):
    data = client.get(f"/forecast/heavy-probability?date={sample_date}&lead_time_days=1").json()
    assert len(data["district_probabilities"]) > 0
    assert set(data["thresholds_mm"]) == {"heavy", "very_heavy", "extremely_heavy"}


def test_districts_and_geojson(client):
    data = client.get("/districts").json()
    assert data["total"] > 0
    assert "district_id" in data["districts"][0]

    geo = client.get("/districts/geojson").json()
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == data["total"]


def test_verification_endpoints(client):
    summary = client.get("/verification/summary").json()
    assert "continuous_metrics" in summary
    assert summary["data_provenance"] == "SYNTHETIC_PHYSICALLY_PLAUSIBLE"
    # the significance statement is a product feature, not an internal debug field
    claims = summary["significance_statement"]["claims"]
    assert any(not c["supported"] for c in claims), \
        "the published claim set must include the unsupported claim"

    heavy = client.get("/verification/heavy-events").json()
    assert heavy["threshold_mm"] == 64.5
    assert "overall_by_system" in heavy and "by_regime" in heavy

    rv = client.get("/verification/regime-value").json()
    assert rv["available"] is True
    assert "regime_conditioning_gain_over_regime_agnostic_csi" in rv["headline"]


def test_verification_pdf_endpoint(client):
    res = client.get("/verification/report.pdf")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 10000


def test_case_replays_endpoint(client):
    data = client.get("/case-replays").json()
    assert len(data["cases"]) >= 3
    for case in data["cases"]:
        assert case["dates"] and case["key_districts"]


def test_explain_endpoint(client, sample_date):
    listing = client.get(f"/console?date={sample_date}&lead=1").json()["districts"]
    data = client.get(f"/explain/{listing[0]['district_id']}/{sample_date}").json()
    assert "top_feature_attributions" in data
