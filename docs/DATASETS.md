# MonsoonIQ — datasets

Problem statement 26080 asks for a regime classifier, a bias-corrected rainfall forecast, heavy-rain
probabilities, a district product and a verification report. Each of these needs a specific kind of
data. This page lists what MonsoonIQ uses, why, where to get it, and how it enters the pipeline.

Two archives share one schema, and the whole chain (train → evaluate → console → API → UI) runs on
either one:

| Profile | Archive | What it is for |
|---|---|---|
| `synthetic` (default) | `data/synthetic/district_daily.parquet` (committed) | Reproducible benchmark. Every number rebuilds offline from seed 42. NWP errors are simulated, so skill here shows the pipeline is internally consistent. It is **not** operational skill. |
| `real` | `data/real/district_daily.parquet` or `.csv.gz` (built on your machine) | IMD-observed rainfall and archived operational NWP forecasts. Skill here is genuine out-of-sample verification. |

Select the profile with `MONSOONIQ_PROFILE=synthetic|real`. The real profile writes models and
metrics to `artifacts/real/`, so the two profiles never overwrite each other.

---

## 1. Observed truth — IMD 0.25° daily gridded rainfall

* **What:** gauge-based daily rainfall analysis over India from ~6,900 stations, interpolated
  with inverse distance weighting onto a 0.25° grid. It runs from 1901 to the present, and
  real-time files cover the current season.
* **Why:** this is the analysis IMD and NCMRWF use themselves to verify rainfall forecasts. The
  IMD warning thresholds (64.5, 115.6 and 204.5 mm) are defined against station and gridded
  rainfall of this kind.
* **Citation:** Pai D.S., Sridhar L., Rajeevan M., Sreejith O.P., Satbhai N.S., Mukhopadhyay B.
  (2014). *Development of a new high spatial resolution (0.25° × 0.25°) long period (1901–2010)
  daily gridded rainfall data set over India and its comparison with existing data sets over the
  region.* Mausam 65(1):1–18.
* **Access (no login):**
  * Final yearly files: `POST https://imdpune.gov.in/cmpg/Griddata/rainfall.php` with `rain=<YYYY>`.
  * Real-time daily files: `POST https://imdpune.gov.in/cmpg/Realtimedata/Rainfall/rain.php` with `rain=<DDMMYYYY>`.
* **Format:** float32 little-endian binary, shape (days, 129 lat, 135 lon). Latitude runs 6.5–38.5°N,
  longitude 66.5–100°E, and missing values are −999. `src/data/real/imd.py` reads this format with
  numpy alone, and the tests write and re-read files in exactly this layout.
* **Day convention:** IMD labels rain by the 0830 IST (03 UTC) reading that ends the period, so day D
  is (D−1 03 UTC, D 03 UTC]. The builder's lag-correlation check found this on real data: observed
  rain correlates 0.58 with the Day-1 forecast on this window, against 0.48 for the opposite
  convention. `day_offset: 1` in the config applies it.
* **District aggregation:** each IMD cell is weighted by its area overlap with the Census-2011
  polygon, estimated from 5 × 5 sub-samples per cell. The builder writes the district mean, the
  max over cells with at least 25 % overlap, and the p90.

## 2. Raw NWP forecast

### 2a. Operational target — NCMRWF NCUM-G / NEPS-G

* **What:** the NCMRWF Unified Model global deterministic run (~12 km) and the 23-member
  ensemble, NEPS-G. These are the forecasts this post-processor is meant to correct in operations.
* **Access:** the NCMRWF data portal, after registration. Data may also be shared for SIH through
  the problem-statement owner.
* **How it plugs in:** `src/data/real/gridded_nwp.py` reads NetCDF in either common layout (one
  file per initialisation with a valid-time axis, or init × lead). It regrids to districts by area
  overlap and accumulates over the IMD day:

  ```bash
  python scripts/fetch_real_data.py forecast build --gridded "data/real/raw/ncum/*.nc"
  ```

### 2b. Open archive — Open-Meteo Previous Runs API (default, no key)

* **What:** what each global model actually predicted N days before the valid time. For a given
  hour, `precipitation_previous_dayN` comes from the run initialised at least 24·N hours earlier,
  so Day-N is a genuine N-day-ahead forecast with no hindcast leakage.
* **Models (archive depth, probed September 2026 by `scripts/probe_forecast_archive.py`):**
  **`jma_gsm` (Japan Meteorological Agency GSM) has Day 1–5 rain for 2021–2025 and is the default.**
  `gfs_seamless`, `ecmwf_ifs025`, `icon_global`, `gem_global` and CMA GRAPES only go back to 2024.
  `ukmo_global_deterministic_10km` has no archived precipitation.
* **Licence / citation:** CC BY 4.0 — Zippenfenig, P. (2023) Open-Meteo.com Weather API,
  doi:10.5281/zenodo.7970649.
* **Quota:** about 10,000 free calls per day. Only the JJAS seasons are requested, never the
  months in between. All responses are cached under `data/real/raw/`, so if a fetch hits the daily
  limit it exits with code 2, and re-running the next day resumes where it stopped.

## 3. Regime predictors — dynamics

* **ERA5** (the reference): Hersbach et al. (2020), QJRMS 146:1999–2049, 0.25° hourly, from the
  Copernicus CDS with a free key (`--era5`). Variables: 850/500 hPa u, v, relative vorticity,
  specific humidity and geopotential, plus MSLP and CAPE.
* **Open-Meteo Historical Forecast API** (no key, the default). It gives the same variables at
  each district's representative point. Relative vorticity comes from a small wind stencil:
  `forward` uses 2 extra points and fits the free quota, `cross` uses 4.
* **Timing (important for honesty):** the predictors are the **00 UTC analysis of D−1**, the
  state available when the Day-1 forecast for D is issued. They are never the analysis at the
  valid time. A test enforces this (`test_issue_time_predictors_lag_one_day`).
* **Derived fields:** specific humidity from RH and T (Bolton 1980), u/v from speed and direction,
  and moisture flux = |V850| · q850. MSLP and z500 anomalies are taken against a per-district,
  per-month climatology computed from the **training years only**. The trough latitude is the
  latitude of the lowest MSLP among districts in the 74–88°E band.

## 4. Convection — NOAA Interpolated OLR (series ends 2022, so it is neutralised for 2021–2025)

Liebmann & Smith (1996), BAMS 77:1275–1277. Daily, 2.5°, 1974 to the present, from NOAA PSL
THREDDS via NetCDF Subset in CSV mode, so no NetCDF library is needed. Anomalies are taken
against the PSL 1991–2020 daily long-term mean. OLR is optional: if PSL is unreachable, the
builder records `olr_source=unavailable` and fills climatology.

## 5. Regime labels — physically defined, never from the forecast

`src/regime/real_labeller.py` applies these rules in priority order, per district-day:

| Regime | Criterion | Source |
|---|---|---|
| Western Disturbance | Oct–May, ≥25°N, ≤82°E, z500 anomaly ≤ −35 gpm | IMD WD climatology |
| Monsoon Low/Depression | MSLP anomaly ≤ −3 hPa and ζ850 ≥ 3×10⁻⁵ s⁻¹ | IMD low/depression definitions |
| Orographic | terrain ≥ 350 m and upslope 850 hPa flow ≥ 6 m/s | Western Ghats / Himalaya literature |
| Coastal | ≤ 65 km from coast and onshore 850 hPa flow ≥ 5 m/s | coastal-convergence literature |
| Active / Break | core-monsoon-zone standardised rainfall ≥ +1 / ≤ −1 SD for ≥ 3 consecutive days | Rajeevan, Bhate & Jaswal (2010), J. Earth Syst. Sci. 119:229–247 |
| Weak/Normal | none of the above | — |

The CMZ index climatology uses training years only. The classifier then learns to predict these
labels from issue-time predictors. This replaces the synthetic archive's latent regime, which the
audit (AUDIT.md) flagged as "handing the model the answer".

## 6. Geography

* **Districts:** Census 2011, 641 districts, from DataMeet (CC BY 2.5 IN), with names and extent
  following the Census of India administrative atlas. They are simplified to about 1 km for
  display. The 53 study districts use the same polygons. See `scripts/build_district_boundaries.py`.
* **Coastline:** Natural Earth 1:10m (public domain), used for distance to coast.
* **Terrain:** Copernicus GLO-90 DEM via the Open-Meteo Elevation API. Slope comes from a
  ±0.1° stencil.

## 7. Building it

```bash
pip install -r requirements.txt            # + requirements-real.txt for ERA5 / NCUM NetCDF
PYTHONPATH=. python scripts/fetch_real_data.py all          # or: make real-data
MONSOONIQ_PROFILE=real make real serve                      # train, verify, console, serve
```

On Windows, run `.\scripts\real_data.ps1`.

The built archive, with results, is described in detail in [REAL_DATA_REPORT.md](REAL_DATA_REPORT.md).
`configs/data_sources.yaml` sets the years (default 2021–2025 JJAS), the split (train 2021–23,
val 2024, test 2025), `districts: study | all`, the model and the labelling thresholds. The builder
drops years it could not fetch and re-derives the split; the result is recorded in
`data/real/dataset_metadata.json` alongside sources, regime counts, predictor-gap fractions and the
day-alignment check.

## 8. Known limits

* The IMD gridded analysis smooths extremes where gauges are sparse, in NE India and the Himalaya.
* The Open-Meteo archives start in 2021 (GFS) and 2024 (most other models), so only a few seasons
  are available. Enlarge the archive by adding NCUM-G reforecasts through the gridded adapter.
* Predictors at Day 3–5 still use the D−1 analysis. This is mildly optimistic; the operational
  fix is to use the forecast's own dynamics at the valid time.
