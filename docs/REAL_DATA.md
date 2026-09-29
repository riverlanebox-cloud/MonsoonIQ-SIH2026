# Real data: IMD observations + NOAA GFS forecasts

MonsoonIQ now runs by default on real data: five south-west monsoon seasons,
1 June – 30 September 2021–2025, over the 53 modelled districts. The seeded
simulator is still available (`make synthetic`, or `MONSOONIQ_DATA=synthetic`).

| | Source | What is used |
|---|---|---|
| Observations | IMD Pune 0.25° daily gridded rainfall (Pai et al. 2014), [imdpune.gov.in](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html) | 24 h accumulation ending 03 UTC, 1 Jun – 30 Sep |
| Raw NWP forecast | NOAA GFS, 00 UTC runs, [AWS Open Data `noaa-gfs-bdp-pds`](https://registry.opendata.aws/noaa-gfs-bdp-pds/) | Day 1–5 rainfall (0.5°), nine dynamical predictors per lead day (1°) |
| District boundaries | [DataMeet India maps](https://github.com/datameet/maps), Census 2011 (CC BY 2.5 IN) | real polygons for the 53 districts |
| Terrain | GFS 0.5° orography and land mask | slope, distance to coast |

**Why GFS and not NCMRWF NCUM?** The problem statement (SIH26080, MoES/NCMRWF)
concerns post-processing a dynamical model's rainfall. NCUM output is not publicly
downloadable; GFS is the closest open, archived global model with multi-day
forecasts. The post-processing chain is model-agnostic: replace the GFS bundle
with NCUM fields on the same grid and nothing downstream changes.

**Why 2021–2025?** NOAA's AWS archive of full GFS forecasts starts in February
2021. (NCEI's online GFS archive keeps only 0–6 h forecasts after May 2020.)

## Reproduce

```bash
make setup                 # python deps (requests, numpy, ...)
make fetch-real            # ~3.5 GB transferred, writes ~200 MB of bundles to data/raw/bundles
make real                  # build data/real/, then train, evaluate, console artifacts
make serve                 # console on :8000
```

`scripts/fetch_real_data.py` downloads only the GRIB2 messages it needs (via the
`.idx` byte-range index) and decodes them with `src/data/grib2_lite.py`, a
NumPy GRIB2 decoder, so no eccodes/cfgrib installation is needed. The committed
results were fetched with an equivalent in-browser fetcher that writes identical
bundles; the Python decoder was checked value-for-value against it.

## Processing, step by step (`src/data/real_archive.py`)

1. **Forecast day = IMD day.** For the 00 UTC run of date *D*, Day-*k* rainfall is
   the GFS accumulation from 03 UTC on *D+k−1* to 03 UTC on *D+k*:
   `A(24k+3) − A(24k−21)`, where `A(F)` is the 0–*F* h accumulation. Which IMD
   calendar date that window matches is **measured**, not assumed: the lag with the
   highest correlation between all-India GFS Day-1 and IMD rainfall is used and
   written to `dataset_metadata.json` (`day_alignment`).
2. **Observed district rainfall.** IMD cells are weighted by the fraction of each
   0.25° cell inside the Census-2011 district polygon (5×5 sub-sampling, so small
   districts like Mumbai City work). `obs_rain_mean` is the weighted mean;
   `obs_rain_max` is the heaviest cell overlapping the district by at least 20%.
   Warning categories use the IMD scale on the district maximum.
3. **Forecast district rainfall.** The same area weighting on the 0.5° GFS grid
   gives `raw_nwp_d1` … `raw_nwp_d5`.
4. **Dynamical predictors, per lead.** For every district-day and lead *k*, the GFS
   forecast *for that valid day* at lead *k* (12 UTC): 850 hPa u, v and relative
   vorticity, 500 hPa specific humidity, CAPE, top-of-atmosphere OLR, MSLP, 500 hPa
   height, RH850, PWAT; plus day-level indices: OLR anomaly over central India,
   monsoon-trough latitude (MSLP minimum 75–85°E), MSLP anomaly of the head-Bay
   box. Columns `<field>_d<k>`; the Day-*k* models only ever see the Day-*k*
   columns, so no model uses information a forecaster would not have had at issue
   time. Anomalies are relative to a seasonal cycle built from the **training
   years only**.
5. **Regime labels.** There is no "true" regime in real data, so each day gets one
   label from objective criteria (`configs/regime_labels_real.yaml`):
   active/break from the Rajeevan et al. (2010) core-monsoon-zone rainfall index,
   lows/depressions from GFS MSLP + 850 hPa vorticity, offshore-trough orographic
   and east-coast spells from zone rainfall, western disturbances from 500 hPa
   height. The classifier must recover these from GFS dynamics alone.
6. **Grid archive.** 0.5° land-cell feature stacks (IMD regridded area-consistently
   from 0.25°) for every held-out day and every third day otherwise, used by the
   grid-native correction and the FSS verification.

## Modelling choices specific to real data (made on the 2023 validation season)

* **Regime classifier inputs.** The labels are domain-scale, so besides the per-district
  predictors the classifier also sees three day-level GFS indices: the maximum 850 hPa vorticity
  in the head-Bay box, mean 850 hPa westerly wind onto the Western Ghats, and the 500 hPa height
  anomaly over NW India. Validation accuracy rose from 0.56 to 0.64 (majority class 0.61).
* **Expert base.** Regime experts correct the global GBM's out-of-fold prediction, with each
  regime's correction shrunk by n/(n+2000) rows, instead of a per-regime quantile-mapping base
  (which is worse than raw GFS at district-day scale). Validation Day-1 RMSE: 11.75 (GBM base)
  vs 12.90 (QM base) vs 11.75 (regime-agnostic GBM) vs 13.50 (raw).

Results on the held-out seasons are in the README (§10) and on the console's Skill lab page.

## Splits

Defined once in `src/config.py`: train 2021–2022, validation 2023,
held-out test 2024–2025. Nothing from 2024–2025 is used for fitting, tuning,
climatologies or anomaly baselines.

## Honest limits

* GFS ≠ NCUM. Skill relative to NCUM will differ.
* Five seasons is short: extreme (≥ 204.5 mm) events are rare in the training
  years, so that probability model can fall back to a base rate (it says so).
* Regime labels are objective rules, not IMD's operational synoptic analysis; the
  active/break seasonal cycle is estimated from the five seasons themselves.
* District rainfall comes from 0.25° gridded analyses, which smooth station-scale
  extremes (a gauge can record far more than any grid cell).
