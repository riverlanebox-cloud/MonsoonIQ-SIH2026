# MonsoonIQ — Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

**SIH problem statement SIH26080 · Ministry of Earth Sciences (MoES) · Disaster Management · Software**

MonsoonIQ takes a dynamical rainfall forecast — the kind of gridded output an NWP model
produces — and turns it into what a district administration actually needs: a bias-corrected
rainfall field, IMD-scaled district warnings, exceedance probabilities, and a filed bulletin.
The mechanism that distinguishes it is that the correction is **conditioned on the prevailing
monsoon regime**: the same raw rainfall value is corrected differently on a monsoon depression
day than on a break-monsoon day, because the error structures are different.

The repository ships the whole chain: data pipeline → regime classifier → regime-conditional
correction → probability modules → verification harness with confidence intervals → FastAPI
service → operations console → documentation. One command reproduces every number shown in the UI.

> **Real data.** The default archive is **observed**: IMD 0.25° daily gridded rainfall
> (Pai et al. 2014) as truth and archived **NOAA GFS** Day 1–5 forecasts as the raw model,
> June–September 2021–2025, over 53 districts with real Census-2011 boundaries. Models are
> fitted on 2021–2022, tuned on 2023 and verified on the held-out 2024–2025 seasons. GFS is a
> public stand-in for NCMRWF's NCUM, which is not openly downloadable; the chain is
> model-agnostic. How the data is fetched and processed: [docs/REAL_DATA.md](docs/REAL_DATA.md).
> The original seeded simulator is still in the repo for method demos (`make synthetic`).

---

## What's new

**Real data is now the default.** Truth is IMD 0.25° daily gridded rainfall, and the raw model is
NOAA GFS Day 1–5 forecasts, for June–September 2021–2025 over 53 districts with real Census-2011
boundaries. Models are trained on 2021–2022, tuned on 2023, and verified on the held-out 2024–2025
seasons (12,402 district-days, Day 1):

| | Raw GFS | Regime-agnostic GBM | **MonsoonIQ** |
|---|---|---|---|
| RMSE (mm/day) | 14.75 | 13.08 | **13.06** (−11% vs raw, significant) |
| Heavy-rain CSI (≥ 64.5 mm) | 0.085 | 0.167 | **0.155** (+82% vs raw, significant) |

Regime conditioning ties the regime-agnostic learner rather than beating it; §10 explains why, and
the console says so on screen. The full method is in [docs/REAL_DATA.md](docs/REAL_DATA.md);
rebuild everything from public data with `make fetch-real && make real`.

**New console look.** A SAGAR-style interface: a dot-matrix India landing map coloured by live
warnings, glass cards, icon navigation, an About dialog with data sources, and page headings.

**Hosted on Vercel.** The console can be deployed as a static site: at build time the trained
models answer every request the console can make (all 585 archived days × 5 lead days) and the
answers are served as files, so no server is needed. Import the repo in Vercel and it builds with
the committed `vercel.json`; details in [docs/DEPLOY.md](docs/DEPLOY.md).

**Environment notes.** `requirements.txt` pins scikit-learn 1.8, which the committed models were
fitted with. When LightGBM is missing, `src/compat.py` provides a scikit-learn-backed stand-in, and
every training manifest records which backend produced it.

---

## Table of contents

0. [What's new](#whats-new)
1. [The problem](#1-the-problem)
2. [What is built](#2-what-is-built)
3. [System architecture](#3-system-architecture)
4. [Module flow and communication](#4-module-flow-and-communication)
5. [User flow (the operations console)](#5-user-flow-the-operations-console)
6. [Data flow and schemas](#6-data-flow-and-schemas)
7. [Repository layout](#7-repository-layout)
8. [Quickstart](#8-quickstart)
9. [Data provenance and honesty](#9-data-provenance-and-honesty)
10. [Verification results](#10-verification-results)
11. [Design decisions and trade-offs](#11-design-decisions-and-trade-offs)
12. [Research basis](#12-research-basis)
13. [Competitive positioning vs typical SIH submissions](#13-competitive-positioning-vs-typical-sih-submissions)
14. [Limitations, risks and what would break in operations](#14-limitations-risks-and-what-would-break-in-operations)
15. [Roadmap: from this prototype to operations](#15-roadmap-from-this-prototype-to-operations)
16. [Testing and reproducibility](#16-testing-and-reproducibility)

---

## 1. The problem

Numerical weather prediction systematically under-forecasts heavy rainfall. Models conserve
mass and energy at grid scale; convective extremes are sub-grid, so the field is smoothed and
the maximum is damped. For a district administration deciding whether to preposition NDRF
teams, a forecast that says "40 mm" when the truth is "200 mm" is worse than useless — it is
misleading in the direction of inaction.

Post-processing (MOS, quantile mapping, EMOS) is the standard operational answer, and IMD's own
MOS guidance is the reference implementation. Its weakness is that it is usually *global*: one
correction function fitted across all synoptic situations.

Monsoon rainfall is not one situation. A monsoon depression produces organised, widespread,
large-scale rainfall. A break-monsoon day produces suppressed rainfall over central India but
intense convection along the Himalayan foothills. An orographic event under a strong low-level
jet produces a narrow, extreme coastal/ghat signature. A global correction averages over these
regimes and is therefore wrong in each of them in a different way.

**The hypothesis this project tests:** fitting the correction per regime — and letting a
classifier's *soft* posterior, not a hard label, drive a mixture of experts — beats a strong
regime-agnostic learner on the metrics that matter for warnings.

**The finding, stated up front, because it belongs next to the claim:** the hypothesis holds for
continuous error and holds decisively against the raw model, but on the heavy-rainfall
categorical score the regime-conditioned system is **statistically indistinguishable from a
regime-agnostic gradient-boosting learner fitted on the same predictors** (ΔCSI +0.0075,
95% CI [−0.0125, +0.0230], one-sided P(Δ>0) = 0.79). That result is shipped in the UI, in the PDF report,
and in `artifacts/metrics/verification_summary.json`. Section
[Verification results](#10-verification-results) explains why we do not paper over it.

---

## 2. What is built

| Capability | How |
|---|---|
| **Regime classification on published criteria** | 7 regimes (Active Monsoon, Break Monsoon, Monsoon Low/Depression, Orographic, Coastal, Western Disturbance, Weak/Normal) from rule thresholds on 850 hPa jet, relative vorticity, moisture flux convergence, OLR, CAPE and terrain (`configs/regime_rules.yaml`), plus a gradient-boosted classifier that emits calibrated **posteriors** over 25+ dynamical features. |
| **Regime-conditional bias correction** | A mixture of experts: 7 regime-specialised residual experts blended by classifier posterior, with quantile mapping and a regime-agnostic GBM as reference systems. |
| **Per-lead models** | Separate fits for Day 1–5, because the error structure changes with lead time (`src/correction/lead_time_models.py`). |
| **Exceedance probabilities** | Dedicated calibrated classifiers for P(≥64.5 mm), P(≥115.6 mm), P(≥204.5 mm) and quantile regression for P10/P50/P90. |
| **Grid and district products** | District-scale correction *and* a grid-native correction trained on 0.5° cells (`src/correction/grid_correction.py`), because transferring a district correction to grid cells measurably hurts (see §10). |
| **Verification harness** | ETS, CSI, POD, FAR, frequency bias, RMSE/MAE/bias/correlation, FSS with neighbourhood windows, Brier score and reliability diagrams, day-block bootstrap CIs, regime/zone stratification with thin strata suppressed, and an ablation across four systems. |
| **Operations console** | React console: warning map in IMD colours, season timeline, ranked significant days, district detail rail, bilingual bulletin, CSV export, keyboard-first navigation. |
| **API** | FastAPI: one payload per screen, precomputed console artifacts, model card, provenance fields, PDF report. |
| **Honesty layer** | Every artifact carries `data_provenance`; claims are machine-verifiable rows with verdicts; negative results are product features, not omissions. |

---

## 3. System architecture

```
                    ┌────────────────────────── OFFLINE (training) ──────────────────────────┐
                    │                                                                        │
  real IMD + GFS    │  scripts/build_real_archive.py ──▶ data/real/district_daily.parquet    │
  (default; or the  │        │                        (53 districts × 585 days, JJAS 2021-25) │
   synthetic sim.)  │        │                                                                │
                    │        ▼                                                                │
                    │  scripts/build_grid_samples.py ──▶ grid_feature_samples.npz (0.5° grid) │
                    │        │                                                                │
                    │        ▼                                                                │
                    │  src/regime/  (rules + ML classifier, posteriors)                       │
                    │        │                                                                │
                    │        ├──▶ src/correction/mixture_of_experts.py     (point correction) │
                    │        ├──▶ src/correction/quantile_regressor.py     (P10/P50/P90)      │
                    │        ├──▶ src/correction/lead_time_models.py       (Day 1–5 bundle)   │
                    │        ├──▶ src/heavy_rain/heavy_rain_classifier.py  (exceedance)      │
                    │        └──▶ src/correction/grid_correction.py        (grid-native)     │
                    │        ▼                                                                │
                    │  artifacts/models/*.joblib  +  artifacts/models/training_manifest.json  │
                    │        ▼                                                                │
                    │  src/evaluate.py ──▶ src/verification/* ──▶ artifacts/metrics/*.json    │
                    │                                          artifacts/reports/*.pdf      │
                    │        ▼                                                                │
                    │  src/console/artifacts.py ──▶ artifacts/console/{timeline,events,meta}  │
                    └────────────────────────────────┬───────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────▼─── ONLINE (serving) ──────────────────┐
                    │   src/api/main.py  (FastAPI)                                           │
                    │     /console  /district/{id}  /bulletin  /grid  /timeline  /events     │
                    │     /verification/*  /model-card  /export/districts.csv  /health       │
                    │            │                                                           │
                    │            ▼                                                           │
                    │   frontend/dist  (React console served by the same process)            │
                    │     Today · Skill lab · Method                                         │
                    └────────────────────────────────────────────────────────────────────────┘
```

**Serving characteristics.** A console screen is one HTTP request: `/console?date=…&lead=…`
returns the summary, the regime, the ranked district table, the five-day outlook and the map
payload together. Inference for one date is ~25 ms, so no precompute is needed for the daily
product; the season timeline (2,088 days) and the ranked significant-day list are precomputed
because they are identical for every user.

---

## 4. Module flow and communication

| Module | Responsibility | Consumes | Produces |
|---|---|---|---|
| `src/data/generator.py` | Physically-parameterised synthetic archive: monsoon regimes, terrain, orographic/coastal effects, NWP-like error model | seed, domain config | `data/synthetic/district_daily.parquet`, metadata |
| `scripts/build_grid_samples.py` | Re-simulates grid fields on a deterministic sampling policy (held-out years sampled densely) | generator seed, `grid_sample_dates.npz` | `data/synthetic/grid_feature_samples.npz` (57×57 grid, 19 fields) |
| `src/regime/rules.py` | Threshold rules on published criteria | raw fields, `configs/regime_rules.yaml` | rule-based regime label + fired criteria |
| `src/regime/ml_classifier.py` | Gradient-boosted classifier over 25+ features; `predict_proba`, `explain_sample` | engineered features | posterior matrix (53 districts × 7 regimes) |
| `src/correction/quantile_mapping.py` | Distributional baseline (the "classic" method) | raw NWP, observations | corrected value |
| `src/correction/residual_expert.py` | One expert per regime, fitted on that regime's residual structure | features + regime mask | corrected value |
| `src/correction/mixture_of_experts.py` | Blend experts by posterior; also blends with the agnostic learner | posteriors, features | `monsooniq`, `global_qm`, `global_lgb`, `raw_nwp` |
| `src/correction/lead_time_models.py` | Day 1–5 bundle: per-lead correction, quantiles and probabilities | features + lead | `day_1…day_5` systems |
| `src/correction/quantile_regressor.py` | P10/P50/P90 | features | quantile fields |
| `src/heavy_rain/heavy_rain_classifier.py` | Calibrated exceedance probabilities | features | `p_heavy`, `p_very_heavy`, `p_extremely_heavy` |
| `src/correction/grid_correction.py` | Grid-native correction (LightGBM on grid cells) | grid predictors + posteriors | corrected grid fields |
| `src/verification/*` | Scoring, bootstrap CIs, stratification, FSS, significance tests, PDF report | model outputs + observations | `artifacts/metrics/*.json`, `artifacts/reports/*.pdf` |
| `src/api/*` | Serving, advisory text, CAP alert fields | artifacts + models | JSON/CSV/PDF |
| `src/console/artifacts.py` | Season timeline + ranked significant days | fitted models | `artifacts/console/*.json` |
| `frontend/src/*` | Console UI | API | screens |

**Interfaces that matter.** Model objects are plain joblib artifacts with `.load()`/`.save()`
and a `predict_*` signature taking a DataFrame of engineered features — this is what makes the
ablation honest: every reference system is called through the same interface on the same rows.
The verification layer never touches a model directly; it consumes the prediction columns, so it
can score a future system without modification.

---

## 5. User flow (the operations console)

The console is designed for a duty officer with a deadline, not for a data scientist exploring.
Every screen is a fixed layout; every state change is one round trip.

```
OPEN  ──▶ Overview screen (0 clicks, no map controls to learn)
          • full-bleed map of India, every district tinted by its warning category on the
            most warning-heavy day of the monsoon season, hover for the values
          • one line of state: date · districts warned · regime and confidence ·
            peak corrected against peak raw
          • one primary button, "Enter operations console", and one secondary,
            "Verification & method" — the two things a visitor can want first
             │
        ──▶ Today console (1 click, or the `h` key to come back)
          • warning map in IMD colours, ranked district table, regime banner,
            five-day outlook, correction-impact list, documented-case strip
             │
             ├─ change day (1 click / 1 keystroke)
             │    timeline drag · ← → · date picker · "jump to significant day"
             │
             ├─ change lead (1 click / 1 keystroke)   1…5
             │
             ├─ inspect a district (1 click)
             │    map click or table click → rail: corrected P10–P90 band vs raw,
             │    regime posterior bars, advisory text (EN + HI), CAP fields,
             │    Day 1–5 consistency
             │
             ├─ issue (1 click)
             │    Bulletin (bilingual, copy/print) · CSV export
             │
             ├─ prove it (1 click)
             │    Skill lab: scorecard, claims with verdicts, per-threshold scores,
             │    lead-time table, regime strata, reliability diagrams, FSS,
             │    regime-value audit, PDF report
             │
             ├─ defend it (1 click)
             │    Method: architecture, module map, user flow, design rules,
             │    research basis, model card, stated limits
             │
             ├─ check a case (1 click)
             │    Documented cases (Kerala 2018, Konkan 2019, …) load that day
             │
             └─ integrate it (1 click)
                  API: every read endpoint, rendered from the live OpenAPI document,
                  each with a one-click probe showing status, latency and payload size
```

**Click budget.** A full operational cycle — *which day is worst → what do I warn → who signs
off → is the system any good* — is 4 clicks. A jury sees a live corrected field before the first
click: the overview screen renders the warning map itself, so the pitch starts from weather
rather than from a login screen. Keyboard-first throughout: `←/→` days, `1–5` lead, `n/p`
significant day, `l` layer, `b` bulletin, `s` story mode, `h` overview, `?` shortcuts.

**Why this shape.** SIH scoring splits roughly 30% problem understanding / 25% technical
implementation / 20% innovation / 15% feasibility / 10% presentation, and the recorded failure
modes of previous submissions are consistent: solutions that are adjacent to the problem
statement, demos that break, no measurable impact story, and teams that cannot defend their own
assumptions behind a chat box. The console therefore (a) speaks in IMD's own warning vocabulary,
(b) never hides the raw forecast behind the correction, (c) shows the failing statistical test
next to the passing ones, and (d) runs the entire demo path offline from local artifacts.

### 5.1 Demo script for a jury (about four minutes, five clicks)

| Step | Do | Say |
|---|---|---|
| 0 | Open the app | "Before you click anything: this is the country on the most warning-heavy day of the season, every district coloured on IMD's own scale, and the numbers underneath are the corrected forecast against the raw one." |
| 1 | Click **Enter operations console**, point at the map | "Green to red is IMD's own warning scale. The number in each box is the corrected value; the raw model value is one layer switch away, so we are never hiding the adjustment." |
| 2 | Click the top district (**click 1**) | "This is the rail: the P10–P90 band, the raw model tick inside it, the regime posteriors that actually fed the correction, the advisory in English and Hindi, and the CAP fields." |
| 3 | Press **5** for Day 5 | "Same district, five days out. Per-lead models, so the correction is refitted, not reused." |
| 4 | Open **Bulletin** (**click 2**) and copy | "This is what the SDMA files. Machine-generated, rule-based, bilingual, with the basis and the provenance printed on it." |
| 5 | Open **Skill lab** (**click 3**) | "And here is the part most teams skip: the claim table. Three claims supported, one not supported — regime conditioning does not significantly beat a regime-agnostic learner on the heavy-rain categorical score. We publish that, in the product and in the report." |
| 6 | Open **Method** (**click 4**) | "Architecture, module interfaces, research basis, model card, stated limits, and the exact commands that reproduce every number you just saw." |
| 7 | Open **API** (**click 5**) and press **Send** on any row | "The same corrections are available to a state EOC as JSON — status and latency shown live, no integration meeting needed." |
| 8 | Run `python -m pytest tests/ -q` if challenged | "34 tests, including payload contracts and leakage guards." |

The whole path works with no network: data, models, artifacts and UI are all local.

### 5.2 Interface design

The interface is a deliberate counter-position to the chat-box submission. It is an operations
console: a dark surface that reads as instrumentation, colour reserved for weather, and no
conversational framing anywhere in the product.

| Rule | Why |
|---|---|
| Colour means weather, not decoration | Green, yellow, orange and red are IMD's warning categories and nothing else. The rainfall ramp runs pale cyan to deep violet; the correction map runs blue for a downward adjustment, amber for an upward one. |
| One accent, used to mean "live" | A single cyan marks interactive state — the active control, the selected district, loaded data, the progress rail while a request is in flight. Nothing glows to look futuristic. |
| Raw is always adjacent to corrected | Every corrected value on screen sits next to the raw model value and the applied adjustment, because the whole product claim is the size of that adjustment. |
| The failing test stays on the page | The regime-conditioning claim is rendered in the same table as the supported ones, with its confidence interval and its NOT SUPPORTED verdict. |
| Motion is state, not garnish | There is no entrance animation to sit through. The only moving elements are the loading rail and the story-mode stepper. |
| Readable under a projector | Every foreground colour was measured against all four surfaces: the lowest ratio in the palette is 4.50:1, above the WCAG AA floor of 4.5:1; primary text sits at 14.9–17.0:1 and secondary at 10.3–11.8:1. |
| Keyboard-first | `←/→` days, `1–5` lead day, `n/p` next or previous significant day, `l` map layer, `b` bulletin, `s` story mode, `h` overview, `?` shortcut list. A duty officer on a phone call never has to find a control. |
| Degrade, never blank | The map sits in an error boundary; if tiles or canvas are unavailable the table, warnings and advisory still render, with a sentence saying what failed. |
| A warning can be audited | Every category on screen carries the criterion that produced it — predicted amount or exceedance probability, with the threshold that was crossed. Probability-driven warnings say so explicitly, because those are the ones a duty officer should double-check. |
| Say when data is stale | If the engine does not answer for a date, the console keeps the last successful load on screen, dims it, names the date it actually belongs to, shows the HTTP detail and offers a retry — rather than silently showing yesterday's field under today's heading. |

**The opening frame.** The first screen is the forecast itself: a full-bleed map of India with every
district tinted by its warning category for the most warning-heavy day of the season, a strip of
state underneath it, and one primary button. A visitor learns what the product does before deciding
to explore it, and a jury sees a corrected rainfall field within a second of the page loading.

**The API reference.** The last tab documents the service from its own OpenAPI document, so the
reference cannot drift from the code. Each endpoint carries a **Send** button that issues a real
probe against a date in the archive and reports status, latency and payload size — the claim
"integration-ready" is demonstrated rather than asserted.

---

## 6. Data flow and schemas

The active archive is chosen in `src/config.py` (`MONSOONIQ_DATA=real|synthetic`; real is the
default once built). Both archives share one schema, so every module downstream is identical.

**Real inputs** — `data/raw/bundles/` (not committed): IMD 0.25° daily rainfall for JJAS
2021–2025, and per season the GFS 00 UTC Day 1–5 rainfall on the IMD 03–03 UTC day (0.5°) plus
nine predictors per lead day (1°). `scripts/fetch_real_data.py` writes them;
`scripts/build_real_archive.py` turns them into `data/real/` (details in docs/REAL_DATA.md).
The real archive adds per-lead predictor columns (`u850_d3`, `olr_anomaly_d5`, …): the Day-*k*
models only see what a Day-*k* forecast knew.

**District archive** — `data/<mode>/district_daily.parquet` (+ `.csv.gz`), one row per district-day:

| Group | Fields |
|---|---|
| Keys | `date`, `district_id`, `district_name`, `state_name`, `zone`, `latitude`, `longitude`, `elevation` |
| Dynamics | `u850`, `v850`, `vorticity_850`, `moisture_flux`, `q500`, `cape`, `olr`, `mslp` |
| Raw forecasts | `raw_nwp_d1` … `raw_nwp_d5` (model rainfall, one per lead) |
| Truth | `obs_rain_max`, `obs_rain_mean`, `regime`, `regime_name`, heavy/hervery-heavy flags |

**Grid archive** — `data/<mode>/grid_feature_samples.npz`, 57×57 grid (0.5°), 19 fields
per day, `true_rain`, `raw_nwp_d1/d3`, dynamical fields, `land_index`, elevation.

**API payloads** — see `/docs` (OpenAPI) when the service is running. The two that matter:

* `GET /console?date&lead` → `{ summary, regime, horizon[], districts[], navigation, provenance }`
* `GET /district/{id}?date&lead` → distribution, posteriors, advisory, CAP, `lead_trend[]`

**Model outputs** — `artifacts/models/` holds `regime_classifier.joblib`,
`mixture_of_experts.joblib`, `lead_time_bundle.joblib`, `heavy_rain_module.joblib`,
`quantile_regressor.joblib`, `grid_correction.joblib` and a `training_manifest.json` recording
split, sample counts, timings and artifact hashes.

**Verification outputs** — `artifacts/metrics/verification_summary.json` (continuous,
categorical, per-lead, stratified, probabilistic, grid FSS, scorecard, significance statement),
`heavy_events_summary.json`, `regime_value_audit.json`, and
`artifacts/reports/MonsoonIQ_Verification_Report.pdf`.

---

## 7. Repository layout

```
├── configs/regime_rules.yaml        # published criteria for the 7 regimes
├── data/synthetic/                  # generated archive + geo metadata (not committed if large)
├── src/
│   ├── data/                        # generator and metadata
│   ├── regime/                      # rules + ML classifier
│   ├── correction/                  # QM, experts, MoE, quantiles, per-lead, grid
│   ├── heavy_rain/                  # exceedance classifiers
│   ├── verification/                # metrics, CIs, stratification, FSS, report
│   ├── console/                     # precomputed timeline + ranked days
│   ├── api/                         # FastAPI service, schemas, advisories
│   ├── train.py                     # 6-stage training pipeline
│   └── evaluate.py                  # verification entry point
├── frontend/                        # React 19 + Vite console (Today / Skill lab / Method)
│   ├── scripts/smoke.mjs            # jsdom render test against recorded fixtures
│   └── fixtures/                    # recorded API responses for the smoke test
├── scripts/                         # grid sample builder, API fixture recorder
├── tests/                           # pytest suite
├── artifacts/                       # models, metrics, reports, console payloads
├── AUDIT.md                         # phase-1 adversarial audit of this project's own claims
├── Dockerfile / docker-compose.yml  # containerised service
└── Makefile                         # data → train → evaluate → serve
```

---

## 8. Quickstart

```bash
# Python environment
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Real data (default). The processed archive and trained models are committed, so
# `make serve` works straight away; these steps rebuild everything from source.
make fetch-real  # IMD + GFS bundles into data/raw/bundles (~3.5 GB transferred, network-bound)
make real        # build data/real/, then train, evaluate, console artifacts (~3 min)
make serve       # FastAPI on :8000, serving the built console

# Synthetic simulator (method demos): every screen switches to the seeded archive
make synthetic   # or MONSOONIQ_DATA=synthetic make serve

# Frontend (development)
cd frontend && npm install && npm run dev     # http://localhost:3000, proxies /api to :8000
npm run build                                 # production bundle served by the API at /
npm run smoke                                 # headless render test of the whole app
```

Docker: `docker compose up --build` starts the API and the console on port 8000.

Static hosting (Vercel, or any static host): `bash scripts/vercel_build.sh` exports every API
response the console needs into `frontend/public/snapshot/` and builds the console against it
(`VITE_STATIC=1`). On Vercel this runs automatically; see [docs/DEPLOY.md](docs/DEPLOY.md).

---

## 9. Data provenance and honesty

**What is real (default archive).** Observed rainfall: IMD Pune 0.25° daily gridded analysis
(Pai et al. 2014). Raw forecasts and every dynamical predictor: NOAA GFS 00 UTC runs from the
AWS Open Data archive, decoded from the original GRIB2 messages. District boundaries: Census 2011
(DataMeet, CC BY 2.5 India). Terrain: GFS orography and land mask. The IMD seasonal totals
reproduced from the fetched files match IMD's published All-India JJAS rainfall (e.g. 2023:
816 mm here vs 820 mm published), and the forecast-day/IMD-day alignment is measured from the
data (correlation 0.93 at the chosen lag, vs 0.77 without the shift).

**What is not observed.** Regime labels: real data carries no "true" regime, so each day is
labelled by objective rules (Rajeevan et al. 2010 active/break index on IMD rainfall; GFS
MSLP/vorticity for lows; zone-rainfall spells; 500 hPa height for western disturbances), in
`configs/regime_labels_real.yaml`. The ML classifier must recover those labels from GFS dynamics
alone. The raw model is GFS, not NCMRWF NCUM (not publicly downloadable).

**What that means.** Skill numbers on the default archive are genuine out-of-sample verification
of a post-processing chain against IMD observations for 2024–2025, over 53 districts. They are
not an operational IMD/NCMRWF evaluation: one global model, five seasons, gridded (not station)
truth.

**No leakage by construction.** Splits live in one place (`src/config.py`): train 2021–2022,
validation 2023, test 2024–2025. Anomaly climatologies use the training years only; the Day-*k*
models use Day-*k* forecast predictors only; `tests/test_leakage.py` checks the split.

**The synthetic simulator.** Still in the repository (`src/data/synthetic_generator.py`,
`make synthetic`) for method demos and unit tests. On that archive the skill figures measure
internal consistency of the pipeline only, and every screen says so when it is active.

`AUDIT.md` in this repository is a phase-1 adversarial audit of this project's own claims; it is
kept because the corrections it forced are part of the design.

---

## 10. Verification results

**Real data, held-out seasons 2024–2025** (never used for fitting, tuning or anomaly
baselines): 234 calendar days × 53 districts = 12,402 district-days. Observations: IMD 0.25°
gridded rainfall. Raw model: NOAA GFS. All comparisons are paired on the same rows with day-block
bootstrap 95% CIs. Every number below is in `artifacts/metrics/verification_summary.json`.

**Systems compared.** `raw_nwp` (GFS), `global_qm` (quantile mapping), `global_lgb` (gradient
boosting on the same predictors *without* regime information — the honest baseline), `monsooniq`
(regime-aware mixture of experts + exceedance modules).

**Day 1** (district mean rainfall for RMSE; district maximum for the categorical scores, as IMD
district warnings are issued on the heaviest rainfall in the district)

| System | RMSE (mm/day) | Bias | CSI ≥ 15.6 | CSI ≥ 64.5 | POD ≥ 64.5 | FAR ≥ 64.5 |
|---|---|---|---|---|---|---|
| Raw GFS | 14.75 | −0.89 | 0.344 | 0.085 | 0.088 | 0.319 |
| Quantile mapping | 16.78 | +0.87 | 0.399 | 0.162 | 0.179 | 0.372 |
| GBM (regime-agnostic) | 13.08 | +0.94 | 0.388 | 0.167 | 0.173 | 0.167 |
| **MonsoonIQ** | **13.06** | +0.68 | 0.384 | 0.155 | 0.160 | **0.157** |

**Skill by lead** (RMSE mm/day / CSI ≥ 64.5)

| Lead | Raw GFS | GBM (agnostic) | MonsoonIQ |
|---|---|---|---|
| Day 1 | 14.75 / 0.085 | 13.08 / 0.167 | 13.06 / 0.155 |
| Day 2 | 15.77 / 0.065 | 14.20 / 0.149 | 14.18 / 0.134 |
| Day 3 | 16.60 / 0.041 | 14.67 / 0.093 | 14.68 / 0.075 |
| Day 4 | 17.08 / 0.041 | 15.03 / 0.090 | 15.02 / 0.084 |
| Day 5 | 17.27 / 0.037 | 15.24 / 0.079 | 15.12 / 0.076 |

**Claims and verdicts** (the machine-checked rows the UI displays)

| Claim | Evidence | Verdict |
|---|---|---|
| Post-processing reduces RMSE vs raw GFS | ΔRMSE −1.655 mm/day, CI [−2.206, −1.181] | **supported** (−11%) |
| Heavy-rain categorical score improves vs raw GFS | ΔCSI +0.070, CI [+0.047, +0.098] | **supported** (+82%) |
| Regime conditioning beats the regime-agnostic model on RMSE | ΔRMSE −0.025, CI [−0.096, +0.053] | **not supported** (tie) |
| Regime conditioning beats the regime-agnostic model on heavy CSI | ΔCSI −0.012, CI [−0.023, −0.003] | **not supported** (slightly worse) |

**Exceedance probabilities are the warning engine, and they are reliable.** P(≥64.5 mm):
Brier 0.061, mean forecast probability 0.098 vs observed frequency 0.098. P(≥115.6 mm): Brier
0.022, 0.029 vs 0.028. P(≥204.5 mm): Brier 0.0035, 0.0031 vs 0.0037 (46 events). All three
reliability diagrams are classed reliable. Validation AUCs (2023): 0.89 / 0.94 / 0.96.

**What the regime-value ablation says.** The regime classifier recovers the objective day-level
labels from GFS dynamics with 64% validation accuracy (majority class: 61%). With the *true*
regime supplied (oracle), the mixture reaches RMSE 12.95 and CSI 0.165 — better than the
regime-agnostic model on RMSE — so regime information is useful; the accuracy sweep puts
break-even with the agnostic model at ~65% classifier accuracy, right where the classifier is.
On real data the bottleneck is regime *recognition*, exactly what the synthetic audit predicted.

**What does not work (and is shown on screen).** A mean-targeted correction shrinks extremes: at
≥115.6 mm the corrected point forecast detects fewer events than raw GFS (POD 0.009 vs 0.047).
Warnings therefore do not rely on the point forecast at that level; the console's categories
use the calibrated exceedance probabilities and the P90 band, and say which criterion fired.

**Design choice made on validation, not test.** The original experts corrected a regime
quantile-mapping base; on real district-day data quantile mapping is worse than raw GFS (RMSE
16.8), so experts built on it could not recover. On the 2023 validation season, experts on the
GBM's out-of-fold prediction (shrunk towards zero for rare regimes) matched the agnostic model
(Day-1 RMSE 11.75 vs 11.75) where the QM-based ones did not (12.90). That base is used on the real
archive; the synthetic archive keeps the original.

**Synthetic archive** (`make synthetic`): the previous results on the seeded simulator are
reproducible with the same harness; they measure pipeline consistency only.

---

## 11. Design decisions and trade-offs

| Decision | Alternative | Why |
|---|---|---|
| Soft posterior weighting of experts | Hard regime label | Hard labels create discontinuities at regime transitions — the days where forecasts matter most. |
| Keep a regime-agnostic GBM as a shipped reference system | Compare only against raw NWP | Beating raw NWP is table stakes; a post-processing project that only beats the raw model has shown almost nothing. The honest baseline is what exposes the categorical result. |
| Per-lead models rather than one model with a lead feature | Single model + lead input | Error growth differs structurally by lead; per-lead fits also let the UI show degradation honestly. Cost: 5× training time (still ~3 minutes total). |
| Grid-native correction alongside district correction | Transfer district correction to grid | The transfer was measured, and it loses. Keeping the losing variant as a labelled negative control is more valuable than deleting it. |
| Precomputed season timeline | Compute per request | 2,088 days × 5 leads of inference per page load would be ~2.7 s; precompute makes the scrubber instant. |
| One API payload per screen | REST-shaped per-field endpoints | The console's job is a single screen state; a fan-out of requests adds latency and partial-failure states for no benefit. |
| Real IMD observations + archived GFS, verified on held-out seasons | Invented "validated against IMD" claims | Every skill number is recomputable from public data with `make fetch-real && make real`. |
| No chat interface, no LLM in the loop | "Ask the forecast" assistant | The user is a duty officer with a deadline and a warning scale, not a conversationalist. Model inputs (regime posteriors) are shown directly instead of being narrated. |

---

## 12. Research basis

* **IMD MOS guidance and downscaling practice** (IMD Pune training material,
  `imdpune.gov.in/…/NWP-TRAINING-MOS-DOWNSCALING`, lecture 2) — heavy-rain
  post-processing in Indian operations is statistical correction of NWP fields with
  recent-history updating. MonsoonIQ is in that family, with regime conditioning added and
  evaluated rather than assumed.
* **QJRMS 2024: quantile mapping vs EMOS for precipitation over India**
  ([doi:10.1002/qj.4677](https://doi.org/10.1002/qj.4677), IMD 0.25° V6.9, 2018–2022) — establishes heavy-precipitation post-processing over India as an active published
  research area and fixes the standard error metrics; both methods appear here as baselines.
* **IMD impact-based warning scale** — ≥64.5 mm watch (be aware), ≥115.6 mm alert (be prepared),
  ≥204.5 mm warning (take action) per 24 h, valid up to 5 days, as published by IMD and reported
  in national press during monsoon events. This drives the colour scale, the advisory wording, the CAP alert
  fields and the bulletin format, so the product speaks the language already in use.
* **FSS with multiple neighbourhood windows** (Roberts & Lean) — used alongside district CSI
  because a smoothed forecast can win on district averages while displacing the rainfall centre;
  the included negative control demonstrates that failure mode with our own numbers.
* **Verification practice for small samples** — day-block bootstrap resampling (weather is
  autocorrelated within ~2 days), thin-stratum suppression, and explicit reliability labels
  (`reliable` ≥ 10 events / `indicative` / `insufficient`) rather than point estimates presented
  as skill.
* **SIH evaluation criteria and prize-winning post-mortems** (published guides and winner
  write-ups: problem understanding ≈ 30%, technical implementation ≈ 25%, innovation ≈ 20%,
  feasibility/scalability ≈ 15%, presentation ≈ 10%; the recurring judge comment is that most
  teams "built something adjacent to the problem statement" and that a solution nobody can defend
  without a chatbot is not defensible). This is why the console is one-click, the statistics are
  intervaled, and the failing test is on screen.

---

## 13. Competitive positioning vs typical SIH submissions

| Common pattern in submissions | What MonsoonIQ does instead |
|---|---|
| Trains a model, reports accuracy, stops | Ships the full chain to a filed bulletin, with the verification of *every* reference system. |
| Claims improvement over the raw model only | Publishes the strong regime-agnostic baseline and the test it does **not** win. |
| Single headline number, no interval | Paired day-block bootstrap CIs on every claim; strata with <10 events are suppressed. |
| District-scale metric hides spatial error | FSS on the grid, with the district→grid transfer included as a measured negative result. |
| Dashboard with a chart of predictions | Operations console: IMD colour semantics, ranked warning table, bilingual bulletin, CSV hand-off, keyboard-only operation. |
| "AI" framing, chat box, gradient hero section | Instrument-like console: dense tables, hairline rules, colour reserved for the warning scale. |
| Research verification presented as operational | Provenance field on every artifact, stated on screen: real IMD + GFS, held-out 2024–2025, not an official IMD/NCMRWF product. |
| Demo depends on the network | Whole demo path (data, models, artifacts, console) runs locally from the repository. |
| Cannot run the pipeline live | `make data && make train && make evaluate` reproduces every displayed figure in minutes. |

**Rubric mapping** — where the evidence for each scoring axis actually lives:

| Rubric axis (≈weight) | Evidence in this repository |
|---|---|
| Problem understanding (30%) | Regime taxonomy taken from published criteria; IMD warning thresholds and impact language used verbatim; the problem is framed as *decision support under a deadline*, and the console is built for that, not for exploration. |
| Technical implementation (25%) | Four systems implemented and scored through one interface; day-block bootstrap CIs; grid FSS with neighbourhood windows; calibrated exceedance probabilities with reliability diagrams; per-lead models; 34 tests; deterministic data generation with a determinism gate. |
| Innovation (20%) | Regime-conditional correction driven by soft posteriors rather than hard labels; grid-native correction adopted *after* measuring that district→grid transfer fails; a shipped significance layer that reports its own unsupported claim. |
| Feasibility / scalability (15%) | One process serves API and console; inference ~25 ms per date so nothing needs precompute; precomputed artifacts only for shared views; containerised; the synthetic-to-real swap is a data-path change, documented in §15. |
| Presentation (10%) | A four-click demo path (§5.1), a dense console that looks like an operations tool, a PDF verification report, and this document. |

The differentiator is not the mixture of experts — it is that the project **measures itself
honestly and ships the measurement in the product**. That is also what makes the method
transferable: the harness survived the move to real data unchanged, and it is what reported
that the regime-aware model ties rather than wins.

---

## 14. Limitations, risks and what would break in operations

1. **GFS, five seasons, gridded truth.** Verification is real but not operational: the raw model
   is GFS (NCUM is not public), there are only five monsoon seasons, and IMD 0.25° analyses
   smooth station-scale extremes. See §9 and docs/REAL_DATA.md.
2. **Regime conditioning does not beat a regime-agnostic learner on real data.** It ties on RMSE
   and is slightly worse on heavy-rain CSI. The oracle ablation shows the regimes carry
   information; the classifier (64% accurate) cannot yet deliver it. Fixes: more seasons, IMD's
   own synoptic analyses as labels, better synoptic predictors, regime-specific exceedance heads.
3. **Thin strata.** Five of seven regimes have too few events in the held-out period to support
   a claim. Operationally this is expected (most heavy events are Western Disturbance linked) and
   is why the evaluation suppresses rather than averages.
4. **Coarse spatial resolution.** 0.5° GFS grid and 0.25° IMD truth; real Census-2011 district
   boundaries; sub-kilometre orographic extremes are unresolved.
5. **No real-time ingestion.** The architecture supports it (an ingestion adapter writes the same
   feature frame), but no live scheduler, no GTS/MOSDAC feed, no QC for missing or physically implausible fields in this
   prototype.
6. **Bias towards climatology.** Statistical post-processing cannot invent a rainfall maximum the
   dynamics never placed; the value it adds is restoring amplitude and sharpness, not predicting
   unprecedented events.
7. **Rapid cyclogenesis and transitions.** Regime transitions are the hardest case: posteriors are
   split between regimes and both experts contribute — which is the intended behaviour, but it has
   not been verified on real transition cases.
8. **Advisory text is rule-based.** It is generated from thresholds and regime, in English and
   Hindi, and is deliberately not LLM-generated: a public warning must be traceable to a rule.
9. **The red rule can fire on probability alone, and the very-heavy head is nearly degenerate.**
   `determine_alert_level` promotes a district to red when P(≥115.6 mm) ≥ 0.65, as well as when the
   predicted total reaches 204.5 mm. Measured over 114 sampled dates (6,042 district-days), 13 red
   warnings were issued and **4 of them had no rainfall value above 115.6 mm** — the label came from
   the probability clause alone. The cause is upstream: P(≥115.6 mm) and P(≥64.5 mm) come from
   separately fitted, separately calibrated heads whose outputs are reconciled only by a `min()`
   clamp, so on the fixture date the two are numerically identical for 49 of 53 districts. The
   console now prints the criterion behind every category, which is how this was found, and marks
   probability-driven warnings distinctly. Fixing it means a nested head (P(≥115.6 | ≥64.5)) or a
   monotone ordinal model and a re-run of §10 — a modelling change rather than a display change, so
   it is left visible here rather than quietly patched.

---

## 15. Roadmap: from this prototype to operations

| Phase | Work | Success test |
|---|---|---|
| 1 | Full pipeline + harness + console on a synthetic archive | `make synthetic` reproduces every figure |
| 2 (this repo) | IMD 0.25° gridded rainfall + archived GFS, JJAS 2021–2025 | Done: `make fetch-real && make real`; §10 is recomputed against observations |
| 2b | Swap GFS for NCMRWF NCUM (same grid bundle format) and extend to 2016+ | Same harness; skill table vs NCUM |
| 3 | Regime-specific exceedance heads; more seasons for thin strata | ΔCSI vs agnostic significant at 95% on ≥3 regimes |
| 3b | Nested exceedance head: P(≥115.6 \| ≥64.5) instead of independently calibrated heads reconciled by a clamp (see §14.9) | `p_very_heavy` stops equalling `p_heavy`, and no red warning is issued without a rainfall value or a genuinely distinct probability behind it |
| 4 | Operational ingest (GTS/MOSDAC), scheduler, missing-data QC | Fresh forecast visible in the console within 15 minutes of model availability |
| 5 | Pilot with one state SDMA: warnings alongside the official bulletin, blind | Measured lead-time gain and false-alarm cost in a real monsoon season |

---

## 16. Testing and reproducibility

```bash
python -m pytest tests/ -q          # unit + API tests
cd frontend && npm run smoke        # renders the whole app against recorded fixtures
PYTHONPATH=. python scripts/dump_api_fixtures.py --date 2020-08-05 --lead 1   # re-record fixtures
```

* **Determinism, measured.** The generator is seeded, and repeated runs inside one environment
  are **bit-identical**: three consecutive `make data` runs produced the same SHA-256 for
  `district_daily.parquet`. Across environments, NumPy/BLAS version differences move the archive
  by ~1e-13 (0.65% of cells, largest absolute difference 4.5e-13), which is why
  `scripts/build_grid_samples.py` gates on a 1e-4 tolerance and refuses to write if the
  re-simulated meteorology disagrees with the committed sample.
* **Traceable results.** Every `make evaluate` run records the archive's SHA-256 (first 16 hex)
  in `provenance_detail.archive_sha256_16`, surfaced in the console's Method page and the model
  card. Re-running the whole chain — `make data && make grid && make train && make evaluate &&
  make console` — after regenerating the archive reproduced **all 1,646 published fields
  exactly**; the only difference was the new fingerprint itself. Training records sample counts
  and artifact hashes in `training_manifest.json`.
* **Test counts.** 34 pytest tests (metrics, payload contracts, leakage guards, pipeline smoke,
  bootstrap grouping) and 47 console smoke checks. `tests/test_api.py` locks the payload shapes
  the UI depends on, so a schema change fails in CI rather than in the browser.
* **The console smoke test** (`frontend/scripts/smoke.mjs`) renders the entire application in
  jsdom against recorded API responses — including that the overview screen renders before any
  click, that entering the console and returning to the overview both work, that a documented case
  loads its day, that the API reference degrades to its static list when the spec is unreachable,
  that the theme tokens exist and no light-theme surface was left behind, that an API probe runs
  and reports its outcome, and that keyboard navigation triggers a reload. It also forces the
  console endpoint to fail once, to prove the failure banner appears, marks the load stale, and
  clears on retry. It exists because a production
  build can compile cleanly and still crash on the first payload.
* **Recoverable UI.** The map is wrapped in an error boundary: if the tile server or canvas is
  unavailable, the rest of the console still works, because a demo that dies with a blank screen
  is a failed demo.
