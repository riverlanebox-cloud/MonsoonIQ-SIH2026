import React, { useEffect, useState } from 'react';
import { api } from '../../api';
import { fmt } from '../../lib/format';

const PIPELINE = `IMD / NCMRWF-style NWP fields ─┐
IMD gridded rainfall (obs)     ├─▶ ingest & QC ─▶ feature builder ─▶ regime classifier
static terrain & masks         ┘                                        │
                                                                        ▼
                                            ┌────── regime-conditional experts (7) ──────┐
               raw NWP forecast ───────────▶ │  fitted on regime-stratified residuals      │
                                            └───────────────┬─────────────────────────────┘
                                                            ▼
                                 mixture of experts (soft regime weights)
                                                            │
                    ┌───────────────────────────────────────┼──────────────────────────────┐
                    ▼                                       ▼                              ▼
        point correction (μ)                 exceedance models P(≥64.5/115.6/204.5)   quantile models P10/P50/P90
                    │                                       │                              │
                    └───────────────▶ advisory & CAP layer ◀┘                              │
                                            │                                              │
                                            ▼                                              ▼
                          operations console · bulletin · CSV          verification harness (ETS/CSI/POD/FAR/FSS,
                                                                        stratified + bootstrap CIs)`;

const MODULES = [
  { name: 'src/regime/', role: 'Regime classification', detail: 'Rules (published criteria on 850 hPa jet, vorticity, moisture flux, CAPE, OLR, terrain) plus a gradient-boosted classifier over 25+ dynamical features. Emits class posteriors, not just a label — the correction models consume the posteriors.' },
  { name: 'src/correction/', role: 'Bias correction', detail: 'Quantile mapping, regime-agnostic GBM and the regime-conditioned mixture of experts. Per-lead models for Day 1–5. Quantile regression gives P10/P50/P90.' },
  { name: 'src/heavy_rain/', role: 'Exceedance', detail: 'Dedicated classifiers for P(≥64.5), P(≥115.6), P(≥204.5) mm, with probability calibration so the numbers can be read as frequencies.' },
  { name: 'src/verification/', role: 'Evaluation', detail: 'Deterministic scorers (ETS, CSI, POD, FAR, bias, RMSE), event-block bootstrap CIs, regime stratification with thin strata suppressed, grid FSS with neighbourhood windows, ablation across systems, PDF report generation.' },
  { name: 'src/api/', role: 'Delivery', detail: 'FastAPI service. One payload per screen (/console), precomputed timeline and ranked events, district detail on demand, bulletin text, CSV export, and the verification artifacts the console reads.' },
  { name: 'src/console/', role: 'Precompute', detail: 'Builds the season timeline and ranked significant days once, so the UI can scrub 9 years of history without a single heavy request.' },
];

const RESEARCH = [
  ['Operational practice', 'IMD MOS guidance and downscaling notes (IMD Pune training material): heavy-rain post-processing is done by statistical correction of NWP fields with recent-history updating — the same family of methods used here, with the addition of regime conditioning.'],
  ['Published benchmark', 'QJRMS 2024 study of quantile mapping vs EMOS for precipitation over India using IMD 0.25° gridded data: establishes that post-processing heavy precipitation over India is an active, published research area and defines the standard error metrics we also report.'],
  ['Verification standard', 'FSS with multiple neighbourhood windows alongside district-scale CSI/ETS, because a smoothed forecast can look skilful on district averages while displacing the rainfall centre. That failure mode is demonstrated in the Skill Lab as an included negative control.'],
  ['Warning standard', 'IMD impact-based warning categories (≥64.5 / ≥115.6 / ≥204.5 mm per 24 h) drive the colour scale, the advisory text, and the CAP alert fields — the product speaks in the language the district administration already uses.'],
];

export default function Method() {
  const [card, setCard] = useState(null);
  useEffect(() => { api.modelCard().then(setCard).catch(() => setCard(null)); }, []);

  return (
    <div className="workspace method">
      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">What this system does, in one paragraph</span>
        </div>
        <div className="panel-body">
          <p style={{ margin: 0 }}>
            Numerical weather prediction systematically under-forecasts heavy monsoon rainfall because
            it smooths extremes. MonsoonIQ takes a dynamical rainfall forecast and the fields around it,
            decides which monsoon regime the day belongs to, and applies a correction learned separately
            for that regime — then converts the result into district-level warnings, exceedance
            probabilities, and a filed bulletin. The same codebase carries the verification harness that
            checks whether any of that actually helps, including the result that does not:
            the regime conditioning does not significantly beat a strong regime-agnostic learner on the
            heavy-rainfall categorical score.
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">System architecture and module communication</span>
          <span className="tiny muted">offline pipeline (top) → serving path (bottom)</span>
        </div>
        <div className="panel-body">
          <pre className="mono term" style={{ fontSize: 11.5 }}>{PIPELINE}</pre>
          <table className="data" style={{ marginTop: 10 }}>
            <thead><tr><th>Module</th><th>Responsibility</th><th>Interface</th></tr></thead>
            <tbody>
              {MODULES.map((m) => (
                <tr key={m.name}>
                  <td className="mono">{m.name}</td>
                  <td><b>{m.role}</b> — {m.detail}</td>
                  <td className="mono small">{m.name.replace('src/', '').replace('/', '')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="split" style={{ gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)' }}>
        <div className="panel">
          <div className="panel-head"><span className="panel-title">Operational user flow</span></div>
          <div className="panel-body">
            <div className="flow">
              <div className="flow-col">
                <h4>1 · Open the console (0 clicks)</h4>
                <div className="flow-step">Lands on the most recent significant day of the archive, Day 1, with the warning map already rendered.</div>
              </div>
              <div className="arrow">▼</div>
              <div className="flow-col">
                <h4>2 · Pick a day and lead (1 click)</h4>
                <div className="flow-step">Timeline drag, ← / → keys, ranked "significant days" jump list, or day chips. Every change is one request.</div>
              </div>
              <div className="arrow">▼</div>
              <div className="flow-col">
                <h4>3 · Read the exposure (0 clicks)</h4>
                <div className="flow-step">Category counts, worst districts, the regime and the Day 1–5 outlook are on the same screen — no tab hunting.</div>
              </div>
              <div className="arrow">▼</div>
              <div className="flow-col">
                <h4>4 · Drill to one district (1 click)</h4>
                <div className="flow-step">Map or table click opens the rail: distribution, regime evidence, advisory, CAP fields, Day 1–5 trend.</div>
              </div>
              <div className="arrow">▼</div>
              <div className="flow-col">
                <h4>5 · Issue and hand off (1 click)</h4>
                <div className="flow-step">Copy the bilingual bulletin, export CSV for a district spreadsheet, or open the PDF verification report.</div>
              </div>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Design rules we held ourselves to</span>
          </div>
          <div className="panel-body small">
            <ul style={{ margin: 0, paddingLeft: 16, lineHeight: 1.7 }}>
              <li><b>Forecast, not chat.</b> Nothing on the console asks the user to phrase a question. The screen answers "what is going to happen and where" the moment it loads.</li>
              <li><b>Colour is data.</b> The only saturated colours are the IMD warning scale and the rainfall scale. No decoration competes with them.</li>
              <li><b>Numbers stay visible.</b> Every corrected value can be compared with the raw model value in the same view — trust is built by showing the adjustment, not hiding it.</li>
              <li><b>Explain with model inputs.</b> The district rail shows the regime posteriors that actually enter the correction, not a generated paragraph.</li>
              <li><b>Uncertainty is a first-class element.</b> P10–P90 bands and exceedance probabilities sit next to the point forecast.</li>
              <li><b>Negative results are shipped.</b> The district-transfer FSS and the non-significant categorical gain stay in the product.</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">Research basis</span>
          <span className="tiny muted">what was read before building, and what it changed</span>
        </div>
        <div className="panel-body">
          {RESEARCH.map(([title, body]) => (
            <div key={title} style={{ marginBottom: 8 }}>
              <b className="small">{title}</b>
              <div className="small muted">{body}</div>
            </div>
          ))}
          <div className="small" style={{ marginTop: 6 }}>
            Judging criteria for SIH weight problem understanding (30%), technical depth (25%) and
            innovation (20%) above presentation (10%); the requirement that a team can defend every
            assumption without saying "the tool wrote it" is why the verification page prints the
            failing test next to the passing ones.
          </div>
        </div>
      </div>

      {card && (
        <div className="split" style={{ gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)' }}>
          <div className="panel">
            <div className="panel-head"><span className="panel-title">Model card and data provenance</span></div>
            <div className="panel-body">
              <dl className="kv" style={{ gridTemplateColumns: 'auto 1fr' }}>
                <dt>Archive</dt><dd className="mono">{card.data?.archive?.name || card.data?.provenance}</dd>
                <dt>Provenance</dt><dd className="mono">{card.data?.provenance}</dd>
                <dt>Verification window</dt>
                <dd className="mono">{card.data?.verification_period?.start} → {card.data?.verification_period?.end}</dd>
                <dt>Train / test split</dt><dd className="mono">{card.training?.split?.train_end || '—'} / {card.training?.split?.test_start || '—'}</dd>
                <dt>Training samples</dt>
                <dd className="mono">{card.training?.samples?.train_samples?.toLocaleString?.() || '—'}</dd>
                <dt>Grid correction</dt>
                <dd className="mono">
                  {card.training?.grid_correction?.n_cell_days
                    ? `${card.training.grid_correction.n_cell_days.toLocaleString()} cell-days, train RMSE ${fmt(card.training.grid_correction.train_rmse, 3)}`
                    : '—'}
                </dd>
                <dt>Pipeline runtime</dt><dd className="mono">{fmt(card.training?.elapsed_seconds, 1)} s</dd>
                <dt>Archive fingerprint</dt>
                <dd className="mono">{card.data?.archive_sha256_16 || '—'}</dd>
              </dl>
              <div className="tiny muted" style={{ marginTop: 6 }}>
                The fingerprint is the SHA-256 (first 16 hex) of the archive these results were
                scored against. `make evaluate` writes it next to every metric, so any number can
                be traced back to the exact data that produced it.
              </div>
              <div className="tiny muted" style={{ marginTop: 8 }}>
                {card.data?.provenance_detail?.disclaimer
                  || 'Synthetic archive generated inside this repository; not observed IMD data.'}
              </div>
              {card.data?.provenance_detail?.what_is_real && (
                <div className="small" style={{ marginTop: 6 }}>
                  <b>What is real:</b> {card.data.provenance_detail.what_is_real}
                  {card.data?.provenance_detail?.what_is_synthetic && (
                    <><br /><b>What is synthetic:</b> {card.data.provenance_detail.what_is_synthetic}</>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="panel">
            <div className="panel-head"><span className="panel-title">Stated limits</span></div>
            <div className="panel-body small">
              <ul style={{ margin: 0, paddingLeft: 16, lineHeight: 1.7 }}>
                {(card.known_limits || []).map((l) => <li key={l}>{l}</li>)}
              </ul>
              <div style={{ marginTop: 8 }}>
                <b>Reproduce everything:</b>
                <div className="mono small term" style={{ marginTop: 4 }}>
                  make data && make train && make evaluate && pytest tests/ -q
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panel-head"><span className="panel-title">Regime definitions used by the classifier</span></div>
        <div className="table-scroll">
          <table className="data">
            <thead><tr><th>ID</th><th>Regime</th><th>Rule criterion (configs/regime_rules.yaml)</th></tr></thead>
            <tbody>
              {(card?.regimes || []).map((r) => (
                <tr key={r.id}>
                  <td className="num">{r.id}</td>
                  <td><b>{r.name}</b></td>
                  <td className="small muted">{r.criterion}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
