import React, { useEffect, useMemo, useState } from 'react';
import { api } from '../../api';
import { fmt, pct, signed } from '../../lib/format';

const ORDER = ['raw_nwp', 'global_qm', 'global_lgb', 'monsooniq'];
const SHORT = {
  raw_nwp: 'Raw NWP',
  global_qm: 'Quantile mapping',
  global_lgb: 'GBM (agnostic)',
  monsooniq: 'MonsoonIQ',
  moe_uniform_weights: 'MoE, uniform weights',
  moe_classifier: 'MoE, classifier weights',
  moe_oracle_regime: 'MoE, oracle regime',
};
const SYSTEM_FALLBACK = (key) => SHORT[key] || key.replace(/_/g, ' ');

const THRESHOLDS = ['2.5', '15.6', '64.5', '115.6', '204.5'];

function Verdict({ supported, positive = 'supported', negative = 'not supported' }) {
  return <span className={supported ? 'verdict-ok' : 'verdict-no'}>{supported ? positive : negative}</span>;
}

/** Reliability diagram: forecast probability vs observed frequency. */
function ReliabilityDiagram({ diagram }) {
  if (!diagram?.bin_edges) return null;
  const size = 190, pad = 26;
  const x = (p) => pad + p * (size - pad * 2);
  const y = (f) => size - pad - f * (size - pad * 2);
  const pts = diagram.mean_forecast_probs.map((p, i) => [p, diagram.observed_frequencies[i], diagram.sample_counts[i]]);
  return (
    <svg width={size} height={size} role="img" aria-label="reliability diagram">
      <rect x={pad} y={pad} width={size - pad * 2} height={size - pad * 2} fill="#08151f" stroke="#1b3549" />
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="#2b5a7d" strokeDasharray="3 3" />
      {pts.map(([p, f, n], i) => (
        <circle key={i} cx={x(p)} cy={y(f)} r={2 + Math.min(Math.sqrt(n) / 8, 4)}
                fill="#00d4ff" fillOpacity="0.85" />
      ))}
      <text x={size / 2} y={size - 6} fontSize="9" textAnchor="middle" fill="#7f97ab">forecast probability →</text>
      <text x={9} y={size / 2} fontSize="9" textAnchor="middle" fill="#7f97ab"
            transform={`rotate(-90 9 ${size / 2})`}>observed frequency →</text>
    </svg>
  );
}

/** Accuracy sweep: how much does regime-classifier accuracy have to buy? */
function AccuracySweep({ sweep, agnosticCsi }) {
  if (!sweep?.length) return null;
  const w = 320, h = 150, pad = 30;
  const xs = sweep.map((s) => s.regime_classifier_accuracy).reverse();
  const min = Math.min(...xs), max = Math.max(...xs);
  const px = (a) => pad + ((a - min) / Math.max(max - min, 1e-6)) * (w - pad * 2);
  const csis = sweep.map((s) => s.moe_heavy_csi);
  const lo = Math.min(...csis, agnosticCsi) - 0.01;
  const hi = Math.max(...csis, agnosticCsi) + 0.01;
  const py = (c) => h - pad - ((c - lo) / (hi - lo)) * (h - pad * 2);
  const line = sweep
    .slice().sort((a, b) => a.regime_classifier_accuracy - b.regime_classifier_accuracy)
    .map((s) => `${px(s.regime_classifier_accuracy)},${py(s.moe_heavy_csi)}`).join(' ');
  return (
    <svg width={w} height={h} role="img" aria-label="regime classifier accuracy sweep">
      <rect x={pad} y={pad} width={w - pad * 2} height={h - pad * 2} fill="#08151f" stroke="#1b3549" />
      <line x1={pad} y1={py(agnosticCsi)} x2={w - pad} y2={py(agnosticCsi)}
            stroke="#ff3b30" strokeDasharray="4 3" />
      <text x={w - pad} y={py(agnosticCsi) - 3} fontSize="9" textAnchor="end" fill="#ff7369">
        regime-agnostic baseline {fmt(agnosticCsi, 3)}
      </text>
      <polyline points={line} fill="none" stroke="#00d4ff" strokeWidth="2" />
      {sweep.map((s) => (
        <circle key={s.regime_classifier_accuracy} cx={px(s.regime_classifier_accuracy)}
                cy={py(s.moe_heavy_csi)} r="3" fill="#00d4ff" />
      ))}
      <text x={w / 2} y={h - 6} fontSize="9" textAnchor="middle" fill="#7f97ab">classifier accuracy →</text>
    </svg>
  );
}

export default function SkillLab() {
  const [summary, setSummary] = useState(null);
  const [events, setEvents] = useState(null);
  const [threshold, setThreshold] = useState('64.5');
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.verification(), api.heavyEvents().catch(() => null)])
      .then(([s, e]) => { setSummary(s); setEvents(e); })
      .catch((err) => setError(String(err.message || err)));
  }, []);

  const tm = useMemo(() => summary?.threshold_metrics?.[threshold], [summary, threshold]);
  const period = summary?.verification_period || {};
  const rv = summary?.regime_value || {};
  const grid = summary?.grid_fss;

  if (error) {
    return (
      <div className="workspace">
        <div className="note">Verification artifacts unavailable ({error}). Run
          {' '}<span className="mono">python src/evaluate.py</span> first.</div>
      </div>
    );
  }
  if (!summary) return <div className="workspace"><div className="skeleton" style={{ height: 340 }} /></div>;

  const cards = summary.scorecard?.cards || [];
  const claims = summary.significance_statement?.claims || [];

  return (
    <div className="workspace">
      <div className="note">
        <b>Read this first.</b> Every number below is an internal comparison computed on the archive in
        this repository (provenance <span className="mono">{summary.data_provenance}</span>), not IMD
        verification. Held-out years <span className="mono">{(period.test_years || []).join(', ')}</span>
        {' '}· <span className="mono">{period.calendar_days}</span> calendar days ·
        {' '}<span className="mono">{period.district_days?.toLocaleString()}</span> district-days ·
        {' '}<span className="mono">{period.districts}</span> districts. What makes the comparison worth
        reading is that the reference systems are fitted and scored on exactly the same fixtures:
        one command reproduces every figure.
      </div>

      {/* ---------------------------------------------- scorecard */}
      <div className="vgrid">
        {cards.map((c) => {
          const better = c.lower_is_better ? -1 : 1;
          return (
            <div className="card" key={c.label}>
              <h3>{c.label}</h3>
              <div className="big">{fmt(c.corrected, c.unit ? 2 : 3)}<span className="u" style={{ fontSize: 12, color: 'var(--muted)' }}> {c.unit}</span></div>
              <div className="small muted">
                raw {fmt(c.raw, 3)} → agnostic {fmt(c.baseline, 3)}
              </div>
              <div className="small" style={{ marginTop: 4 }}>
                vs raw <span className={(c.vs_raw_pct * better) > 0 ? 'delta-up' : 'delta-down'}>{signed(c.vs_raw_pct, 1)}%</span>
                {'  '}vs agnostic <span className={(c.vs_baseline_pct * better) > 0 ? 'delta-up' : 'delta-down'}>{signed(c.vs_baseline_pct, 1)}%</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* ---------------------------------------------- claims */}
      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">Claims and their verdicts</span>
          <span className="tiny muted">
            paired day-block bootstrap · {summary.confidence_intervals?.rmse?.method}
          </span>
        </div>
        <div className="panel-body tight">
          <table className="data">
            <thead><tr><th>Claim</th><th>Evidence</th><th>Verdict</th></tr></thead>
            <tbody>
              {claims.map((c) => (
                <tr key={c.claim}>
                  <td style={{ maxWidth: 380 }}>{c.claim}</td>
                  <td className="mono small">{c.evidence}</td>
                  <td><Verdict supported={c.supported} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="panel-body small">
            <b>{summary.significance_statement?.headline}</b>
            <div className="muted" style={{ marginTop: 3 }}>{summary.significance_statement?.caveat}</div>
          </div>
        </div>
      </div>

      {/* ---------------------------------------------- categorical */}
      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">Categorical verification by rainfall threshold</span>
          <div className="seg">
            {THRESHOLDS.map((t) => (
              <button key={t} aria-pressed={threshold === t} onClick={() => setThreshold(t)}>≥ {t} mm</button>
            ))}
          </div>
        </div>
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>System</th><th className="num">Events</th><th className="num">Hits</th>
                <th className="num">Misses</th><th className="num">False alarms</th>
                <th className="num">POD</th><th className="num">FAR</th><th className="num">CSI</th>
                <th className="num">ETS</th><th className="num">Freq. bias</th><th>Reliability</th>
              </tr>
            </thead>
            <tbody>
              {ORDER.filter((s) => tm?.[s]).map((s) => {
                const r = tm[s];
                return (
                  <tr key={s}>
                    <td><b>{SYSTEM_FALLBACK(s)}</b></td>
                    <td className="num">{r.event_count}</td>
                    <td className="num">{r.hits}</td>
                    <td className="num">{r.misses}</td>
                    <td className="num">{r.false_alarms}</td>
                    <td className="num">{fmt(r.pod, 3)}</td>
                    <td className="num">{fmt(r.far, 3)}</td>
                    <td className="num"><b>{fmt(r.csi, 3)}</b></td>
                    <td className="num">{fmt(r.ets, 3)}</td>
                    <td className="num">{fmt(r.frequency_bias, 3)}</td>
                    <td className={r.reliability === 'reliable' ? 'small' : 'suppressed small'}>
                      {r.reliability}{r.reportable === false ? ' (not reportable)' : ''}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="panel-body small muted">
          {tm?.raw_nwp?.reliability === 'insufficient'
            ? 'This threshold has too few events to support a comparison; treat every cell here as descriptive only. The evaluation pipeline labels it rather than hiding it.'
            : `Stratum reliability is labelled per threshold: strata with too few events are marked insufficient rather than presented as skill.`}
        </div>
      </div>

      {/* ---------------------------------------------- lead + regime */}
      <div className="split" style={{ gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)' }}>
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Skill by lead time (≥ 64.5 mm)</span>
            <span className="tiny muted">{summary.lead_time_breakdown?.note}</span>
          </div>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr><th>Lead</th><th className="num">MonsoonIQ CSI</th><th className="num">Agnostic CSI</th>
                  <th className="num">Raw CSI</th><th className="num">MonsoonIQ RMSE</th>
                  <th className="num">Agnostic RMSE</th><th className="num">Raw RMSE</th></tr>
              </thead>
              <tbody>
                {Object.entries(summary.lead_time_breakdown?.leads || {}).map(([day, entry]) => {
                  const s = entry.systems || {};
                  const csi = (sys) => fmt(s[sys]?.['csi_64.5'], 3);
                  return (
                    <tr key={day}>
                      <td><b>{day.replace('_', ' ')}</b></td>
                      <td className="num">{csi('monsooniq')}</td>
                      <td className="num">{csi('global_lgb')}</td>
                      <td className="num">{csi('raw_nwp')}</td>
                      <td className="num">{fmt(s.monsooniq?.rmse, 3)}</td>
                      <td className="num">{fmt(s.global_lgb?.rmse, 3)}</td>
                      <td className="num">{fmt(s.raw_nwp?.rmse, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Does the gain hold inside each regime?</span>
            <span className="tiny muted">thin strata are suppressed, not shown as zeros</span>
          </div>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr><th>Regime</th><th className="num">Events</th><th className="num">ΔCSI vs raw</th>
                  <th className="num">95% CI</th><th>Reliability</th></tr>
              </thead>
              <tbody>
                {Object.entries(summary.regime_stratified || {}).map(([name, v]) => (
                  <tr key={name}>
                    <td><b>{name}</b></td>
                    <td className="num">{v.reportable === false ? '—' : v.event_counts?.heavy ?? v.events}</td>
                    <td className="num">{v.improvement ? signed(v.improvement.delta, 4) : '—'}</td>
                    <td className="num">{v.improvement?.ci95
                      ? `[${fmt(v.improvement.ci95[0], 3)}, ${fmt(v.improvement.ci95[1], 3)}]` : '—'}</td>
                    <td className={v.reliability === 'reliable' ? 'small' : 'suppressed small'}>
                      {v.reliability || 'insufficient'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ---------------------------------------------- probabilistic */}
      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">Probability forecasts: are the numbers calibrated?</span>
          <span className="tiny muted">reliability diagram, exceedance modules</span>
        </div>
        <div className="panel-body">
          <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
            {Object.entries(summary.probabilistic_verification || {}).map(([key, v]) => (
              <div key={key} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <ReliabilityDiagram diagram={v.reliability_diagram} />
                <div className="small" style={{ minWidth: 170 }}>
                  <div><b>P(≥ {v.threshold_mm} mm)</b></div>
                  <dl className="kv" style={{ gridTemplateColumns: '1fr auto', marginTop: 4 }}>
                    <dt>Events</dt><dd>{v.event_count}</dd>
                    <dt>Brier score</dt><dd>{fmt(v.brier_score, 5)}</dd>
                    <dt>Mean forecast</dt><dd>{fmt(v.mean_forecast_probability, 4)}</dd>
                    <dt>Observed freq.</dt><dd>{fmt(v.observed_frequency, 4)}</dd>
                    <dt>ROC AUC</dt><dd>{v.roc?.auc !== undefined ? fmt(v.roc.auc, 4) : 'suppressed'}</dd>
                    <dt>Reliability</dt><dd>{v.reliability}</dd>
                  </dl>
                  {v.roC_suppressed && <div className="tiny muted" style={{ marginTop: 4 }}>{v.roC_suppressed}</div>}
                </div>
              </div>
            ))}
          </div>
          <div className="small muted" style={{ marginTop: 8 }}>
            The diagram is the honest test of a probability product: points on the diagonal mean a
            stated 30% chance happens about 30% of the time. Bubble size is the number of
            district-days in the bin, so the sparse bins on the right are visibly weightless.
          </div>
        </div>
      </div>

      {/* ---------------------------------------------- grid FSS */}
      {grid?.available && (
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Fractional skill score on the grid</span>
            <span className="tiny muted">
              {grid.grid?.dates_scored} held-out dates · {grid.grid?.grid_cells_in_districts} cells in
              district boxes · windows {grid.neighbourhood_windows_cells?.join(', ')} grid lengths
            </span>
          </div>
          <div className="panel-body">
            <div className="small" style={{ marginBottom: 8 }}>
              A district score can hide structure: a smooth field can win on district averages while
              putting the rain in the wrong cell. FSS forces a spatial comparison. The method note
              recorded with this artifact: <span className="muted">{grid.method}</span>
            </div>
            {(grid.thresholds_mm || []).map((thr) => (
              <div key={thr} style={{ marginBottom: 10 }}>
                <div className="small"><b>Threshold {thr} mm</b></div>
                <table className="data">
                  <thead>
                    <tr>
                      <th>System</th>
                      {grid.neighbourhood_windows_cells.map((w) => (
                        <th key={w} className="num">FSS w={w}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {ORDER.concat(['regime_aware_grid_native', 'global_qm_district_transfer',
                      'regime_aware_district_transfer'])
                      .filter((s) => grid.scores?.[String(thr)]?.w1?.[s])
                      .map((s) => (
                        <tr key={s}>
                          <td>{SYSTEM_FALLBACK(s)}</td>
                          {grid.neighbourhood_windows_cells.map((w) => (
                            <td key={w} className="num">
                              {fmt(grid.scores[String(thr)][`w${w}`][s]?.fss, 3)}
                            </td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ))}
            <div className="small muted">
              District-to-grid transfer rows are the negative control: taking a correction fitted on
              district averages and applying it cell by cell scores <i>below</i> the raw model, because
              a single multiplier cannot restore spatial structure. That is why the operational grid
              product is a model trained natively on grid cells.
            </div>
          </div>
        </div>
      )}

      {/* ---------------------------------------------- regime value */}
      {rv.available && (
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Is the regime layer earning its complexity?</span>
            <span className="tiny muted">regime value audit</span>
          </div>
          <div className="panel-body">
            <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
              <AccuracySweep sweep={rv.accuracy_sweep} agnosticCsi={rv.systems?.global_lgb?.csi} />
              <div style={{ minWidth: 260, flex: 1 }}>
                <table className="data">
                  <thead><tr><th>Blend</th><th className="num">RMSE</th><th className="num">CSI</th><th className="num">ETS</th></tr></thead>
                  <tbody>
                    {Object.entries(rv.systems || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td>{SYSTEM_FALLBACK(k)}</td>
                        <td className="num">{fmt(v.rmse, 3)}</td>
                        <td className="num">{fmt(v.csi, 3)}</td>
                        <td className="num">{fmt(v.ets, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="small muted" style={{ marginTop: 6 }}>{rv.interpretation}</div>
                <div className="small" style={{ marginTop: 6 }}>
                  Classifier hold-out accuracy <span className="mono">{pct(rv.classifier?.holdout_accuracy, 1)}</span>
                  {' '}— and that number is a property of the generator, not evidence of a deployable
                  classifier: {rv.classifier?.note}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ---------------------------------------------- events by lead/regime */}
      {events && (
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Heavy-rainfall event inventory (≥ {events.threshold_mm} mm)</span>
            <span className="tiny muted">which days and regimes carry these statistics</span>
          </div>
          <div className="panel-body tight">
            <table className="data">
              <thead>
                <tr><th>Slice</th><th className="num">Events</th><th className="num">Raw POD</th>
                  <th className="num">MonsoonIQ POD</th><th className="num">Raw CSI</th>
                  <th className="num">MonsoonIQ CSI</th><th>Reliability</th></tr>
              </thead>
              <tbody>
                {events.overall_by_system?.monsooniq && (
                  <tr>
                    <td><b>All held-out days</b></td>
                    <td className="num">{events.overall_by_system.monsooniq.event_count ?? '—'}</td>
                    <td className="num">{fmt(events.overall_by_system.raw_nwp?.pod, 3)}</td>
                    <td className="num">{fmt(events.overall_by_system.monsooniq?.pod, 3)}</td>
                    <td className="num">{fmt(events.overall_by_system.raw_nwp?.csi, 3)}</td>
                    <td className="num">{fmt(events.overall_by_system.monsooniq?.csi, 3)}</td>
                    <td className="small">{events.overall_by_system.monsooniq?.reliability || '—'}</td>
                  </tr>
                )}
                {Object.entries(events.by_regime || {}).map(([name, v]) => (
                  <tr key={name}>
                    <td>Regime: {name}</td>
                    <td className="num">{v.event_count}</td>
                    <td className="num">{fmt(v.systems?.raw_nwp?.pod, 3)}</td>
                    <td className="num">{fmt(v.systems?.monsooniq?.pod, 3)}</td>
                    <td className="num">{fmt(v.systems?.raw_nwp?.csi, 3)}</td>
                    <td className="num">{fmt(v.systems?.monsooniq?.csi, 3)}</td>
                    <td className={v.reliability === 'reliable' ? 'small' : 'suppressed small'}>{v.reliability}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ---------------------------------------------- zones + reproduce */}
      <div className="split" style={{ gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)' }}>
        <div className="panel">
          <div className="panel-head"><span className="panel-title">Skill by agro-climatic zone</span></div>
          <div className="table-scroll">
            <table className="data">
              <thead><tr><th>Zone</th><th className="num">Days</th><th className="num">Heavy events</th>
                <th className="num">RMSE raw</th><th className="num">RMSE MonsoonIQ</th><th>Reliability</th></tr></thead>
              <tbody>
                {Object.entries(summary.zone_breakdown || {}).map(([zone, v]) => (
                  <tr key={zone}>
                    <td><b>{zone}</b></td>
                    <td className="num">{v.sample_count}</td>
                    <td className="num">{v.heavy_events}</td>
                    <td className="num">{fmt(v.rmse?.raw_nwp, 3)}</td>
                    <td className="num">{fmt(v.rmse?.monsooniq, 3)}</td>
                    <td className={v.reliability === 'reliable' ? 'small' : 'suppressed small'}>{v.reliability}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Reproduce every number on this page</span>
            <a className="btn" href={api.reportPdfUrl()} target="_blank" rel="noreferrer">Open PDF report</a>
          </div>
          <div className="panel-body">
            <div className="mono small term">
              make data&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# deterministic synthetic archive<br />
              make train&nbsp;&nbsp;&nbsp;&nbsp;# classifier, per-lead correction, exceedance, grid model<br />
              make evaluate&nbsp;# writes the artifacts this page reads<br />
              python -m pytest tests/ -q
            </div>
            <div className="small" style={{ marginTop: 8 }}>
              <b>Honest baseline used everywhere:</b>{' '}
              {summary.scorecard?.honest_baseline?.[1] || 'quantile mapping and a regime-agnostic learner'}.
              Beating the raw model is table stakes; the comparison that matters is against that.
            </div>
            <div className="small" style={{ marginTop: 6 }}>
              Continuous metrics (held-out): {Object.entries(summary.continuous_metrics || {}).map(([k, v]) => (
                <span key={k} className="nowrap" style={{ marginRight: 10 }}>
                  {SYSTEM_FALLBACK(k)} RMSE <span className="mono">{fmt(v.rmse, 3)}</span>,
                  bias <span className="mono">{signed(v.bias, 3)}</span>
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
