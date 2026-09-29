import React, { useEffect, useMemo, useState } from 'react';
import {
  FiChevronLeft, FiChevronRight, FiLayers, FiX, FiMessageSquare, FiAlertTriangle, FiCpu,
  FiMaximize2,
} from 'react-icons/fi';
import { api } from '../api';
import IndiaMap from './IndiaMap';
import AskBar from './AskBar';
import DistrictRail from './DistrictRail';
import {
  CATEGORY_META, CATEGORY_FILL, MAP_LAYERS, REGIME_COLOR, RAIN_LEGEND, adjustmentColor,
  probabilityColor, rainColor, fmt, pct, longDate,
} from '../lib/format';

function Legend({ layer }) {
  if (layer === 'category') {
    return ['red', 'orange', 'yellow', 'green'].map((c) => (
      <div key={c}><i style={{ background: CATEGORY_FILL[c] }} />{CATEGORY_META[c].label} · {CATEGORY_META[c].mm} mm</div>
    ));
  }
  if (layer === 'corrected' || layer === 'raw') {
    return RAIN_LEGEND.map(([label, v]) => <div key={label}><i style={{ background: rainColor(v) }} />{label} mm</div>);
  }
  if (layer === 'adjustment') {
    return [[30, 'made wetter'], [0, 'unchanged'], [-30, 'made drier']].map(([v, l]) => (
      <div key={l}><i style={{ background: adjustmentColor(v) }} />{l}</div>
    ));
  }
  if (layer === 'p_heavy') {
    return [[0.8, '≥ 75%'], [0.6, '50–75%'], [0.4, '30–50%'], [0.2, '15–30%'], [0.05, '< 15%']].map(([v, l]) => (
      <div key={l}><i style={{ background: probabilityColor(v) }} />{l}</div>
    ));
  }
  return Object.entries(REGIME_COLOR).map(([n, c]) => <div key={n}><i style={{ background: c }} />{n}</div>);
}

/**
 * Full-screen forecast map (the SAGAR-style "globe view"): offline India map in the middle,
 * control panel left, analysis panel right, and the Ask MonsoonIQ bar at the bottom.
 */
export default function ForecastMap({ session, onSession, initialLayer = 'category', onTab }) {
  const { date, lead } = session;
  const [layer, setLayer] = useState(initialLayer);
  const [c, setC] = useState(null);
  const [geo, setGeo] = useState(null);
  const [base, setBase] = useState(null);
  const [meta, setMeta] = useState(null);
  const [selected, setSelected] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [leftOpen, setLeftOpen] = useState(() => typeof window === 'undefined' || window.innerWidth > 720);

  useEffect(() => { setLayer(initialLayer); }, [initialLayer]);
  useEffect(() => {
    api.geojson().then(setGeo).catch(() => {});
    api.geojsonAll().then(setBase).catch(() => {});
    api.timeline().then((t) => setMeta(t.meta || null)).catch(() => {});
  }, []);
  useEffect(() => {
    if (!date && meta?.default_date) onSession({ date: meta.default_date });
  }, [date, meta, onSession]);
  useEffect(() => {
    if (!date) return;
    setBusy(true);
    api.console(date, lead, false).then(setC).catch(() => {}).finally(() => setBusy(false));
  }, [date, lead]);

  const districts = c?.districts || [];
  const posterior = useMemo(() => Object.entries(c?.regime?.posterior || {}).sort((a, b) => b[1] - a[1]), [c]);

  const ask = (q) => {
    setAsking(true);
    api.ask(q, c?.date || date, lead)
      .then((r) => {
        setAnswer({ q, ...r });
        setSelected(null);
        if (r.lead && r.lead !== lead) onSession({ lead: r.lead });
        if (r.intent === 'regime') setLayer('regime');
        else if (r.intent === 'adjustment') setLayer('adjustment');
        else if (r.parsed?.prob) setLayer('p_heavy');
        else if (r.parsed?.category) setLayer('category');
        else if (r.intent === 'detail') setLayer('corrected');
      })
      .catch((e) => setAnswer({ q, answer: `Could not answer: ${e.message}`, rows: [], districts: [] }))
      .finally(() => setAsking(false));
  };

  const step = (dir) => {
    const next = dir > 0 ? c?.navigation?.next_date : c?.navigation?.prev_date;
    if (next) onSession({ date: next });
  };

  return (
    <div className="fmap">
      {busy && <div className="progressbar" />}
      <IndiaMap baseGeo={base} studyGeo={geo} districts={districts} layer={layer}
                selectedId={selected} onSelect={(id) => { setSelected(id); setAnswer(null); }}
                highlight={answer?.districts?.length ? answer.districts : null}
                className="fmap-canvas" />

      <aside className={`float-panel left ${leftOpen ? '' : 'collapsed'}`}>
        <div className="fp-head">
          <span><FiLayers /> Forecast controls</span>
          <button className="icon-btn" onClick={() => setLeftOpen(!leftOpen)} title="collapse">
            {leftOpen ? <FiChevronLeft /> : <FiChevronRight />}
          </button>
        </div>
        {leftOpen && (
          <div className="fp-body">
            <label className="fp-label">Valid date</label>
            <div className="date-stepper full">
              <button onClick={() => step(-1)} disabled={!c?.navigation?.prev_date}><FiChevronLeft /></button>
              <input type="date" value={c?.date || date || ''} onChange={(e) => onSession({ date: e.target.value })} />
              <button onClick={() => step(1)} disabled={!c?.navigation?.next_date}><FiChevronRight /></button>
            </div>
            <div className="tiny muted" style={{ marginTop: 4 }}>{longDate(c?.date)}</div>

            <label className="fp-label">Lead time</label>
            <div className="seg lead-seg full">
              {[1, 2, 3, 4, 5].map((l) => (
                <button key={l} aria-pressed={lead === l} onClick={() => onSession({ lead: l })}>Day {l}</button>
              ))}
            </div>

            <label className="fp-label">Map layer</label>
            <div className="layer-list">
              {MAP_LAYERS.map((l) => (
                <button key={l.id} className={layer === l.id ? 'on' : ''} onClick={() => setLayer(l.id)}>
                  <span className="radio" />{l.label}
                </button>
              ))}
            </div>

            <label className="fp-label">Legend</label>
            <div className="legend-col"><Legend layer={layer} /></div>

            <label className="fp-label"><FiCpu /> Regime posterior</label>
            <div className="post-bars compact">
              {posterior.slice(0, 5).map(([name, p]) => (
                <div key={name} className="post-row">
                  <span className="name">{name}</span>
                  <div className="track"><div style={{ width: `${Math.max(1, p * 100)}%`, background: REGIME_COLOR[name] }} /></div>
                  <span className="val mono">{pct(p)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </aside>

      <aside className={`float-panel right${answer || selected ? " has-content" : ""}`}>
        {answer ? (
          <>
            <div className="fp-head">
              <span><FiMessageSquare /> Ask MonsoonIQ</span>
              <button className="icon-btn" onClick={() => setAnswer(null)} title="clear"><FiX /></button>
            </div>
            <div className="fp-body">
              <div className="ask-q">“{answer.q}”</div>
              <div className="ask-a">{answer.answer}</div>
              <div className="tiny muted" style={{ margin: '6px 0 10px' }}>
                Answered from the live forecast ({answer.date}, Day {answer.lead}) — {answer.districts?.length || 0} district(s) highlighted.
              </div>
              <div className="mini-list">
                {(answer.rows || []).map((r) => (
                  <button key={r.district_id} onClick={() => { setSelected(r.district_id); setAnswer(null); }}>
                    <span className={`cat-dot ${r.category}`} />
                    <span className="nm">{r.district_name}<span className="tiny muted"> · {r.state_name}</span></span>
                    <span className="mono">{fmt(r.corrected_mm)} mm</span>
                  </button>
                ))}
              </div>
            </div>
          </>
        ) : selected ? (
          <div className="rail-wrap">
            <DistrictRail districtId={selected} date={c?.date || date} lead={lead} onClose={() => setSelected(null)} />
          </div>
        ) : (
          <>
            <div className="fp-head"><span><FiAlertTriangle /> Highest exposure</span>
              <button className="icon-btn" onClick={() => onTab('today')} title="Operations console"><FiMaximize2 /></button>
            </div>
            <div className="fp-body">
              <div className="exposure-sum">
                <div><b className="mono">{c?.summary?.districts_in_warning ?? '—'}</b><span>warned</span></div>
                <div><b className="mono">{fmt(c?.summary?.max_corrected_mm)}</b><span>mm peak</span></div>
                <div><b style={{ color: REGIME_COLOR[c?.regime?.name] }}>{c?.regime?.name?.split(' ')[0] || '—'}</b><span>regime</span></div>
              </div>
              <div className="mini-list">
                {districts.slice(0, 10).map((r) => (
                  <button key={r.district_id} onClick={() => setSelected(r.district_id)}>
                    <span className={`cat-dot ${r.category}`} />
                    <span className="nm">{r.district_name}<span className="tiny muted"> · {r.regime}</span></span>
                    <span className="mono">{fmt(r.corrected_mm)} mm</span>
                  </button>
                ))}
              </div>
              <div className="tiny muted" style={{ marginTop: 8 }}>Click a district for its forecast band, regime posterior and advisory.</div>
            </div>
          </>
        )}
      </aside>

      <AskBar onAsk={ask} busy={asking} />
    </div>
  );
}
