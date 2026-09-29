import React, { useEffect, useMemo, useState } from 'react';
import {
  FiArrowRight, FiCloudRain, FiSliders, FiAlertTriangle, FiMap, FiBarChart2, FiDatabase,
  FiChevronLeft, FiChevronRight, FiFileText, FiDownload, FiCalendar, FiZap,
} from 'react-icons/fi';
import { api } from '../api';
import IndiaMap from './IndiaMap';
import BulletinModal from './BulletinModal';
import { CATEGORY_COLOR, CATEGORY_META, REGIME_COLOR, fmt, pct, signed, longDate } from '../lib/format';

/**
 * Mission control: one card per expected outcome of the problem statement, each showing a
 * live number from the engine and opening the module that produces it.
 */
export default function Dashboard({ session, onSession, onTab, onOpenMap }) {
  const { date, lead } = session;
  const [c, setC] = useState(null);
  const [geo, setGeo] = useState(null);
  const [base, setBase] = useState(null);
  const [ver, setVer] = useState(null);
  const [events, setEvents] = useState([]);
  const [ds, setDs] = useState(null);
  const [meta, setMeta] = useState(null);
  const [bulletin, setBulletin] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.geojson().then(setGeo).catch(() => {});
    api.geojsonAll().then(setBase).catch(() => {});
    api.verification().then(setVer).catch(() => {});
    api.events(12).then((e) => setEvents(e.events || [])).catch(() => {});
    api.dataSources().then(setDs).catch(() => {});
    api.timeline().then((t) => setMeta(t.meta || null)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!date && meta?.default_date) onSession({ date: meta.default_date });
  }, [date, meta, onSession]);

  useEffect(() => {
    if (!date) return;
    setErr(null);
    api.console(date, lead).then(setC).catch((e) => setErr(String(e.message || e)));
  }, [date, lead]);

  const districts = c?.districts || [];
  const counts = c?.summary?.counts;
  const posterior = useMemo(() => Object.entries(c?.regime?.posterior || {}).sort((a, b) => b[1] - a[1]), [c]);
  const probHigh = districts.filter((d) => d.p_heavy >= 0.5).length;
  const cards = ver?.scorecard?.cards || [];
  const csi = cards.find((x) => x.label.startsWith('Heavy-rain warning CSI'))
    || cards.find((x) => x.label.startsWith('Heavy-rain CSI'));
  const rmse = cards.find((x) => x.label === 'RMSE');
  const topAdj = [...districts].sort((a, b) => Math.abs(b.adjustment_mm) - Math.abs(a.adjustment_mm))[0];

  const modules = [
    {
      n: 1, icon: FiCloudRain, color: 'cyan', title: 'Weather Regime Classifier',
      tags: ['Outcome 1', 'LightGBM · calibrated'],
      metric: c?.regime?.name || '—', sub: `posterior ${pct(c?.regime?.confidence)} · ${Math.round((c?.regime?.district_agreement ?? 0) * 100)}% district agreement`,
      desc: 'Classifies active, break, depression, orographic, coastal and western-disturbance regimes with soft probabilities per district.',
      bar: c?.regime?.confidence ?? 0, go: () => onOpenMap('regime'),
    },
    {
      n: 2, icon: FiSliders, color: 'green', title: 'Bias-Corrected Forecast',
      tags: ['Outcome 2', 'Mixture of experts'],
      metric: `${signed(c?.summary?.mean_bias_adjustment_mm)} mm`, sub: topAdj ? `largest: ${topAdj.district_name} ${fmt(topAdj.raw_mm)}→${fmt(topAdj.corrected_mm)} mm` : 'mean correction applied',
      desc: 'Regime-conditional quantile mapping plus a residual expert per regime, blended by the classifier posterior.',
      bar: rmse ? Math.min(1, Math.abs(rmse.vs_raw_pct) / 100) : 0, barLabel: rmse ? `RMSE ${rmse.vs_raw_pct}% vs raw` : null,
      go: () => onOpenMap('adjustment'),
    },
    {
      n: 3, icon: FiAlertTriangle, color: 'orange', title: 'Heavy Rainfall Probability',
      tags: ['Outcome 3', 'IMD thresholds'],
      metric: `${probHigh} districts`, sub: 'P(≥ 64.5 mm) ≥ 50% · also ≥115.6 / ≥204.5 mm',
      desc: 'Calibrated exceedance probabilities for heavy, very heavy and extremely heavy rain, plus a P10–P90 band.',
      bar: districts.length ? probHigh / districts.length : 0, go: () => onOpenMap('p_heavy'),
    },
    {
      n: 4, icon: FiMap, color: 'yellow', title: 'District Rainfall Product',
      tags: ['Outcome 4', 'Map · table · CSV · CAP'],
      metric: `${c?.summary?.districts_in_warning ?? '—'} warned`, sub: counts ? `${counts.red} red · ${counts.orange} orange · ${counts.yellow} yellow` : '',
      desc: 'District warning table and map on the IMD colour scale, bilingual bulletin, CSV export and CAP alerts.',
      bar: districts.length ? (c?.summary?.districts_in_warning ?? 0) / districts.length : 0, go: () => onTab('today'),
    },
    {
      n: 5, icon: FiBarChart2, color: 'teal', title: 'Verification Report',
      tags: ['Outcome 5', 'RMSE · ETS · CSI · POD · FAR · FSS'],
      metric: csi ? `CSI ${fmt(csi.raw, 2)} → ${fmt(csi.corrected, 2)}` : '—', sub: 'heavy rain ≥ 64.5 mm, held-out years, day-block bootstrap CIs',
      desc: 'Skill against raw NWP and a regime-agnostic learner, by regime, zone and lead, with a downloadable PDF.',
      bar: csi ? Math.min(1, csi.corrected) : 0, go: () => onTab('skill'),
    },
    {
      n: 6, icon: FiDatabase, color: 'purple', title: 'Data & Real-Data Pipeline',
      tags: ['IMD 0.25°', 'NCUM / GFS / ECMWF', 'ERA5'],
      metric: ds ? (ds.active_profile === 'real' ? 'IMD observed' : 'Synthetic benchmark') : '—',
      sub: ds ? `${ds.sources.length} catalogued sources · split ${ds.split?.train?.[0]}–${ds.split?.test?.slice(-1)[0]}` : '',
      desc: 'IMD gridded rainfall, archived NWP runs, ERA5 dynamics, NOAA OLR, Census-2011 districts — one command to build.',
      bar: ds?.active_profile === 'real' ? 1 : 0.5, go: () => onTab('data'),
    },
  ];

  const step = (dir) => {
    const next = dir > 0 ? c?.navigation?.next_date : c?.navigation?.prev_date;
    if (next) onSession({ date: next });
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Mission Control</h1>
          <div className="page-sub">
            <FiCalendar /> {longDate(c?.date || date)} · Day {lead} forecast
            {c?.issued_utc && <span className="muted"> · models loaded {c.issued_utc}</span>}
          </div>
        </div>
        <div className="head-actions">
          <div className="date-stepper">
            <button onClick={() => step(-1)} title="previous day" disabled={!c?.navigation?.prev_date}><FiChevronLeft /></button>
            <input type="date" value={date || ''} onChange={(e) => onSession({ date: e.target.value })} />
            <button onClick={() => step(1)} title="next day" disabled={!c?.navigation?.next_date}><FiChevronRight /></button>
          </div>
          <div className="seg lead-seg" role="group" aria-label="lead time">
            {[1, 2, 3, 4, 5].map((l) => (
              <button key={l} aria-pressed={lead === l} onClick={() => onSession({ lead: l })}>D{l}</button>
            ))}
          </div>
          <button className="btn2" onClick={() => setBulletin(true)}><FiFileText /> Bulletin</button>
          <a className="btn2" href={api.exportCsvUrl(date, lead)} target="_blank" rel="noreferrer"><FiDownload /> CSV</a>
        </div>
      </div>

      {err && <div className="notice err">Could not load {date} · Day {lead}: {err}</div>}

      <div className="kpi-row">
        <div className="kpi">
          <div className="k">Regime detected</div>
          <div className="v" style={{ color: REGIME_COLOR[c?.regime?.name] }}>{c?.regime?.name || '—'}</div>
          <div className="u">confidence {pct(c?.regime?.confidence)}</div>
        </div>
        <div className={`kpi ${(counts?.red ?? 0) > 0 ? 'hot' : ''}`}>
          <div className="k">Districts in warning</div>
          <div className="v">{c?.summary?.districts_in_warning ?? '—'}<span className="u"> / {districts.length}</span></div>
          <div className="u cats">
            {['red', 'orange', 'yellow'].map((k) => (
              <span key={k}><i style={{ background: CATEGORY_COLOR[k] }} />{counts?.[k] ?? 0}</span>
            ))}
          </div>
        </div>
        <div className="kpi">
          <div className="k">Peak corrected rainfall</div>
          <div className="v mono">{fmt(c?.summary?.max_corrected_mm)}<span className="u"> mm</span></div>
          <div className="u">raw NWP peak {fmt(c?.summary?.max_raw_mm)} mm</div>
        </div>
        <div className="kpi">
          <div className="k">Heavy-rain skill (CSI)</div>
          <div className="v mono">{csi ? fmt(csi.corrected, 2) : '—'}</div>
          <div className="u">raw NWP {csi ? fmt(csi.raw, 2) : '—'} · {csi ? `${csi.vs_raw_pct > 0 ? '+' : ''}${csi.vs_raw_pct}%` : ''}</div>
        </div>
      </div>

      <div className="module-grid">
        {modules.map((m, i) => (
          <div key={m.n} className={`module-card c-${m.color}`} role="button" tabIndex={0}
               onClick={m.go} onKeyDown={(e) => (e.key === 'Enter' ? m.go() : null)}
               style={{ animationDelay: `${i * 0.06}s` }}>
            <div className="mc-glow" />
            <div className="mc-top">
              <div className="mc-icon"><m.icon /></div>
              <div className="mc-tags">{m.tags.map((t) => <span key={t}>{t}</span>)}</div>
            </div>
            <h3>{m.title}</h3>
            <div className="mc-metric">{m.metric}</div>
            <div className="mc-sub">{m.sub}</div>
            <p>{m.desc}</p>
            <div className="mc-bar"><div style={{ width: `${Math.round(Math.max(0.04, m.bar) * 100)}%` }} /></div>
            {m.barLabel && <div className="tiny muted" style={{ marginTop: 4 }}>{m.barLabel}</div>}
            <div className="mc-cta">Open module <FiArrowRight /></div>
          </div>
        ))}
      </div>

      <div className="dash-lower">
        <div className="glass">
          <div className="glass-head">
            <span>District warnings · {c?.date || '—'} · Day {lead}</span>
            <button className="link-btn" onClick={() => onOpenMap('category')}>Open full map <FiArrowRight /></button>
          </div>
          <IndiaMap baseGeo={base} studyGeo={geo} districts={districts} layer="category"
                    onSelect={() => onOpenMap('category')} interactive={false} style={{ height: 430 }} />
          <div className="legend-row">
            {['red', 'orange', 'yellow', 'green'].map((k) => (
              <span key={k}><i style={{ background: CATEGORY_COLOR[k] }} />{CATEGORY_META[k].label} ({CATEGORY_META[k].mm} mm)</span>
            ))}
          </div>
        </div>

        <div className="dash-side">
          <div className="glass">
            <div className="glass-head"><span>Regime posterior (mean over districts)</span></div>
            <div className="post-bars">
              {posterior.map(([name, p]) => (
                <div key={name} className="post-row">
                  <span className="name">{name}</span>
                  <div className="track"><div style={{ width: `${Math.max(1, p * 100)}%`, background: REGIME_COLOR[name] }} /></div>
                  <span className="val mono">{pct(p)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="glass">
            <div className="glass-head"><span>Five-day outlook</span><span className="tiny muted">districts warned by lead</span></div>
            <div className="outlook">
              {(c?.horizon || []).map((h) => (
                <button key={h.lead} className={`ol ${h.lead === lead ? 'on' : ''}`} onClick={() => onSession({ lead: h.lead })}>
                  <div className="ol-bar"><div style={{ height: `${Math.min(100, (h.districts_warned / Math.max(1, districts.length)) * 100 * 2.2 + 6)}%` }} /></div>
                  <div className="mono">{h.districts_warned}</div>
                  <div className="tiny muted">D{h.lead}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="glass">
            <div className="glass-head"><span><FiZap /> Significant days</span><span className="tiny muted">ranked by severity</span></div>
            <div className="event-list">
              {events.slice(0, 7).map((e) => (
                <button key={e.date} className={`ev ${e.date === c?.date ? 'on' : ''}`} onClick={() => onSession({ date: e.date })}>
                  <span className="mono">{e.date}</span>
                  <span className="muted small">{e.regime_name}</span>
                  <span className="ev-n">{e.red > 0 ? `${e.red} red` : `${e.warned_districts} warned`}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {bulletin && <BulletinModal date={c?.date || date} lead={lead} onClose={() => setBulletin(false)} />}
    </div>
  );
}
