import React, { useEffect, useMemo, useState } from 'react';
import { FiArrowRight, FiMap, FiBarChart2, FiDatabase } from 'react-icons/fi';
import { api } from '../api';
import { CATEGORY_COLOR, fmt } from '../lib/format';
import { WIDTH, HEIGHT, project } from '../lib/geo';
import dotsData from '../assets/india_dots.json';
import { LogoMark } from './AppHeader';

/**
 * Opening screen.
 *
 * A dotted map of India lit by the live forecast: every dot belongs to a real Census-2011
 * district; dots inside forecast districts take that district's IMD warning colour, and the
 * two monsoon branches (Arabian Sea / Somali jet and Bay of Bengal) are drawn as moving flow
 * lines. The numbers in the strip are the numbers the console will show.
 */
const FLOWS = [
  { id: 'as', label: 'Arabian Sea branch', pts: [[67.8, 7.2], [71.5, 9.5], [75.6, 11.2]] },
  { id: 'wc', label: '', pts: [[69.5, 11.5], [71.0, 16.0], [72.9, 19.2]] },
  { id: 'bob', label: 'Bay of Bengal branch', pts: [[84.0, 7.0], [89.5, 14.0], [91.6, 25.3]] },
  { id: 'gp', label: '', pts: [[90.0, 24.8], [85.0, 27.2], [78.0, 28.6]] },
];

function flowPath(pts) {
  const p = pts.map(([lon, lat]) => project(lon, lat));
  return `M${p[0][0]},${p[0][1]} Q${p[1][0]},${p[1][1]} ${p[2][0]},${p[2][1]}`;
}

export default function Landing({ onEnter, onTab }) {
  const [console_, setConsole_] = useState(null);
  const [health, setHealth] = useState(null);
  const [defaultDate, setDefaultDate] = useState(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.timeline()
      .then((t) => setDefaultDate(t.meta?.default_date || t.days?.[t.days.length - 1]?.date))
      .catch(() => setDefaultDate(null));
  }, []);

  useEffect(() => {
    api.console(defaultDate || undefined, 1, false).then(setConsole_).catch(() => setConsole_(null));
  }, [defaultDate]);

  const byId = useMemo(() => {
    const m = {};
    for (const d of console_?.districts || []) m[d.district_id] = d;
    return m;
  }, [console_]);

  const dots = useMemo(() => dotsData.dots.map(([lon, lat, id]) => {
    const [x, y] = project(lon, lat);
    const d = id ? byId[id] : null;
    return { x, y, id, cat: d?.category, key: `${lon},${lat}` };
  }), [byId]);

  const counts = console_?.summary?.counts;
  const top = console_?.districts?.[0];
  const real = health?.profile === 'real';

  return (
    <div className="landing2">
      <div className="landing2-bg" aria-hidden="true">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="xMidYMid meet">
          <defs>
            <linearGradient id="flow-grad" x1="0" x2="1">
              <stop offset="0" stopColor="#00d4ff" stopOpacity="0" />
              <stop offset=".25" stopColor="#00d4ff" stopOpacity=".95" />
              <stop offset="1" stopColor="#20b2aa" stopOpacity=".2" />
            </linearGradient>
          </defs>
          {dots.map((d) => (
            <circle key={d.key} cx={d.x} cy={d.y} r={d.cat && d.cat !== 'green' ? 2.6 : 1.75}
                    className={`dot ${d.id ? 'study' : ''} ${d.cat || ''}`}
                    style={d.cat && d.cat !== 'green' ? { fill: CATEGORY_COLOR[d.cat] } : undefined} />
          ))}
          {FLOWS.map((f, i) => (
            <g key={f.id}>
              <path d={flowPath(f.pts)} className="flow-base" />
              <path d={flowPath(f.pts)} className="flow-dash" style={{ animationDelay: `${i * 0.6}s` }} />
            </g>
          ))}
          {FLOWS.filter((f) => f.label).map((f) => {
            const [x, y] = project(f.pts[0][0] + 0.4, f.pts[0][1] - 0.2);
            return <text key={f.id} x={x} y={y + 14} className="flow-label">{f.label}</text>;
          })}
          {dots.filter((d) => d.cat === 'red' || d.cat === 'orange').filter((_, i) => i % 3 === 0).map((d) => (
            <circle key={`p${d.key}`} cx={d.x} cy={d.y} r="3" className={`land-pulse ${d.cat}`} />
          ))}
        </svg>
        <div className="rain" />
      </div>

      <div className="landing2-content">
        <div className="kicker fade-up" style={{ animationDelay: '0s' }}>
          SIH 2026 · PS 26080 · Ministry of Earth Sciences · NCMRWF
        </div>
        <div className="hero-logo fade-up" style={{ animationDelay: '.1s' }}><LogoMark size={64} /></div>
        <h1 className="hero-title fade-up" style={{ animationDelay: '.15s' }}>
          Monsoon<span>IQ</span>
        </h1>
        <p className="hero-sub fade-up" style={{ animationDelay: '.3s' }}>
          Regime-aware AI post-processing of monsoon rainfall forecasts
        </p>
        <p className="hero-desc fade-up" style={{ animationDelay: '.4s' }}>
          Identify the weather regime — active, break, depression, orographic, coastal, western
          disturbance — then correct the raw NWP rainfall with the expert trained for it.
          District warnings on the IMD scale, heavy-rain probabilities, and verification with
          RMSE, ETS, CSI, POD, FAR and FSS.
        </p>

        <div className="hero-ctas fade-up" style={{ animationDelay: '.55s' }}>
          <button className="cta primary" onClick={() => onEnter('dashboard')}>
            Enter Dashboard <FiArrowRight />
          </button>
          <button className="cta teal" onClick={() => onEnter('map')}>
            <FiMap /> Live Forecast Map
          </button>
          <button className="cta ghost" onClick={() => onTab('skill')}>
            <FiBarChart2 /> Verification Report
          </button>
        </div>

        <div className="hero-strip fade-up" style={{ animationDelay: '.7s' }}>
          <div className="hs">
            <div className="k">Valid for</div>
            <div className="v mono">{console_?.date || '—'}</div>
            <div className="u">Day-1 forecast · most significant day</div>
          </div>
          <div className="hs">
            <div className="k">Districts warned</div>
            <div className="v"><span style={{ color: 'var(--red)' }}>{console_?.summary?.districts_in_warning ?? '—'}</span>
              <span className="u"> / {console_?.districts?.length ?? '—'}</span></div>
            <div className="u">{counts ? `${counts.red} red · ${counts.orange} orange · ${counts.yellow} yellow` : 'loading'}</div>
          </div>
          <div className="hs">
            <div className="k">Regime detected</div>
            <div className="v small-v">{console_?.regime?.name || '—'}</div>
            <div className="u">confidence {console_?.regime ? `${Math.round(console_.regime.confidence * 100)}%` : '—'}</div>
          </div>
          <div className="hs">
            <div className="k">Peak corrected</div>
            <div className="v mono">{fmt(console_?.summary?.max_corrected_mm)}<span className="u"> mm</span></div>
            <div className="u">raw NWP {fmt(console_?.summary?.max_raw_mm)} mm{top ? ` · ${top.district_name}` : ''}</div>
          </div>
        </div>

        <div className="hero-tags fade-up" style={{ animationDelay: '.85s' }}>
          <span className="tag cyan">7 weather regimes</span>
          <span className="tag green">Mixture of experts</span>
          <span className="tag yellow">Day 1–5 leads</span>
          <span className="tag orange">IMD warning scale</span>
          <span className="tag teal">641 districts mapped</span>
          <button className="tag link" onClick={() => onTab('data')}><FiDatabase /> {real ? 'IMD observed data' : 'Synthetic benchmark · real-data pipeline ready'}</button>
        </div>
      </div>

      <div className="landing2-foot">
        {real
          ? 'Forecasts verified against IMD 0.25° gridded rainfall. Research prototype — not an official IMD/NCMRWF product.'
          : 'Demo archive is synthetic and reproducible (not official IMD data); run the real-data pipeline to verify on IMD observations.'}
      </div>
    </div>
  );
}
