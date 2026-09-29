import React, { useMemo, useRef, useState } from 'react';
import { REGIME_COLOR, REGIME_SHORT, CATEGORY_COLOR, shortDate } from '../lib/format';

/**
 * Season strip: one bar per day showing the dominant regime, with warning
 * counts underneath. Click or drag anywhere to change the valid date. This is
 * the fastest way to see how skill and regime move together, and it removes the
 * need for a date picker in most sessions.
 */
export default function RegimeTimeline({ days, currentDate, lead = 1, onSelect, events = [], cases = [] }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);

  const { spells, warnings, maxWarn, ticks } = useMemo(() => {
    if (!days?.length) return { spells: [], warnings: [], maxWarn: 1, ticks: [] };
    const key = `d${lead}`;
    const spells = [];
    let current = null;
    const warnings = [];
    for (const d of days) {
      const l = d.leads?.[key] || d.leads?.d1 || {};
      warnings.push((l.red || 0) * 3 + (l.orange || 0) * 2 + (l.yellow || 0) * 1);
      if (!current || current.regime !== d.regime_name) {
        current = { regime: d.regime_name, start: d.date, end: d.date, days: 1 };
        spells.push(current);
      } else {
        current.end = d.date;
        current.days += 1;
      }
    }
    const maxWarn = Math.max(1, ...warnings);
    const ticks = [];
    let lastYear = null;
    for (const d of days) {
      const y = d.date.slice(0, 4);
      if (y !== lastYear) { ticks.push({ date: d.date, year: y }); lastYear = y; }
    }
    return { spells, warnings, maxWarn, ticks };
  }, [days, lead]);

  const pick = (clientX) => {
    if (!ref.current || !days?.length) return;
    const rect = ref.current.getBoundingClientRect();
    const frac = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 0.9999);
    const i = Math.floor(frac * days.length);
    onSelect(days[i].date);
  };

  if (!days?.length) {
    return <div className="panel"><div className="panel-body muted small">Timeline unavailable — run <span className="mono">python -m src.console.artifacts</span>.</div></div>;
  }

  const cursorPct = (days.findIndex((d) => d.date === currentDate) / days.length) * 100;
  const hoverEntry = hover !== null ? days[hover] : null;

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">
          Season timeline
          <span className="muted small" style={{ marginLeft: 8, fontWeight: 400 }}>
            {hoverEntry
              ? `${hoverEntry.date} · ${hoverEntry.regime_name} · ${hoverEntry.observed_heavy_districts} districts had heavy rain`
              : `${days.length} days · click a day to open it`}
          </span>
        </span>
        <div className="cb-group">
          {events.length > 0 && (
            <select value="" onChange={(e) => e.target.value && onSelect(e.target.value)}
                    aria-label="jump to a significant day" title="Days with the most warnings (n / p)">
              <option value="">Busiest days…</option>
              {events.map((e) => (
                <option key={e.date} value={e.date}>
                  {e.date} · {e.regime_name} · {e.red} red, {e.orange} orange
                </option>
              ))}
            </select>
          )}
        </div>
      </div>
      <div className="timeline">
        <div
          className="timeline-track"
          ref={ref}
          onMouseMove={(e) => {
            const rect = ref.current.getBoundingClientRect();
            const frac = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 0.9999);
            setHover(Math.floor(frac * days.length));
          }}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => pick(e.clientX)}
          onMouseDown={(e) => pick(e.clientX)}
        >
          {spells.map((s) => {
            const start = days.findIndex((d) => d.date === s.start);
            const left = (start / days.length) * 100;
            const width = (s.days / days.length) * 100;
            return (
              <div key={`${s.regime}-${s.start}`} className="timeline-band"
                   style={{ left: `${left}%`, width: `${width}%`,
                            background: REGIME_COLOR[s.regime] || '#7c8fa3', opacity: 0.42 }} />
            );
          })}
          {warnings.map((w, i) => (
            <div key={days[i].date}
                 style={{
                   position: 'absolute',
                   left: `${(i / days.length) * 100}%`,
                   width: `${Math.max(100 / days.length, 0.6)}%`,
                   bottom: 0,
                   height: `${Math.max((w / maxWarn) * 22, w > 0 ? 2 : 0)}px`,
                   background: w >= 12 ? CATEGORY_COLOR.red : w >= 6 ? CATEGORY_COLOR.orange : CATEGORY_COLOR.yellow,
                 }} />
          ))}
          {cursorPct >= 0 && <div className="timeline-cursor" style={{ left: `${cursorPct}%` }} />}
        </div>
        <div className="timeline-ticks">
          {ticks.map((t) => <span key={t.year}>{t.year}</span>)}
        </div>
        <div className="tiny muted" style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 4 }}>
          {Object.entries(REGIME_COLOR).map(([name, color]) => (
            <span key={name}>
              <span className="swatch" style={{ background: color }} />
              {REGIME_SHORT[name] || name}
            </span>
          ))}
          <span>· showing {shortDate(currentDate)}</span>
        </div>
        {cases.length > 0 && (
          <div className="cases-row">
            <span className="tiny muted">Documented cases</span>
            {cases.map((c) => {
              const active = c.dates.includes(currentDate);
              return (
                <button key={c.id} className={`chip case${active ? ' on' : ''}`}
                        onClick={() => onSelect(c.dates[0], 1)} title={`${c.regime} — ${c.description}`}>
                  {c.name}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
