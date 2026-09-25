import React from 'react';
import { CATEGORY_META } from '../lib/format';

const LEADS = [1, 2, 3, 4, 5];

export default function CommandBar({
  date, onDate, lead, onLead, counts, events, onJumpEvent, query, onQuery,
  onExport, onBulletin, onRefresh, dates, busy, modelScope,
}) {
  const days = dates || [];
  const idx = days.indexOf(date);
  const step = (delta) => {
    const base = idx === -1 ? days.length - 1 : idx;
    const next = days[Math.min(Math.max(base + delta, 0), days.length - 1)];
    if (next) onDate(next);
  };
  return (
    <div className="commandbar">
      <div className="cb-group">
        <span className="cb-label">Valid for</span>
        <button className="btn icon" onClick={() => step(-1)} title="Previous day (←)"
                aria-label="previous day">◀</button>
        <input type="date" value={date || ''} min={days[0]} max={days[days.length - 1]}
               onChange={(e) => e.target.value && onDate(e.target.value)} />
        <button className="btn icon" onClick={() => step(1)} title="Next day (→)"
                aria-label="next day">▶</button>
      </div>

      <div className="cb-group">
        <span className="cb-label">Lead</span>
        <div className="seg" role="group" aria-label="forecast lead time">
          {LEADS.map((d) => (
            <button key={d} aria-pressed={lead === d} onClick={() => onLead(d)}
                    title={`Day ${d} (press ${d})`}>D{d}</button>
          ))}
        </div>
      </div>

      <div className="cb-group">
        <span className="cb-label">Significant days</span>
        <select value="" onChange={(e) => e.target.value && onJumpEvent(e.target.value)}
                title="Jump to a ranked significant day ([ and ])">
          <option value="">jump to…</option>
          {events.map((e) => (
            <option key={e.date} value={e.date}>
              {e.date} · {e.regime_name} · {e.red} red / {e.orange} orange
            </option>
          ))}
        </select>
      </div>

      <div className="cb-group" style={{ marginLeft: 'auto' }}>
        <input type="search" value={query} placeholder="find district  ( / )"
               onChange={(e) => onQuery(e.target.value)} />
      </div>

      <div className="cb-group">
        {counts && (
          <span className="chip" title="districts in each IMD warning category">
            <span className="dot" style={{ background: 'var(--red)' }} />{counts.red}
            <span className="dot" style={{ background: 'var(--orange)', marginLeft: 6 }} />{counts.orange}
            <span className="dot" style={{ background: 'var(--yellow)', marginLeft: 6 }} />{counts.yellow}
            <span className="muted" style={{ marginLeft: 4 }}>warned districts</span>
          </span>
        )}
        <button className="btn" onClick={onBulletin} title="Issue a district warning bulletin (b)">
          Bulletin
        </button>
        <button className="btn" onClick={onExport} title="Download this view as CSV">CSV</button>
        <button className="btn" onClick={onRefresh} disabled={busy} title="Reload from the API">
          {busy ? '…' : 'Refresh'}
        </button>
      </div>

      {modelScope && (
        <span className="tiny muted" style={{ width: '100%' }}>
          {modelScope === 'per_lead'
            ? 'Lead-specific models: each lead has its own fitted correction'
            : 'Day-1 models applied to this lead'}
        </span>
      )}
    </div>
  );
}

export function categoryTooltip(cat) {
  const meta = CATEGORY_META[cat];
  return meta ? `${meta.label} (${meta.action})` : '';
}
