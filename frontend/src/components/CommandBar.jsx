import React from 'react';
import Icon from './Icon';
import { CATEGORY_META, fmt } from '../lib/format';

/**
 * One row of controls: the date, the lead day, and the two things a duty
 * officer does with a screen (issue the bulletin, export the table).
 *
 * The lead buttons double as the 5-day outlook — each one shows how many
 * districts are warned on that day — so there is one control for lead time,
 * not a selector and a strip that disagree.
 */
export default function CommandBar({
  date, onDate, lead, onLead, horizon, onBulletin, onExport, onHelp, dates,
}) {
  const days = dates || [];
  const idx = days.indexOf(date);
  const step = (delta) => {
    const base = idx === -1 ? days.length - 1 : idx;
    const next = days[Math.min(Math.max(base + delta, 0), days.length - 1)];
    if (next) onDate(next);
  };
  const byLead = Object.fromEntries((horizon || []).map((h) => [h.lead, h]));

  return (
    <div className="commandbar">
      <div className="cb-group">
        <button className="btn icon" onClick={() => step(-1)} title="Previous day (←)" aria-label="previous day">◀</button>
        <input type="date" value={date || ''} min={days[0]} max={days[days.length - 1]}
               onChange={(e) => e.target.value && onDate(e.target.value)} aria-label="valid date" />
        <button className="btn icon" onClick={() => step(1)} title="Next day (→)" aria-label="next day">▶</button>
      </div>

      <div className="seg leads" role="group" aria-label="forecast lead day">
        {[1, 2, 3, 4, 5].map((d) => {
          const h = byLead[d];
          return (
            <button key={d} aria-pressed={lead === d} onClick={() => onLead(d)}
                    title={h ? `Day ${d}: ${h.districts_warned} districts warned, peak ${fmt(h.max_corrected_mm)} mm (press ${d})` : `Day ${d} (press ${d})`}>
              <span className="lead-day">Day {d}</span>
              {h && (
                <span className="lead-sub">
                  {h.red > 0 && <i className="dot" style={{ background: 'var(--red)' }} />}
                  {h.districts_warned} warned
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className="cb-group" style={{ marginLeft: 'auto' }}>
        <button className="btn primary" onClick={onBulletin} title="Open the bulletin for this day (b)">
          <Icon name="file" size={15} /> Bulletin
        </button>
        <button className="btn" onClick={onExport} title="Download the warning table as CSV">
          <Icon name="download" size={15} /> CSV
        </button>
        <button className="btn icon" onClick={onHelp} title="Keyboard shortcuts (?)" aria-label="keyboard shortcuts">
          <Icon name="help" size={15} />
        </button>
      </div>
    </div>
  );
}

export function categoryTooltip(cat) {
  const meta = CATEGORY_META[cat];
  return meta ? `${meta.label} (${meta.action})` : '';
}
