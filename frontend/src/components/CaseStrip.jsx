import React, { useEffect, useState } from 'react';
import { api } from '../api';

/**
 * Documented cases.
 *
 * Three named events with a known failure mode for the raw model, each one click
 * away. A duty officer comparing a live day against a case they remember is the
 * fastest way to build trust in a correction; for a jury it is also the shortest
 * path from "interesting" to "I can see what this does".
 */
export default function CaseStrip({ onGo, currentDate }) {
  const [cases, setCases] = useState([]);
  const [open, setOpen] = useState(null);

  useEffect(() => { api.cases().then((d) => setCases(d.cases || [])).catch(() => setCases([])); }, []);
  if (!cases.length) return null;

  return (
    <div className="panel">
      <div className="panel-head" style={{ paddingBottom: 6 }}>
        <span className="panel-title">Documented cases</span>
        <span className="tiny muted">
          named events with a known raw-model failure · one click to load the day
        </span>
      </div>
      <div className="panel-body" style={{ paddingTop: 8 }}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {cases.map((c) => {
            const active = c.dates.includes(currentDate);
            return (
              <button key={c.id} className="btn"
                      onClick={() => { onGo(c.dates[0], 1); setOpen(open === c.id ? null : c.id); }}
                      title={`${c.regime} — ${c.description}`}
                      style={active ? { borderColor: 'var(--cyan)', color: 'var(--cyan)', boxShadow: '0 0 14px rgba(0,212,255,.18)' } : undefined}>
                {c.name}
              </button>
            );
          })}
        </div>
        {open && (
          <div className="small muted" style={{ marginTop: 8 }}>
            {cases.find((c) => c.id === open)?.description}
            {' '}Dates in the archive: <span className="mono">
              {cases.find((c) => c.id === open)?.dates.join(', ')}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
