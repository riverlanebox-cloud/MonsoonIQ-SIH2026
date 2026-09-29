import React, { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import { fmt } from '../lib/format';
import DottedIndia from './DottedIndia';
import { isReal, provenanceFooter } from '../lib/provenance';

/**
 * Opening screen.
 *
 * SAGAR-style hero: a dot-matrix map of India behind a centred title and two
 * calls to action. The dots are not decoration — every modelled district is
 * tinted by its current IMD warning category for the most warning-heavy monsoon
 * day in the archive. The whole
 * console is one click away, and the numbers on the strip are the same numbers
 * the console will show, so nothing here is decoration.
 */
export default function Landing({ onEnter, onMethod }) {
  const [geojson, setGeojson] = useState(null);
  const [console_, setConsole_] = useState(null);
  const [defaultDate, setDefaultDate] = useState(null);

  useEffect(() => {
    api.geojson().then(setGeojson).catch(() => setGeojson(null));
    api.timeline()
      .then((t) => setDefaultDate(t.meta?.default_date || t.days?.[t.days.length - 1]?.date))
      .catch(() => setDefaultDate(null));
  }, []);

  useEffect(() => {
    api.console(defaultDate || undefined, 1, false)
      .then(setConsole_)
      .catch(() => setConsole_(null));
  }, [defaultDate]);

  const districts = console_?.districts || [];
  const byId = useMemo(() => {
    const m = {};
    for (const d of districts) m[d.district_id] = d;
    return m;
  }, [console_]);

  const counts = console_?.summary?.counts;
  const maxDistrict = districts[0];

  return (
    <div className="landing">
      <div className="landing-map">
        <DottedIndia geojson={geojson} byId={byId} />
      </div>
      <div className="landing-veil" />

      <div className="landing-inner">
        <div className="landing-kicker anim-up">
          SIH26080 · Ministry of Earth Sciences · Disaster Management
        </div>
        <h1 className="landing-title anim-up d1">MonsoonIQ</h1>
        <p className="landing-sub anim-up d2">
          Turns a raw rainfall forecast into district warnings a duty officer can act on.
          The correction depends on the <b>monsoon regime</b> of the day, and every claim is
          checked against what actually fell.
        </p>

        <div className="landing-ctas anim-up d3">
          <button className="cta primary" onClick={() => onEnter(defaultDate)}>
            Open the console
          </button>
          <button className="cta teal" onClick={onMethod}>
            See the results
          </button>
        </div>

        <div className="landing-stats anim-up d3">
          <div className="landing-stat">
            <div className="k">Showing</div>
            <div className="v" style={{ fontSize: 15 }}>{console_?.date || '—'}</div>
            <div className="tiny muted">{isReal(console_?.provenance) ? 'the busiest day the models never saw' : 'the busiest day in the archive'}</div>
          </div>
          <div className="landing-stat">
            <div className="k">Districts warned</div>
            <div className="v" style={{ color: '#ff7369' }}>
              {console_?.summary?.districts_in_warning ?? '—'}
              <span className="muted" style={{ fontSize: 13 }}> / {districts.length || '—'}</span>
            </div>
            <div className="tiny muted">
              {counts ? `${counts.red} red · ${counts.orange} orange · ${counts.yellow} yellow` : 'loading'}
            </div>
          </div>
          <div className="landing-stat">
            <div className="k">Regime</div>
            <div className="v" style={{ fontSize: 15 }}>
              {console_?.regime?.name || '—'}
            </div>
            <div className="tiny muted">
              confidence {console_?.regime?.confidence !== undefined
                ? `${(console_.regime.confidence * 100).toFixed(0)}%` : '—'}
            </div>
          </div>
          <div className="landing-stat">
            <div className="k">Heaviest forecast</div>
            <div className="v">
              {fmt(console_?.summary?.max_corrected_mm)}
              <span className="muted" style={{ fontSize: 13 }}> mm</span>
            </div>
            <div className="tiny muted">
              raw model {fmt(console_?.summary?.max_raw_mm)} mm
              {maxDistrict ? ` · ${maxDistrict.district_name}` : ''}
            </div>
          </div>
        </div>

        <div className="landing-hint anim-up d3">
          <span className="badge cyan" style={{ marginRight: 8 }}>{isReal(console_?.provenance) ? 'IMD + GFS, 2021–2025' : 'synthetic archive'}</span>
          <span className="badge green" style={{ marginRight: 8 }}>53 districts</span>
          <span className="badge">Day 1–5</span>
        </div>
      </div>

      <div className="landing-foot">
        {provenanceFooter(console_?.provenance)}
      </div>
    </div>
  );
}
