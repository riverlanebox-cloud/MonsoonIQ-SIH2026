import React, { useEffect, useMemo, useState } from 'react';
import { GeoJSON, MapContainer, TileLayer } from 'react-leaflet';
import { api } from '../api';
import { CATEGORY_FILL, CATEGORY_META, fmt } from '../lib/format';

/**
 * Opening screen.
 *
 * One frame that states what the system is and shows the live state of the
 * country underneath it: every district tinted by its current IMD warning
 * category for the most warning-heavy monsoon day in the archive. The whole
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
        <MapContainer
          center={[22.8, 79.5]} zoom={4.4} zoomControl={false} dragging={false}
          scrollWheelZoom={false} doubleClickZoom={false} attributionControl={false}
          keyboard={false} preferCanvas
        >
          <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png" />
          {geojson && (
            <GeoJSON
              key={`${districts.length}-${console_?.date || 'loading'}`}
              data={geojson}
              style={(f) => {
                const d = byId[f.properties.district_id];
                return {
                  color: d && d.category !== 'green' ? '#0a1929' : '#12293c',
                  weight: d && d.category === 'red' ? 1.4 : 0.6,
                  fillColor: d ? CATEGORY_FILL[d.category] : '#0e1f2e',
                  fillOpacity: d ? (d.category === 'red' ? 0.95 : d.category === 'orange' ? 0.9 : 0.82) : 0.5,
                };
              }}
              onEachFeature={(feature, layer) => {
                const d = byId[feature.properties.district_id];
                if (!d) return;
                layer.bindTooltip(
                  `<div class="district-tip"><b>${d.district_name}</b> · ${d.state_name}<br/>
                   ${CATEGORY_META[d.category].short} — ${CATEGORY_META[d.category].action}<br/>
                   corrected ${fmt(d.corrected_mm)} mm · raw ${fmt(d.raw_mm)} mm<br/>
                   P(≥64.5 mm) ${(d.p_heavy * 100).toFixed(0)}%</div>`,
                  { sticky: true, direction: 'top' },
                );
              }}
            />
          )}
        </MapContainer>
      </div>
      <div className="landing-veil" />

      <div className="landing-inner">
        <div className="landing-kicker anim-up">
          SIH26080 · Ministry of Earth Sciences · Disaster Management
        </div>
        <h1 className="landing-title anim-up d1">MonsoonIQ</h1>
        <p className="landing-sub anim-up d2">
          Regime-aware post-processing of numerical rainfall forecasts for India.
          The dynamical field is corrected <b>per monsoon regime</b>, converted into
          IMD-scaled district warnings with exceedance probabilities, and verified —
          including the result that does not work.
        </p>

        <div className="landing-ctas anim-up d3">
          <button className="cta primary" onClick={() => onEnter(defaultDate)}>
            Enter operations console
          </button>
          <button className="cta ghost" onClick={onMethod}>
            Verification &amp; method
          </button>
        </div>

        <div className="landing-stats anim-up d3">
          <div className="landing-stat">
            <div className="k">Valid for</div>
            <div className="v" style={{ fontSize: 15 }}>{console_?.date || '—'}</div>
            <div className="tiny muted">most warning-heavy monsoon day</div>
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
            <div className="k">Regime detected</div>
            <div className="v" style={{ fontSize: 15 }}>
              {console_?.regime?.name || '—'}
            </div>
            <div className="tiny muted">
              confidence {console_?.regime?.confidence !== undefined
                ? `${(console_.regime.confidence * 100).toFixed(0)}%` : '—'}
            </div>
          </div>
          <div className="landing-stat">
            <div className="k">Peak corrected</div>
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
          <span className="badge cyan" style={{ marginRight: 8 }}>3 screens</span>
          <span className="badge green" style={{ marginRight: 8 }}>keyboard-first</span>
          <span className="badge">runs offline</span>
          <div style={{ marginTop: 10 }}>
            Console shortcuts: <kbd>←</kbd> <kbd>→</kbd> days · <kbd>1</kbd>–<kbd>5</kbd> lead ·
            <kbd>n</kbd>/<kbd>p</kbd> significant days · <kbd>l</kbd> map layer · <kbd>b</kbd> bulletin ·
            <kbd>s</kbd> guided demo · <kbd>?</kbd> help
          </div>
        </div>
      </div>

      <div className="landing-foot">
        Research prototype · forecast fields come from a synthetic physically-plausible archive
        generated in this repository ({console_?.provenance || 'SYNTHETIC_PHYSICALLY_PLAUSIBLE'}) and are
        <b> not</b> official IMD/NCMRWF products. Do not use for public warnings.
      </div>
    </div>
  );
}
