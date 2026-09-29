import React, { useEffect, useState } from 'react';
import { api } from '../api';
import IndiaMap from './IndiaMap';
import {
  CATEGORY_META, CATEGORY_FILL, REGIME_COLOR, RAIN_LEGEND, rainColor, adjustmentColor,
  probabilityColor, categoryBasis, fmt,
} from '../lib/format';

const LAYERS = [
  { id: 'category', label: 'Warning category' },
  { id: 'corrected', label: 'Corrected rainfall' },
  { id: 'raw', label: 'Raw model' },
  { id: 'adjustment', label: 'Correction applied' },
  { id: 'p_heavy', label: 'P(≥ 64.5 mm)' },
  { id: 'regime', label: 'Regime' },
];
const VIEWS = [
  { id: 'district', label: 'Districts' },
  { id: 'grid', label: '0.5° grid' },
];

export default function MapPanel({
  geojson, districts, selectedId, onSelect, layer, onLayer, view, onView,
  grid, gridDates, gridDate, onGridDate, date, lead, provenance,
}) {
  const [hovered, setHovered] = useState(null);

  // Offline SVG map (no tile server): the 641 Census-2011 districts as the base layer.
  const [base, setBase] = useState(null);
  useEffect(() => { api.geojsonAll().then(setBase).catch(() => setBase(null)); }, []);

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">
          Spatial view
          <span className="muted small" style={{ marginLeft: 8, fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
            {hovered ? (
              <>
                {hovered.district_name} — {fmt(hovered.corrected_mm)} mm corrected, raw{' '}
                {fmt(hovered.raw_mm)} mm
                {hovered.category && (
                  <>
                    {' · '}
                    <span className={`cat-chip cat-${hovered.category}`}>
                      {CATEGORY_META[hovered.category].short}
                    </span>{' '}
                    {categoryBasis(hovered).text}
                  </>
                )}
              </>
            ) : 'click a district to inspect it'}
          </span>
        </span>
        <div className="cb-group">
          <div className="seg">
            {VIEWS.map((v) => (
              <button key={v.id} aria-pressed={view === v.id} onClick={() => onView(v.id)}>{v.label}</button>
            ))}
          </div>
          {view === 'grid' && gridDates?.length > 1 && (
            <select value={gridDate} onChange={(e) => onGridDate(e.target.value)} title="sampled archive dates">
              {gridDates.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          )}
        </div>
      </div>

      <div className="map-wrap" role="application"
           aria-label={`District map, layer: ${LAYERS.find((l) => l.id === layer)?.label || layer}`}>
        <div className="map-overlay">
          <div className="seg">
            {LAYERS.map((l) => (
              <button key={l.id} aria-pressed={layer === l.id} onClick={() => onLayer(l.id)}>{l.label}</button>
            ))}
          </div>
        </div>

        <IndiaMap
          baseGeo={base}
          studyGeo={view === 'district' ? geojson : null}
          districts={districts}
          layer={layer}
          selectedId={selectedId}
          onSelect={onSelect}
          cells={view === 'grid' ? grid?.cells : null}
          showPulses={view === 'district'}
          onHover={setHovered}
          style={{ height: '100%' }}
        />

        <div className="map-meta tiny muted mono">
          {date ? `${date} · Day ${lead}` : 'spatial view'}
          {provenance ? ` · ${String(provenance).replace(/_/g, ' ').toLowerCase()}` : ''}
        </div>

        <div className="map-legend">
          {layer === 'category' && (
            <>
              <div><b className="small">IMD warning category</b></div>
              {['red', 'orange', 'yellow', 'green'].map((c) => (
                <div key={c}>
                  <span className="swatch" style={{ background: CATEGORY_FILL[c] }} />
                  {CATEGORY_META[c].short.toLowerCase()} · {CATEGORY_META[c].mm} mm
                </div>
              ))}
            </>
          )}
          {(layer === 'corrected' || layer === 'raw') && (
            <>
              <div><b className="small">24 h rainfall</b></div>
              {RAIN_LEGEND.map(([label, v]) => (
                <div key={label}><span className="swatch" style={{ background: rainColor(v) }} />{label} mm</div>
              ))}
            </>
          )}
          {layer === 'adjustment' && (
            <>
              <div><b className="small">corrected − raw</b></div>
              <div><span className="swatch" style={{ background: adjustmentColor(30) }} />wetter</div>
              <div><span className="swatch" style={{ background: adjustmentColor(0) }} />unchanged</div>
              <div><span className="swatch" style={{ background: adjustmentColor(-30) }} />drier</div>
            </>
          )}
          {layer === 'p_heavy' && (
            <>
              <div><b className="small">exceedance</b></div>
              <div><span className="swatch" style={{ background: probabilityColor(0.8) }} />≥ 0.75</div>
              <div><span className="swatch" style={{ background: probabilityColor(0.5) }} />0.50–0.75</div>
              <div><span className="swatch" style={{ background: probabilityColor(0.3) }} />0.30–0.50</div>
              <div><span className="swatch" style={{ background: probabilityColor(0.15) }} />0.15–0.30</div>
            </>
          )}
          {layer === 'regime' && (
            <>
              {Object.entries(REGIME_COLOR).map(([name, color]) => (
                <div key={name}><span className="swatch" style={{ background: color }} />{name}</div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
