import React, { useMemo, useState } from 'react';
import { GeoJSON, MapContainer, TileLayer, CircleMarker, Tooltip, Rectangle } from 'react-leaflet';
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

  const styleFor = (featureId) => {
    const d = districts?.find((x) => x.district_id === featureId);
    if (!d) return { color: '#12293c', weight: 1, fillColor: '#0e1f2e', fillOpacity: 0.55 };
    const selected = d.district_id === selectedId;
    let fill = '#16324a';
    if (layer === 'category') fill = CATEGORY_FILL[d.category];
    else if (layer === 'corrected') fill = rainColor(d.corrected_mm);
    else if (layer === 'raw') fill = rainColor(d.raw_mm);
    else if (layer === 'adjustment') fill = adjustmentColor(d.adjustment_mm);
    else if (layer === 'p_heavy') fill = probabilityColor(d.p_heavy);
    else if (layer === 'regime') fill = REGIME_COLOR[d.regime] || '#9aa7b4';
    return {
      color: selected ? '#00d4ff' : (d.category === 'green' ? '#12293c' : '#061019'),
      weight: selected ? 2.6 : (d.category === 'red' ? 1.3 : 0.7),
      fillColor: fill,
      fillOpacity: d.category === 'green' ? 0.8 : 0.94,
    };
  };

  const onEach = (feature, lyr) => {
    const d = districts?.find((x) => x.district_id === feature.properties.district_id);
    lyr.on({
      click: () => onSelect(feature.properties.district_id),
      mouseover: () => { lyr.setStyle({ weight: 2.4, color: '#00d4ff' }); setHovered(d); },
      mouseout: () => { lyr.setStyle(styleFor(feature.properties.district_id)); setHovered(null); },
    });
  };

  const center = useMemo(() => [22.5, 79.5], []);

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

        <MapContainer center={center} zoom={5} scrollWheelZoom preferCanvas>
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; OpenStreetMap, &copy; CARTO'
          />

          {view === 'district' && geojson && (
            <GeoJSON
              key={`${layer}-${selectedId}-${districts?.length || 0}`}
              data={geojson}
              style={(f) => styleFor(f.properties.district_id)}
              onEachFeature={onEach}
            />
          )}

          {view === 'grid' && grid?.cells?.map((c, i) => (
            <Rectangle
              key={`${c.lat}-${c.lon}-${i}`}
              bounds={[[c.lat - 0.25, c.lon - 0.25], [c.lat + 0.25, c.lon + 0.25]]}
              pathOptions={{
                stroke: false,
                fillColor: rainColor(layer === 'raw' ? c.raw_mm : (c.corrected_mm ?? c.observed_mm)),
                fillOpacity: 0.9,
              }}
            >
              <Tooltip className="district-tip">
                <b>{fmt(c.corrected_mm ?? c.observed_mm)} mm</b> corrected<br />
                raw {fmt(c.raw_mm)} mm · observed {fmt(c.observed_mm)} mm<br />
                <span className="muted">{c.lat.toFixed(2)}°N {c.lon.toFixed(2)}°E</span>
              </Tooltip>
            </Rectangle>
          ))}

          {view === 'grid' && grid && !grid.cells?.some((c) => c.corrected_mm !== undefined) && (
            <CircleMarker center={center} radius={1} opacity={0} fillOpacity={0} />
          )}
        </MapContainer>

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
