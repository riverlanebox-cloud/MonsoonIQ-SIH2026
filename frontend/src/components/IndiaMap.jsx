import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WIDTH, HEIGHT, geometryPath, featureAnchor, project } from '../lib/geo';
import { CATEGORY_META, layerFill, fmt, pct, rainColor } from '../lib/format';

/**
 * Offline SVG map of India.
 *
 * Base layer: all 641 Census-2011 districts (DataMeet). Product layer: the forecast
 * districts, filled by the active layer. No tiles, no network: it renders the same on a
 * projector with no internet as on a laptop. Wheel to zoom, drag to pan, double-click to reset.
 */
export default function IndiaMap({
  baseGeo, studyGeo, districts = [], layer = 'category', selectedId, onSelect,
  highlight, cells, interactive = true, showPulses = true, className = '', style, onHover,
}) {
  const svgRef = useRef(null);
  const [vb, setVb] = useState([0, 0, WIDTH, HEIGHT]);
  const [hover, setHover] = useState(null);
  const drag = useRef(null);

  const byId = useMemo(() => {
    const m = {};
    for (const d of districts) m[d.district_id] = d;
    return m;
  }, [districts]);

  const basePaths = useMemo(
    () => (baseGeo?.features || []).map((f) => ({ id: f.properties.district_id, d: geometryPath(f.geometry) })),
    [baseGeo],
  );
  const studyPaths = useMemo(
    () => (studyGeo?.features || []).map((f) => ({
      id: f.properties.district_id, d: geometryPath(f.geometry), anchor: featureAnchor(f),
    })),
    [studyGeo],
  );
  const hl = useMemo(() => new Set(highlight || []), [highlight]);

  const onWheel = useCallback((e) => {
    if (!interactive) return;
    e.preventDefault();
    const svg = svgRef.current;
    const r = svg.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width;
    const fy = (e.clientY - r.top) / r.height;
    setVb(([x, y, w, h]) => {
      const s = e.deltaY > 0 ? 1.15 : 1 / 1.15;
      const nw = Math.min(WIDTH * 1.2, Math.max(WIDTH / 12, w * s));
      const nh = nw * (h / w);
      return [x + (w - nw) * fx, y + (h - nh) * fy, nw, nh];
    });
  }, [interactive]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return undefined;
    svg.addEventListener('wheel', onWheel, { passive: false });
    return () => svg.removeEventListener('wheel', onWheel);
  }, [onWheel]);

  const onDown = (e) => {
    if (!interactive) return;
    drag.current = { x: e.clientX, y: e.clientY, vb, moved: false };
  };
  const onMove = (e) => {
    const st = drag.current;
    if (!st) return;
    const r = svgRef.current.getBoundingClientRect();
    const dx = ((e.clientX - st.x) / r.width) * st.vb[2];
    const dy = ((e.clientY - st.y) / r.height) * st.vb[3];
    if (Math.abs(dx) + Math.abs(dy) > 2) st.moved = true;
    setVb([st.vb[0] - dx, st.vb[1] - dy, st.vb[2], st.vb[3]]);
  };
  const onUp = () => { setTimeout(() => { drag.current = null; }, 0); };
  const click = (id) => { if (!drag.current?.moved && onSelect) onSelect(id); };

  const hovered = hover ? byId[hover.id] : null;

  return (
    <div className={`india-map ${className}`} style={style}>
      <svg
        ref={svgRef}
        viewBox={vb.join(' ')}
        preserveAspectRatio="xMidYMid meet"
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={() => { drag.current = null; setHover(null); }}
        onDoubleClick={() => setVb([0, 0, WIDTH, HEIGHT])}
        role="img"
        aria-label="District map of India"
      >
        <defs>
          <radialGradient id="pulse-red">
            <stop offset="0%" stopColor="#ff3b30" stopOpacity=".9" />
            <stop offset="100%" stopColor="#ff3b30" stopOpacity="0" />
          </radialGradient>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.2" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        <g className="base-layer">
          {basePaths.map((p) => <path key={p.id} d={p.d} />)}
        </g>

        {cells?.length > 0 && (
          <g className="cell-layer">
            {cells.map((c, i) => {
              const [x0, y0] = project(c.lon - 0.25, c.lat + 0.25);
              const [x1, y1] = project(c.lon + 0.25, c.lat - 0.25);
              const v = layer === 'raw' ? c.raw_mm : (c.corrected_mm ?? c.observed_mm);
              return <rect key={i} x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill={rainColor(v)} opacity=".85" />;
            })}
          </g>
        )}

        <g className="study-layer">
          {studyPaths.map((p) => {
            const d = byId[p.id];
            const fill = layerFill(d, layer) || '#16324a';
            const sel = p.id === selectedId;
            const lit = hl.size > 0 && hl.has(p.id);
            const dim = hl.size > 0 && !lit;
            return (
              <path
                key={p.id}
                d={p.d}
                fill={fill}
                fillOpacity={dim ? 0.25 : (d?.category === 'green' && layer === 'category' ? 0.75 : 0.95)}
                className={`${sel ? 'sel' : ''} ${lit ? 'lit' : ''}`}
                onMouseEnter={() => { setHover(p); onHover?.(d || null); }}
                onMouseLeave={() => { setHover(null); onHover?.(null); }}
                onClick={() => click(p.id)}
              />
            );
          })}
        </g>

        {showPulses && (
          <g className="pulse-layer" pointerEvents="none">
            {studyPaths.filter((p) => ['red', 'orange'].includes(byId[p.id]?.category)).map((p) => (
              <g key={p.id} transform={`translate(${p.anchor[0]},${p.anchor[1]})`}>
                <circle r="3" className={`pulse-core ${byId[p.id].category}`} />
                <circle r="3" className={`pulse-ring ${byId[p.id].category}`} />
              </g>
            ))}
          </g>
        )}
      </svg>

      {hovered && (
        <div className="map-tip">
          <div className="map-tip-title">{hovered.district_name}</div>
          <div className="tiny muted">{hovered.state_name} · {hovered.regime}</div>
          <div className="map-tip-grid">
            <span>Corrected</span><b>{fmt(hovered.corrected_mm)} mm</b>
            <span>Raw NWP</span><b>{fmt(hovered.raw_mm)} mm</b>
            <span>P(≥64.5)</span><b>{pct(hovered.p_heavy)}</b>
          </div>
          <span className={`cat-chip cat-${hovered.category}`}>{CATEGORY_META[hovered.category]?.short}</span>
        </div>
      )}
    </div>
  );
}
