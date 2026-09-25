import React, { useMemo } from 'react';
import { DOT_GRID } from '../lib/indiaDots';
import { CATEGORY_FILL } from '../lib/format';

/**
 * SAGAR-style dot-matrix map of India, coloured by live warnings.
 *
 * The lattice comes from a pre-computed India-point-of-view outline (see
 * lib/indiaDots.js). Every India dot that falls inside a modelled district takes
 * that district's current IMD category colour, so the backdrop is still the day's
 * warning picture rather than decoration. Neighbouring land is drawn very dim for
 * context; dots outside modelled districts stay a neutral white.
 */

function ringContains(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function polygonsOf(geometry) {
  if (!geometry) return [];
  return geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
}

export default function DottedIndia({ geojson, byId, className = '' }) {
  const { west, north, step, rows } = DOT_GRID;

  // District polygons with bounding boxes, so each dot only tests a handful.
  const districts = useMemo(() => {
    if (!geojson?.features) return [];
    return geojson.features.map((f) => {
      const polys = polygonsOf(f.geometry);
      let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
      for (const poly of polys) {
        for (const [x, y] of poly[0]) {
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
      }
      return { id: f.properties.district_id, polys, box: [minX, maxX, minY, maxY] };
    });
  }, [geojson]);

  const dots = useMemo(() => {
    const out = [];
    rows.forEach((row, j) => {
      const offset = j % 2 ? 0.5 : 0;
      for (let i = 0; i < row.length; i += 1) {
        const c = row[i];
        if (c === '.') continue;
        const lon = west + (i + offset) * step;
        const lat = north - j * step;
        let district = null;
        if (c === 'i') {
          for (const d of districts) {
            const [x0, x1, y0, y1] = d.box;
            if (lon < x0 || lon > x1 || lat < y0 || lat > y1) continue;
            if (d.polys.some((p) => ringContains(lon, lat, p[0]))) { district = d.id; break; }
          }
        }
        out.push({ x: i + offset, y: j, kind: c, district });
      }
    });
    return out;
  }, [rows, west, north, step, districts]);

  const width = rows[0].length;
  const height = rows.length;

  return (
    <svg
      className={`dotted-india ${className}`}
      viewBox={`-1 -1 ${width + 1} ${height + 1}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Dot map of India coloured by today's district warning category"
    >
      {dots.map((d) => {
        const rec = d.district ? byId?.[d.district] : null;
        const cat = rec?.category;
        const warned = cat && cat !== 'green';
        const fill = d.kind === 'r'
          ? 'rgba(255,255,255,.08)'
          : cat ? CATEGORY_FILL[cat] : 'rgba(255,255,255,.34)';
        return (
          <circle
            key={`${d.x}-${d.y}`}
            cx={d.x} cy={d.y}
            r={warned ? 0.36 : 0.26}
            fill={fill}
            opacity={cat === 'green' ? 0.55 : 1}
            className={cat === 'red' ? 'dot-hot' : undefined}
          />
        );
      })}
    </svg>
  );
}
