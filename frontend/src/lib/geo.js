/**
 * Offline map geometry: a fixed equirectangular projection tuned for India
 * (x scaled by cos 22°), so maps render with no tile server - the console works on a
 * venue network, an air-gapped laptop, or a projector with no internet.
 */

export const VIEW = { lon0: 67.5, lon1: 98.0, lat0: 6.0, lat1: 37.6 };
const K = 20; // px per degree of latitude
const COS = Math.cos((22 * Math.PI) / 180);

export const WIDTH = (VIEW.lon1 - VIEW.lon0) * K * COS;
export const HEIGHT = (VIEW.lat1 - VIEW.lat0) * K;

export function project(lon, lat) {
  return [(lon - VIEW.lon0) * K * COS, (VIEW.lat1 - lat) * K];
}

function ringPath(ring) {
  let d = '';
  for (let i = 0; i < ring.length; i += 1) {
    const [x, y] = project(ring[i][0], ring[i][1]);
    d += `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
  }
  return `${d}Z`;
}

export function geometryPath(geom) {
  if (!geom) return '';
  const polys = geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates;
  return polys.map((rings) => rings.map(ringPath).join('')).join('');
}

/** Centroid of the projected bounding box - good enough for labels and pulses. */
export function featureAnchor(feature) {
  const p = feature.properties || {};
  if (p.rep_lon && p.rep_lat) return project(p.rep_lon, p.rep_lat);
  if (p.centroid_lon && p.centroid_lat) return project(p.centroid_lon, p.centroid_lat);
  return [WIDTH / 2, HEIGHT / 2];
}
