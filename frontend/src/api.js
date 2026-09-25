/**
 * API client.
 *
 * The console is built around two rhythms: one call per screen
 * (`fetchConsole`) and one small set of shared payloads loaded once
 * (timeline, events, verification). Everything else is on demand.
 */

// In dev the Vite server proxies /api to the FastAPI service; a production build is
// served by FastAPI itself, so the API lives at the same origin and the prefix is empty.
const BASE = import.meta.env.VITE_API_BASE
  ?? (import.meta.env.DEV ? '/api' : '');

const cache = new Map();

async function get(path, { params, useCache = false } = {}) {
  const qs = params
    ? '?' + new URLSearchParams(
        Object.entries(params).filter(([, v]) => v !== undefined && v !== null),
      ).toString()
    : '';
  const url = `${BASE}${path}${qs}`;
  if (useCache && cache.has(url)) return cache.get(url);
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`${res.status} ${path}${detail ? ` — ${detail.slice(0, 160)}` : ''}`);
  }
  const data = await res.json();
  if (useCache) cache.set(url, data);
  return data;
}

/** The API prefix this build talks to (empty when FastAPI serves the bundle). */
export const apiBase = () => BASE;

export const api = {
  health: () => get('/health', { useCache: true }),
  modelCard: () => get('/model-card', { useCache: true }),
  timeline: () => get('/timeline', { useCache: true }),
  events: (limit = 25) => get('/events', { params: { limit }, useCache: true }),
  console: (date, lead, horizon = true) =>
    get('/console', { params: { date, lead, horizon } }),
  district: (id, date, lead) => get(`/district/${id}`, { params: { date, lead } }),
  bulletin: (date, lead, lang = 'en') => get('/bulletin', { params: { date, lead, lang } }),
  grid: (date) => get('/grid', { params: { date } }),
  verification: () => get('/verification/summary', { useCache: true }),
  heavyEvents: () => get('/verification/heavy-events', { useCache: true }),
  regimeValue: () => get('/verification/regime-value', { useCache: true }),
  cases: () => get('/case-replays', { useCache: true }),
  districts: () => get('/districts', { useCache: true }),
  geojson: () => get('/districts/geojson', { useCache: true }),
  explain: (id, date) => get(`/explain/${id}/${date}`),
  exportCsvUrl: (date, lead) => `${BASE}/export/districts.csv?date=${date}&lead=${lead}`,
  reportPdfUrl: () => `${BASE}/verification/report.pdf`,
};
