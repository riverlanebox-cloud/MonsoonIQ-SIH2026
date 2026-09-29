/**
 * API client.
 *
 * The console is built around two rhythms: one call per screen
 * (`fetchConsole`) and one small set of shared payloads loaded once
 * (timeline, events, verification). Everything else is on demand.
 *
 * Two back ends, same payloads:
 *  - live: the FastAPI service (dev proxy at /api, or same origin in production);
 *  - static snapshot (VITE_STATIC=1, e.g. on Vercel): every response the console
 *    can ask for, precomputed at build time by scripts/export_static.py and served
 *    as files under /snapshot. See docs/DEPLOY.md.
 */

export const STATIC = import.meta.env.VITE_STATIC === '1';

// In dev the Vite server proxies /api to the FastAPI service; a production build is
// served by FastAPI itself, so the API lives at the same origin and the prefix is empty.
const BASE = STATIC
  ? '/snapshot'
  : (import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? '/api' : ''));

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

// ------------------------------------------------------------ static snapshot
const memo = new Map();
const once = (key, fn) => {
  if (!memo.has(key)) memo.set(key, fn().catch((e) => { memo.delete(key); throw e; }));
  return memo.get(key);
};

async function fetchFile(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${path} — not in the static snapshot`);
  return res;
}

const staticJson = (path) => once(path, () => fetchFile(path).then((r) => r.json()));

/** One gzip file per day holds every per-day payload. Some hosts add
 *  Content-Encoding and the browser has already inflated it; check the magic bytes. */
async function inflateJson(res) {
  const buf = new Uint8Array(await res.arrayBuffer());
  if (buf[0] === 0x1f && buf[1] === 0x8b && typeof DecompressionStream !== 'undefined') {
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(stream).text());
  }
  return JSON.parse(new TextDecoder().decode(buf));
}

const snapshotIndex = () => staticJson('/index.json');

/** Same rule as the live API: an unknown date resolves to the nearest archived day. */
async function resolveDate(date) {
  const idx = await snapshotIndex();
  if (!date) return idx.default_date;
  if (idx.dates.includes(date)) return date;
  const t = Date.parse(date);
  if (Number.isNaN(t)) return idx.default_date;
  let best = idx.dates[0];
  for (const d of idx.dates) if (Math.abs(Date.parse(d) - t) < Math.abs(Date.parse(best) - t)) best = d;
  return best;
}

async function staticDay(date, lead) {
  const d = await resolveDate(date);
  const day = await once(`day:${d}`, () => fetchFile(`/day/${d}.json.gz`).then(inflateJson));
  const entry = day.leads[String(lead || 1)];
  if (!entry) throw new Error(`404 lead ${lead} not in the static snapshot`);
  return entry;
}

const need = (value, what) => {
  if (value == null) throw new Error(`404 ${what} not in the static snapshot`);
  return value;
};

function saveText(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: filename });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const staticApi = {
  health: () => staticJson('/health.json'),
  modelCard: () => staticJson('/model-card.json'),
  timeline: () => staticJson('/timeline.json'),
  events: async (limit = 25) => {
    const e = await staticJson('/events.json');
    return Array.isArray(e?.events) ? { ...e, events: e.events.slice(0, limit) } : e;
  },
  console: async (date, lead = 1) => need((await staticDay(date, lead)).console, 'console'),
  district: async (id, date, lead = 1) =>
    need((await staticDay(date, lead)).district?.[id], `district ${id}`),
  bulletin: async (date, lead = 1, lang = 'en') =>
    need((await staticDay(date, lead)).bulletin?.[lang], 'bulletin'),
  grid: () => staticJson('/grid.json'),
  verification: () => staticJson('/verification/summary.json'),
  heavyEvents: () => staticJson('/verification/heavy-events.json'),
  regimeValue: () => staticJson('/verification/regime-value.json'),
  cases: () => staticJson('/case-replays.json'),
  districts: () => staticJson('/districts.json'),
  geojson: () => staticJson('/districts/geojson.json'),
  explain: () => Promise.reject(new Error('Feature explanations need the live API')),
  exportCsv: async (date, lead = 1) => {
    const d = await resolveDate(date);
    const csv = need((await staticDay(d, lead)).csv, 'CSV export');
    saveText(csv, `monsooniq_${d}_d${lead}.csv`, 'text/csv');
  },
  reportPdfUrl: () => `${BASE}/verification/report.pdf`,
};

// ------------------------------------------------------------------ live API
const liveApi = {
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
  exportCsv: (date, lead) => {
    window.open(`${BASE}/export/districts.csv?date=${date}&lead=${lead}`, '_blank');
    return Promise.resolve();
  },
  reportPdfUrl: () => `${BASE}/verification/report.pdf`,
};

/** The API prefix this build talks to (empty when FastAPI serves the bundle). */
export const apiBase = () => BASE;

export const api = STATIC ? staticApi : liveApi;
