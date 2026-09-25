/** Shared formatting, the IMD warning scale, and the dark-console colour maps. */

export const CATEGORIES = ['green', 'yellow', 'orange', 'red'];

export const CATEGORY_META = {
  green: { label: 'No warning', short: 'GREEN', action: 'Routine', mm: '< 64.5' },
  yellow: { label: 'Watch', short: 'YELLOW', action: 'Be updated', mm: '64.5 – 115.5' },
  orange: { label: 'Alert', short: 'ORANGE', action: 'Be prepared', mm: '115.6 – 204.4' },
  red: { label: 'Warning', short: 'RED', action: 'Take action', mm: '≥ 204.5' },
};

/** Signal colours (text, borders, strokes). */
export const CATEGORY_COLOR = {
  green: '#00e07a',
  yellow: '#ffd700',
  orange: '#ff6b35',
  red: '#ff3b30',
};

/** Area fills for maps and chips — darker so they sit under the labels. */
export const CATEGORY_FILL = {
  green: '#17734b',
  yellow: '#c39c0e',
  orange: '#d15c1d',
  red: '#d02f26',
};

export const CATEGORY_FILL_SOFT = {
  green: 'rgba(0, 224, 122, .14)',
  yellow: 'rgba(255, 215, 0, .14)',
  orange: 'rgba(255, 107, 53, .16)',
  red: 'rgba(255, 59, 48, .18)',
};

/**
 * Rainfall ramp for the dark basemap: deep blue through cyan to the standard
 * radar yellow/orange/red. Sequential, colour-blind safe at the top end, and it
 * reads on a black basemap without washing out the warning map.
 */
export const RAIN_SCALE = [
  [0.5, '#16324a'],
  [2, '#1b4a70'],
  [10, '#22679a'],
  [25, '#2e8fbe'],
  [50, '#43b9d4'],
  [75, '#ffd700'],
  [115, '#ff6b35'],
  [204, '#ff3b30'],
];

export const RAIN_LEGEND = [
  // The ramp is sampled at the IMD warning thresholds so that, above 64 mm, the
  // rainfall fill and the warning category agree on the same colour.
  ['< 1', 0.5], ['1–2', 2], ['2–10', 10], ['10–25', 25],
  ['25–64', 50], ['64–115', 75], ['115–204', 115], ['≥ 204', 204],
];

export function rainColor(mm) {
  if (mm === null || mm === undefined || Number.isNaN(mm)) return '#122838';
  let color = RAIN_SCALE[0][1];
  for (const [threshold, c] of RAIN_SCALE) if (mm >= threshold) color = c;
  return color;
}

/** Correction map: green = forecast made wetter, red = made drier. */
export function adjustmentColor(mm) {
  if (mm === null || mm === undefined || Number.isNaN(mm)) return '#122838';
  if (mm > 40) return '#00e07a';
  if (mm > 15) return '#12a862';
  if (mm > 4) return '#0d5c3c';
  if (mm > 0) return '#123a30';
  if (mm < -40) return '#ff3b30';
  if (mm < -15) return '#c02a22';
  if (mm < -4) return '#6d1a16';
  if (mm < 0) return '#3a1d1c';
  return '#122838';
}

export function probabilityColor(p) {
  if (p === null || p === undefined) return '#122838';
  if (p >= 0.75) return '#ff3b30';
  if (p >= 0.5) return '#ff6b35';
  if (p >= 0.3) return '#ffd700';
  if (p >= 0.15) return '#4f7a12';
  return '#16324a';
}

export const REGIME_COLOR = {
  'Active Monsoon': '#3aa0ff',
  'Break Monsoon': '#ffb74d',
  'Monsoon Low/Depression': '#b06bff',
  Orographic: '#00e07a',
  Coastal: '#20b2aa',
  'Western Disturbance': '#b09a7a',
  'Weak/Normal': '#7c8fa3',
};

export const REGIME_SHORT = {
  'Active Monsoon': 'ACT',
  'Break Monsoon': 'BRK',
  'Monsoon Low/Depression': 'DEP',
  Orographic: 'ORO',
  Coastal: 'CST',
  'Western Disturbance': 'WD',
  'Weak/Normal': 'WKN',
};

export function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

export function pct(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function signed(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const v = Number(value);
  return `${v > 0 ? '+' : ''}${v.toFixed(digits)}`;
}

export function reliabilityClass(level) {
  if (level === 'reliable') return 'reliability-ok';
  if (level === 'indicative') return 'reliability-warn';
  return 'suppressed';
}

export function longDate(iso) {
  if (!iso) return '—';
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-IN', {
    weekday: 'short', day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC',
  });
}

export function shortDate(iso) {
  if (!iso) return '—';
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-IN', {
    day: '2-digit', month: 'short', timeZone: 'UTC',
  });
}

/** IMD thresholds the warning categories and the probability heads are keyed to. */
export const IMD_THRESHOLDS = [
  [64.5, 'heavy'], [115.6, 'very heavy'], [204.5, 'extremely heavy'],
];

/**
 * Which criterion put a district in its category.
 *
 * Mirrors AdvisoryGenerator.determine_alert_level in src/api/advisory.py, clause
 * for clause and in the same order — the console shows the rule that fired so a
 * warning can be audited instead of taken on trust, and so a warning driven by
 * exceedance probability is distinguishable at a glance from one driven by the
 * predicted amount. `scripts/smoke.mjs` fails if this drifts from the API.
 */
export function categoryBasis(d) {
  if (!d) return null;
  const maxRain = Math.max(d.p90_mm ?? d.p90 ?? 0, d.corrected_mm ?? d.corrected ?? 0);
  const ph = d.p_heavy ?? 0;
  const pvh = d.p_very_heavy ?? 0;
  const mean = d.corrected_mm ?? d.corrected ?? 0;
  const mm = (v) => `${fmt(v)} mm`;

  const at = (category, source, text) => ({ category, source, text });
  if (maxRain >= 204.5) return at('red', 'amount', `predicted maximum ${mm(maxRain)} ≥ 204.5 mm`);
  if (pvh >= 0.65) return at('red', 'probability', `P(≥115.6 mm) = ${pct(pvh, 0)} ≥ 65%`);
  if (maxRain >= 115.6) return at('orange', 'amount', `predicted maximum ${mm(maxRain)} ≥ 115.6 mm`);
  if (pvh >= 0.35) return at('orange', 'probability', `P(≥115.6 mm) = ${pct(pvh, 0)} ≥ 35%`);
  if (ph >= 0.70) return at('orange', 'probability', `P(≥64.5 mm) = ${pct(ph, 0)} ≥ 70%`);
  if (maxRain >= 64.5) return at('yellow', 'amount', `predicted maximum ${mm(maxRain)} ≥ 64.5 mm`);
  if (ph >= 0.30) return at('yellow', 'probability', `P(≥64.5 mm) = ${pct(ph, 0)} ≥ 30%`);
  if (mean >= 35) return at('yellow', 'amount', `district mean ${mm(mean)} ≥ 35 mm`);
  return at('green', 'none', 'no warning criterion met');
}
