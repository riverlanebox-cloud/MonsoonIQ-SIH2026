/**
 * Frontend smoke test.
 *
 * Renders the whole application in jsdom against recorded API fixtures and
 * asserts that the console, the verification page and the method page all
 * produce their key elements without throwing. This catches the class of bug
 * that a production build cannot: a component that compiles but crashes on the
 * first payload (undefined field, bad shape, missing guard).
 *
 * Record fixtures first:  PYTHONPATH=. python scripts/dump_api_fixtures.py
 * Then run:               npm run smoke
 */

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';
import { build } from 'vite';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const fixtureDir = join(root, 'fixtures');

if (!existsSync(join(fixtureDir, 'index.json'))) {
  console.error('No fixtures found. Run: PYTHONPATH=. python scripts/dump_api_fixtures.py');
  process.exit(2);
}

const index = JSON.parse(readFileSync(join(fixtureDir, 'index.json'), 'utf8'));
const fixtures = {};
for (const file of readdirSync(fixtureDir)) {
  if (file.endsWith('.json') && file !== 'index.json') {
    fixtures[file.replace('.json', '')] = JSON.parse(readFileSync(join(fixtureDir, file), 'utf8'));
  }
}

// --- bundle the app for node, then provide a browser-ish environment with a fetch stub
const outDir = join(root, '.smoke-build');
await build({
  root,
  logLevel: 'error',
  resolve: {
    // Leaflet needs layout + canvas, which jsdom does not provide.
    alias: { 'react-leaflet': join(root, 'scripts', 'leaflet-stub.jsx') },
  },
  mode: 'development',
  build: {
    ssr: true,
    minify: false,
    outDir,
    emptyOutDir: true,
    rollupOptions: { input: join(root, 'smoke-entry.jsx') },
  },
});

const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
  url: 'https://localhost:3000/',
  pretendToBeVisual: true,
});
const { window } = dom;
global.window = window;
global.document = window.document;
Object.defineProperty(global, 'navigator', { value: window.navigator, configurable: true });
global.HTMLElement = window.HTMLElement;
global.Element = window.Element;
global.Node = window.Node;
global.getComputedStyle = window.getComputedStyle;
global.requestAnimationFrame = (cb) => setTimeout(cb, 0);
global.cancelAnimationFrame = clearTimeout;
window.requestAnimationFrame = global.requestAnimationFrame;

const calls = [];
// Flipped mid-run by the failure-path checks near the end of this file.
let failConsoleEndpoint = false;
global.fetch = async (url) => {
  const path = String(url).replace(/^https?:\/\/[^/]+/, '');
  const clean = path.split('?')[0];
  calls.push(path);
  if (failConsoleEndpoint && clean === '/console') {
    return { ok: false, status: 503, text: async () => 'engine unavailable', json: async () => ({}) };
  }
  const name = index.paths[clean]
    || index.paths[path]
    || (clean.startsWith('/district/') ? 'district' : null)
    || (clean === '/console' ? 'console' : null)
    || (clean === '/bulletin' ? 'bulletin' : null);
  if (!name || !fixtures[name]) {
    return { ok: false, status: 404, text: async () => 'no fixture', json: async () => ({}) };
  }
  return { ok: true, status: 200, json: async () => fixtures[name], text: async () => '{}' };
};

const failures = [];
const origError = console.error;
console.error = (...args) => { failures.push(args.map(String).join(' ')); origError(...args); };

const { mount } = await import(join(outDir, 'smoke-entry.js'));
await mount(window.document.getElementById('root'));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
await wait(1200);

const html = () => window.document.getElementById('root').innerHTML;
const checks = [];
// Always re-query: React replaces nodes on re-render, so a captured element can go stale.
const buttonByText = (text, selector = 'button') =>
  [...window.document.querySelectorAll(selector)].find((b) => b.textContent.includes(text));
const clickText = (text, selector = 'button') => {
  const b = buttonByText(text, selector);
  b?.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  return !!b;
};
const enterConsole = async () => {
  clickText('Enter operations console');
  await wait(900);
};
const check = (label, condition) => checks.push({ label, ok: !!condition });

// ---------------------------------------------------------------- landing screen
check('landing title + kicker', html().includes('MonsoonIQ') && html().includes('SIH26080'));
check('landing CTAs', html().includes('Enter operations console') && html().includes('Verification'));
check('landing live stat strip', html().includes('Districts warned') && html().includes('Peak corrected'));
check('landing provenance disclaimer', html().toLowerCase().includes('synthetic')
  && html().includes('not an official'.replace(' an ', ' ')) === false || html().includes('official IMD'));
check('landing shortcut hints', html().includes('keyboard-first'));

// ---------------------------------------------------------------- enter the console
check('enter console button present', !!buttonByText('Enter operations console', 'button.cta'));
await enterConsole();

check('masthead + provenance chip', html().includes('MonsoonIQ') && html().includes('Provenance'));
check('command bar with lead selector', html().includes('Lead') && html().includes('D5'));
check('warning summary strip', html().includes('Districts warned'));
check('regime timeline rendered', html().includes('Regime and warning timeline'));
check('map legend present', html().includes('map-legend') || html().includes('Warning category'));
check('district table rows', html().includes('Warning table'));

// The console repeats one backend rule client-side (which criterion set a warning
// category). If that mirror drifts, every category label in the UI becomes a lie,
// so it is checked against the API's own labels for the whole fixture date.
const fixtureRows = fixtures.console?.districts || [];
const basisMismatches = fixtureRows.filter(
  (d) => window.__format?.categoryBasis(d)?.category !== d.category);
check('category basis mirrors the API for every district',
  fixtureRows.length > 0 && basisMismatches.length === 0);
if (basisMismatches.length) console.log('  disagreements:', basisMismatches.slice(0, 4).map((d) => d.district_id));
check('every warning names a criterion', fixtureRows.filter((d) => d.category !== 'green')
  .every((d) => window.__format?.categoryBasis(d)?.source !== 'none'));
check('rail placeholder or detail', html().includes('District detail'));

// Open a district: the rail must explain the category it is showing, which for the
// fixture date is the worked example of a probability-driven warning.
window.document.querySelector('table.data tbody tr')
  ?.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
await wait(700);
check('rail names the criterion behind the category', html().includes('≥ 65%'));
check('probability-driven warning is flagged as such',
  html().includes('issued on exceedance probability'));
check('band is annotated with its parts', html().includes('P10\u2013P90') && html().includes('IMD warning thresholds'));
check('console fetched once per screen', calls.filter((c) => c.startsWith('/console')).length >= 1);

// --- tab navigation
const clickTab = (label) => clickText(label, 'button.tab');

check('overview button returns to landing', clickText('Overview'));
await wait(400);
check('returned to landing', html().includes('Enter operations console'));
await enterConsole();

check('switched to Skill lab', clickTab('Skill lab'));
await wait(500);
check('skill lab honesty note', html().includes('Read this first'));
check('skill lab claim table', html().includes('Claims and their verdicts'));
check('skill lab categorical table', html().includes('Raw NWP'));

// Documented cases: one click from the map to a named event.
check('switched back to Today', clickTab('Today'));
await wait(1100);
check('case strip lists documented events', html().includes('Kerala orographic surge'));
check('case click loads that day', (() => {
  const b = buttonByText('Kerala orographic surge');
  b?.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  return !!b;
})());
await wait(900);
check('case date applied to the console', calls.some((c) => c.startsWith('/console?date=2018-08-14')));

check('switched to Method', clickTab('Method'));
await wait(400);
check('method architecture section', html().includes('System architecture and module communication'));
check('method user flow section', html().includes('Operational user flow'));
check('method model card', html().includes('Model card') || html().includes('Stated limits'));

// --- keyboard navigation back on the console
clickTab('Today');
await wait(300);
const before = calls.filter((c) => c.startsWith('/console')).length;
window.dispatchEvent(new window.KeyboardEvent('keydown', { key: '2', bubbles: true }));
await wait(400);
check('lead shortcut triggers a reload', calls.filter((c) => c.startsWith('/console')).length > before);
// Stylesheets are not applied in a jsdom SSR render, so the theme is asserted at
// the source level: tokens defined, and no light-theme surfaces left behind.
const css = readFileSync(join(root, 'src', 'index.css'), 'utf8');
check('dark theme tokens defined', ['--bg-0', '--cyan', '--panel', '--red'].every((t) => css.includes(`${t}:`)));
check('light theme fully retired', !css.includes('#eceff3') && !css.includes('linear-gradient(180deg, #fff'));
check('IMD signal colours present', ['--green', '--yellow', '--orange', '--red'].every((t) => css.includes(`var(${t})`) || css.includes(`${t}:`)));
// --- API reference
check('switched to API reference', clickTab('API'));
await wait(400);
check('api view lists endpoints', html().includes('/verification/summary') && html().includes('GET'));
check('api view degrades without the spec', html().includes('static list') || html().includes('OpenAPI'));

// The probe hits a route the fixture set does not serve, so this asserts that the
// request path runs and that its outcome renders — not that it returns 200.
const probeBtn = buttonByText('Send');
check('api probe button present', !!probeBtn);
probeBtn?.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
await wait(400);
check('api probe reports an outcome', /\d+ ms · [\d.]+ kB|network error/.test(html()));

// --- failure path: an unreachable engine must say so, keep the last good load
//     visible, mark it stale, and recover on retry
clickTab('Today');
await wait(1100);
failConsoleEndpoint = true;
window.dispatchEvent(new window.KeyboardEvent('keydown', { key: '3', bubbles: true }));
await wait(600);
check('failure banner names the date', html().includes('Could not load'));
check('failure banner offers a retry', !!buttonByText('Retry'));
check('stale load is marked', html().includes('stale') && html().includes('treat it as stale'));
failConsoleEndpoint = false;
buttonByText('Retry')?.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
await wait(900);
check('retry clears the failure', !html().includes('Could not load'));

window.dispatchEvent(new window.KeyboardEvent('keydown', { key: '?', bubbles: true }));
await wait(100);
check('shortcut help toast', html().includes('Shortcuts:'));

const realErrors = failures.filter((f) => !/Leaflet|jest|act\(|Warning: ReactDOM/.test(f));
const failed = checks.filter((c) => !c.ok);

console.log('\nFrontend smoke test');
for (const c of checks) console.log(`  ${c.ok ? 'PASS' : 'FAIL'}  ${c.label}`);
if (realErrors.length) {
  console.log('\nConsole errors:');
  realErrors.slice(0, 8).forEach((e) => console.log(`  ! ${e.slice(0, 200)}`));
}
if (process.env.SMOKE_DEBUG) { console.log('\nAPI calls:', calls); console.log('\nHTML head:', html().slice(0, 1200)); }
console.log(`\n${checks.length - failed.length}/${checks.length} checks passed, ${calls.length} API calls made`);

process.exit(failed.length || realErrors.length ? 1 : 0);
