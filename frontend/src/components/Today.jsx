import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api';
import CommandBar from './CommandBar';
import RegimeTimeline from './RegimeTimeline';
import MapPanel from './MapPanel';
import DistrictTable from './DistrictTable';
import DistrictRail from './DistrictRail';
import BulletinModal from './BulletinModal';
import StoryMode from './StoryMode';
import ErrorBoundary from './ErrorBoundary';
import Icon from './Icon';
import { CATEGORY_META, REGIME_COLOR, fmt, pct } from '../lib/format';

const LAYER_FOR_STEP = { 0: 'category', 1: 'corrected', 2: 'adjustment', 3: 'p_heavy', 4: 'regime' };

const SHORTCUTS = [
  ['← →', 'previous / next day'],
  ['1 … 5', 'lead day'],
  ['n / p', 'next / previous significant day'],
  ['l', 'change map layer'],
  ['b', 'bulletin'],
  ['s', 'guided tour'],
  ['Esc', 'close'],
];

/**
 * The console: one screen per date and lead day.
 *
 * Top to bottom: four numbers, the season timeline, then the map with the
 * warned districts beside it. Clicking a district opens its detail in a drawer
 * over the right edge, so the screen underneath never changes shape. The full
 * 53-row table is one click away.
 */
export default function Today({ session, onSession, onToast, onTab }) {
  const { date, lead } = session;
  const [console_, setConsole_] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [timelineMeta, setTimelineMeta] = useState(null);
  const [events, setEvents] = useState([]);
  const [cases, setCases] = useState([]);
  const [geojson, setGeojson] = useState(null);
  const [grid, setGrid] = useState(null);
  const [gridDates, setGridDates] = useState([]);
  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [layer, setLayer] = useState('category');
  const [view, setView] = useState('district');
  const [bulletin, setBulletin] = useState(false);
  const [story, setStory] = useState(false);
  const [help, setHelp] = useState(false);
  const [fullTable, setFullTable] = useState(false);
  const loadId = useRef(0);

  // --- shared payloads, loaded once
  useEffect(() => {
    api.timeline()
      .then((t) => { setTimeline(t.days || []); setTimelineMeta(t.meta || null); })
      .catch(() => setTimeline([]));
    api.events(30).then((e) => setEvents(e.events || [])).catch(() => setEvents([]));
    api.cases().then((c) => setCases(c.cases || [])).catch(() => setCases([]));
    api.geojson().then(setGeojson).catch(() => setGeojson(null));
    api.grid().then((g) => { setGrid(g); setGridDates(g.available_dates?.slice(-40).reverse() || []); })
      .catch(() => setGrid(null));
  }, []);

  // Open on the season's most warning-heavy held-out day, not an arbitrary one.
  useEffect(() => {
    if (session.date) return;
    if (timelineMeta?.default_date) {
      onSession({ date: timelineMeta.default_date, source: 'default_significant_day' });
      return;
    }
    if (events.length) onSession({ date: events[0].date, source: 'events_fallback' });
  }, [timelineMeta, events, session.date, onSession]);

  // --- one request per screen
  const load = useCallback((d, l) => {
    if (!d) return;
    const id = ++loadId.current;
    setBusy(true); setError(null);
    api.console(d, l)
      .then((res) => { if (id === loadId.current) setConsole_(res); })
      .catch((e) => { if (id === loadId.current) setError(String(e.message || e)); })
      .finally(() => { if (id === loadId.current) setBusy(false); });
  }, []);

  useEffect(() => { load(date, lead); }, [date, lead, load]);

  // --- keyboard
  useEffect(() => {
    const cycle = () => {
      setLayer((cur) => {
        const ids = Object.values(LAYER_FOR_STEP);
        return ids[(ids.indexOf(cur) + 1) % ids.length];
      });
    };
    window.addEventListener('monsooniq:cycle-layer', cycle);
    return () => window.removeEventListener('monsooniq:cycle-layer', cycle);
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);
      if (typing && e.key !== 'Escape') return;
      if (e.key === 'Escape') { setSelected(null); setBulletin(false); setStory(false); setHelp(false); return; }
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const nav = console_?.navigation;
        const next = e.key === 'ArrowRight' ? nav?.next_date : nav?.prev_date;
        if (next) onSession({ date: next });
      } else if (/^[1-5]$/.test(e.key)) onSession({ lead: Number(e.key) });
      else if (e.key === 'b') setBulletin(true);
      else if (e.key === 's') setStory(true);
      else if (e.key === '?') setHelp((h) => !h);
      else if (e.key === 'l') window.dispatchEvent(new Event('monsooniq:cycle-layer'));
      else if (e.key === 'n' || e.key === 'p') {
        const i = events.findIndex((x) => x.date === date);
        const j = e.key === 'n' ? Math.min(i + 1, events.length - 1) : Math.max(i - 1, 0);
        const next = events[j] || events[0];
        if (next) onSession({ date: next.date });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [console_, events, date, onSession]);

  const districts = console_?.districts || [];
  const filtered = useMemo(() => {
    if (!filter.trim()) return districts;
    const q = filter.trim().toLowerCase();
    return districts.filter((d) => d.district_name.toLowerCase().includes(q)
      || d.state_name.toLowerCase().includes(q));
  }, [districts, filter]);
  const warned = useMemo(() => filtered.filter((d) => d.category !== 'green'), [filtered]);
  const counts = console_?.summary?.counts;
  const summary = console_?.summary;
  const peak = districts[0];

  return (
    <>
      {busy && <div className="progressbar" role="progressbar" aria-label="loading forecast" />}
      <CommandBar
        date={date} lead={lead} dates={(timeline || []).map((t) => t.date)}
        horizon={console_?.horizon}
        onDate={(d) => onSession({ date: d, source: 'date_picker' })}
        onLead={(l) => onSession({ lead: l })}
        onBulletin={() => setBulletin(true)}
        onExport={() => api.exportCsv(date, lead).catch(() => {})}
        onHelp={() => setHelp((h) => !h)}
      />

      {help && (
        <div className="help-pop panel" role="dialog" aria-label="keyboard shortcuts">
          <div className="panel-head">
            <span className="panel-title">Keyboard</span>
            <button className="btn icon" onClick={() => setHelp(false)} aria-label="close"><Icon name="close" size={14} /></button>
          </div>
          <div className="panel-body">
            <table className="data small"><tbody>
              {SHORTCUTS.map(([k, what]) => (
                <tr key={k}><td><kbd className="mono">{k}</kbd></td><td className="muted">{what}</td></tr>
              ))}
            </tbody></table>
          </div>
        </div>
      )}

      <div className={`workspace${error ? ' stale' : ''}`}>
        {error && (
          <div className="panel err">
            <div className="panel-head">
              <span className="panel-title">Could not load {date} · Day {lead}</span>
              <button className="btn" style={{ marginLeft: 'auto' }} onClick={() => load(date, lead)}>Retry</button>
            </div>
            <div className="panel-body small">
              {console_
                ? <>Showing the last day that loaded, <b>{console_.date} · Day {console_.lead}</b>.</>
                : 'Nothing has loaded yet.'}
              <div className="mono tiny muted" style={{ marginTop: 5 }}>{error}</div>
            </div>
          </div>
        )}

        <div className="summary-strip four">
          <div className="stat">
            <div className="k">Regime</div>
            <div className="v" style={{ fontSize: 18, color: REGIME_COLOR[console_?.regime?.name] }}>
              {console_?.regime?.name || '—'}
            </div>
            <div className="u">{pct(console_?.regime?.confidence, 0)} confidence · {date} · Day {lead}</div>
          </div>
          <div className={`stat${(counts?.red ?? 0) > 0 ? ' hot' : ''}`}>
            <div className="k">Districts warned</div>
            <div className="v">{summary?.districts_in_warning ?? '—'}<span className="u"> of {districts.length || 53}</span></div>
            <div className="u">
              <span style={{ color: 'var(--red)' }}>●</span> {counts?.red ?? 0} red
              {' '}<span style={{ color: 'var(--orange)' }}>●</span> {counts?.orange ?? 0} orange
              {' '}<span style={{ color: 'var(--yellow)' }}>●</span> {counts?.yellow ?? 0} yellow
            </div>
          </div>
          <div className={`stat${(summary?.max_corrected_mm ?? 0) >= 204.5 ? ' hot' : ''}`}>
            <div className="k">Heaviest forecast</div>
            <div className="v">{fmt(summary?.max_corrected_mm)}<span className="u"> mm</span></div>
            <div className="u">{peak ? `${peak.district_name} · ` : ''}raw model said {fmt(summary?.max_raw_mm)} mm</div>
          </div>
          <div className="stat">
            <div className="k">What fell (reference)</div>
            <div className="v" style={{ color: 'var(--muted)' }}>{fmt(summary?.max_observed_mm)}<span className="u"> mm</span></div>
            <div className="u">heaviest district observed · never a model input</div>
          </div>
        </div>

        <RegimeTimeline
          days={timeline} currentDate={date} lead={lead}
          events={events} cases={cases}
          onSelect={(d, l) => onSession({ date: d, lead: l || lead, source: 'timeline' })}
        />

        <div className="split">
          <ErrorBoundary
            label="Map"
            fallback="The map could not start in this browser. Warnings, corrections and probabilities are unaffected — use the list."
          >
            <MapPanel
              geojson={geojson} districts={districts} selectedId={selected} onSelect={setSelected}
              layer={layer} onLayer={setLayer} view={view} onView={setView}
              grid={grid} gridDates={gridDates} gridDate={gridDates[0]} onGridDate={() => {}}
              date={date} lead={lead} provenance={console_?.provenance}
            />
          </ErrorBoundary>

          <div className="panel warned">
            <div className="panel-head">
              <span className="panel-title">Warned districts</span>
              <input type="search" value={filter} placeholder="find a district" aria-label="find a district"
                     onChange={(e) => setFilter(e.target.value)} />
            </div>
            <div className="table-scroll warned-scroll">
              <table className="data">
                <thead>
                  <tr><th>District</th><th>Level</th><th className="num">Forecast</th><th className="num">P ≥ 65 mm</th></tr>
                </thead>
                <tbody>
                  {(warned.length ? warned : filtered.slice(0, 12)).map((d) => (
                    <tr key={d.district_id} onClick={() => setSelected(d.district_id)}
                        aria-selected={d.district_id === selected}>
                      <td><b>{d.district_name}</b><div className="tiny muted">{d.state_name}</div></td>
                      <td><span className={`cat-chip cat-${d.category}`}>{CATEGORY_META[d.category].short}</span></td>
                      <td className="num"><b>{fmt(d.corrected_mm)}</b><div className="tiny muted">raw {fmt(d.raw_mm)}</div></td>
                      <td className="num">{pct(d.p_heavy, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!warned.length && districts.length > 0 && (
                <div className="small muted" style={{ padding: '10px 14px' }}>
                  No district reaches a warning category on this day{filter ? ' for that search' : ''}.
                </div>
              )}
            </div>
            <div className="panel-foot">
              <button className="btn" onClick={() => setFullTable((v) => !v)}>
                {fullTable ? 'Hide' : 'Show'} all {districts.length} districts
              </button>
            </div>
          </div>
        </div>

        {fullTable && (
          <DistrictTable districts={filtered} selectedId={selected} onSelect={setSelected}
                         filter={filter} onFilterChange={setFilter} />
        )}
      </div>

      {selected && (
        <>
          <div className="drawer-backdrop" onClick={() => setSelected(null)} />
          <aside className="drawer" aria-label="district detail">
            <DistrictRail districtId={selected} date={date} lead={lead} onClose={() => setSelected(null)} />
          </aside>
        </>
      )}

      {bulletin && <BulletinModal date={date} lead={lead} onClose={() => setBulletin(false)} />}
      {story && (
        <StoryMode
          onClose={() => setStory(false)}
          onNavigate={(d, l) => onSession({ date: d, lead: l, source: 'story' })}
          currentDate={date}
          onLayer={setLayer}
          onTab={onTab}
        />
      )}
    </>
  );
}
