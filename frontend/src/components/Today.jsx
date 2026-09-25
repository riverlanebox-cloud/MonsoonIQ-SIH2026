import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api';
import CommandBar from './CommandBar';
import RegimeTimeline from './RegimeTimeline';
import MapPanel from './MapPanel';
import CaseStrip from './CaseStrip';
import DistrictTable from './DistrictTable';
import DistrictRail from './DistrictRail';
import BulletinModal from './BulletinModal';
import StoryMode from './StoryMode';
import ErrorBoundary from './ErrorBoundary';
import { CATEGORY_META, REGIME_COLOR, fmt, pct, signed } from '../lib/format';

const LAYER_FOR_STEP = { 0: 'category', 1: 'corrected', 2: 'adjustment', 3: 'p_heavy', 4: 'regime' };

export default function Today({ session, onSession, onToast, onTab }) {
  const { date, lead } = session;
  const [console_, setConsole_] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [timelineMeta, setTimelineMeta] = useState(null);
  const [events, setEvents] = useState([]);
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
  const loadId = useRef(0);

  // --- shared payloads, loaded once
  useEffect(() => {
    api.timeline()
      .then((t) => { setTimeline(t.days || []); setTimelineMeta(t.meta || null); })
      .catch(() => setTimeline([]));
    api.events(30).then((e) => setEvents(e.events || [])).catch(() => setEvents([]));
    api.geojson().then(setGeojson).catch(() => setGeojson(null));
    api.grid().then((g) => { setGrid(g); setGridDates(g.available_dates?.slice(-40).reverse() || []); })
      .catch(() => setGrid(null));
  }, []);

  // --- Open on a significant day rather than an arbitrary one, so the console never
  //     greets a duty officer with a quiet Tuesday. The opening day is computed once,
  //     when the timeline artifacts are built (see src/console/artifacts.py).
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

  // --- keyboard: everything reachable without the mouse
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
      if (e.key === 'Escape') { setSelected(null); setBulletin(false); setStory(false); return; }
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const nav = console_?.navigation;
        const next = e.key === 'ArrowRight' ? nav?.next_date : nav?.prev_date;
        if (next) onSession({ date: next });
      } else if (/^[1-5]$/.test(e.key)) onSession({ lead: Number(e.key) });
      else if (e.key === 'b') setBulletin(true);
      else if (e.key === 's') setStory(true);
      else if (e.key === 'n') {
        const i = events.findIndex((x) => x.date === date);
        const next = events[Math.min(i + 1, events.length - 1)] || events[0];
        if (next) onSession({ date: next.date });
      } else if (e.key === 'p') {
        const i = events.findIndex((x) => x.date === date);
        const prev = events[Math.max(i - 1, 0)] || events[0];
        if (prev) onSession({ date: prev.date });
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

  const topDistricts = useMemo(() => districts.slice(0, 6), [districts]);
  const counts = console_?.summary?.counts;

  return (
    <>
      {busy && <div className="progressbar" role="progressbar" aria-label="loading forecast" />}
      <CommandBar
        date={date}
        lead={lead}
        onDate={(d) => onSession({ date: d, source: 'date_picker' })}
        onLead={(l) => onSession({ lead: l })}
        counts={counts}
        events={events}
        onJumpEvent={(d) => onSession({ date: d, source: 'significant_day' })}
        query={filter}
        onQuery={setFilter}
        onExport={() => window.open(api.exportCsvUrl(date, lead), '_blank')}
        onBulletin={() => setBulletin(true)}
        onRefresh={() => load(date, lead)}
        dates={(timeline || []).map((t) => t.date)}
        busy={busy}
        modelScope={console_?.model_scope}
      />

      <div className={`workspace${error ? ' stale' : ''}`}>
        {error && (
          <div className="panel err">
            <div className="panel-head">
              <span className="panel-title">Could not load {date} · Day {lead}</span>
              <button className="btn" style={{ marginLeft: 'auto' }} onClick={() => load(date, lead)}>
                Retry
              </button>
            </div>
            <div className="panel-body small">
              The engine did not answer for this date.{' '}
              {console_
                ? <>Everything below is the last successful load, <b>{console_.date} · Day {console_.lead}</b> —
                    treat it as stale until this reloads.</>
                : 'Nothing has loaded yet, so there is nothing below to misread.'}
              <div className="mono tiny muted" style={{ marginTop: 5 }}>{error}</div>
            </div>
          </div>
        )}

        <div className="panel">
          <div className="summary-strip">
            <div className="stat">
              <div className="k">Valid for</div>
              <div className="v" style={{ fontSize: 15 }}>{date}</div>
              <div className="u">Day {lead} · issued {console_?.issued_utc || '—'}</div>
            </div>
            <div className="stat">
              <div className="k">Regime detected</div>
              <div className="v" style={{ fontSize: 15, color: REGIME_COLOR[console_?.regime?.name] }}>
                {console_?.regime?.name || '—'}
              </div>
              <div className="u">classifier confidence {pct(console_?.regime?.confidence, 0)}</div>
            </div>
            <div className={`stat${(counts?.red ?? 0) > 0 ? ' hot' : ''}`}>
              <div className="k">Districts warned</div>
              <div className="v">{console_?.summary?.districts_in_warning ?? '—'}
                <span className="u"> / {districts.length}</span></div>
              <div className="u">
                <span style={{ color: 'var(--red)' }}>■</span> {counts?.red ?? 0}
                {' '}<span style={{ color: 'var(--orange)' }}>■</span> {counts?.orange ?? 0}
                {' '}<span style={{ color: 'var(--yellow)' }}>■</span> {counts?.yellow ?? 0}
              </div>
            </div>
            <div className={`stat${(console_?.summary?.max_corrected_mm ?? 0) >= 204.5 ? ' hot' : ''}`}>
              <div className="k">Highest corrected</div>
              <div className="v">{fmt(console_?.summary?.max_corrected_mm)}<span className="u"> mm</span></div>
              <div className="u">raw model {fmt(console_?.summary?.max_raw_mm)} mm</div>
            </div>
            <div className="stat">
              <div className="k">Mean correction</div>
              <div className="v">{signed(console_?.summary?.mean_bias_adjustment_mm)}<span className="u"> mm</span></div>
              <div className="u">applied to the raw field</div>
            </div>
            <div className="stat">
              <div className="k">Worst observed</div>
              <div className="v" style={{ color: 'var(--muted)' }}>{fmt(console_?.summary?.max_observed_mm)}<span className="u"> mm</span></div>
              <div className="u">reference only, not a model input</div>
            </div>
          </div>

          {console_?.horizon?.length > 0 && (
            <div className="panel-body" style={{ paddingTop: 8 }}>
              <div className="small muted" style={{ marginBottom: 4 }}>
                Five-day outlook for this date · districts in warning by lead
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {console_.horizon.map((h) => {
                  const width = Math.max(120, Math.min(240, (h.max_corrected_mm / 320) * 240));
                  return (
                    <button key={h.lead} onClick={() => onSession({ lead: h.lead })}
                            className="panel" style={{ padding: '6px 9px', minWidth: width, textAlign: 'left', border: lead === h.lead ? '1px solid var(--cyan)' : '1px solid var(--line)', boxShadow: lead === h.lead ? '0 0 16px rgba(0,212,255,.20)' : 'none' }}>
                      <div className="tiny muted">DAY {h.lead}{lead === h.lead ? ' · shown' : ''}</div>
                      <div className="mono" style={{ fontSize: 14, fontWeight: 600 }}>{fmt(h.max_corrected_mm)} mm</div>
                      <div className="tiny">
                        {h.districts_warned} warned · {h.red} red · P90 max {fmt(h.max_p90_mm, 0)} mm
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <RegimeTimeline days={timeline} currentDate={date} lead={lead}
                        onSelect={(d) => onSession({ date: d, source: 'timeline' })} />

        <div className="split">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <ErrorBoundary
              label="Map"
              fallback="The interactive map could not start (no tile access, or the browser blocked canvas). Warnings, corrections and probabilities are unaffected — use the table below."
            >
              <MapPanel
                geojson={geojson} districts={districts} selectedId={selected} onSelect={setSelected}
                layer={layer} onLayer={setLayer} view={view} onView={setView}
                grid={grid} gridDates={gridDates} gridDate={gridDates[0]} onGridDate={() => {}}
                date={date} lead={lead} provenance={console_?.provenance}
              />
            </ErrorBoundary>

            <CaseStrip currentDate={date}
                       onGo={(d, l) => onSession({ date: d, lead: l, source: 'case' })} />

            <DistrictTable districts={filtered} selectedId={selected} onSelect={setSelected}
                           filter={filter} onFilterChange={setFilter} />

            <div className="panel">
              <div className="panel-head">
                <span className="panel-title">Where the correction bites hardest today</span>
                <span className="tiny muted">largest absolute adjustment, raw model → corrected</span>
              </div>
              <div className="panel-body">
                {[...districts]
                  .sort((a, b) => Math.abs(b.adjustment_mm) - Math.abs(a.adjustment_mm))
                  .slice(0, 8)
                  .map((d) => (
                    <div key={d.district_id} className="bar-row" style={{ marginBottom: 4, gridTemplateColumns: '118px 1fr 128px' }}>
                      <button className="btn" style={{ padding: '2px 6px', textAlign: 'left' }}
                              onClick={() => setSelected(d.district_id)}>
                        {d.district_name}
                      </button>
                      <div className="bar-track">
                        <div className="bar-fill"
                             style={{
                               width: `${Math.min(Math.abs(d.adjustment_mm) / 120 * 100, 100)}%`,
                               background: d.adjustment_mm >= 0 ? 'var(--green)' : 'var(--red)',
                             }} />
                      </div>
                      <span className="num small">
                        {signed(d.adjustment_mm)} mm ({fmt(d.raw_mm)}→{fmt(d.corrected_mm)})
                      </span>
                    </div>
                  ))}
                <div className="tiny muted" style={{ marginTop: 6 }}>
                  The raw field is damped almost everywhere on heavy days; the districts where the
                  correction is largest are exactly the ones a duty officer would otherwise miss.
                </div>
              </div>
            </div>
          </div>

          <div className="rail">
            <div className="panel">
              <div className="panel-head">
                <span className="panel-title">Highest exposure on {date}</span>
                <span className="tiny muted">districts in warning, ranked</span>
              </div>
              <div className="table-scroll" style={{ maxHeight: 250 }}>
                <table className="data">
                  <thead>
                    <tr><th>District</th><th>Category</th><th className="num">Corr.</th><th className="num">P(≥65)</th></tr>
                  </thead>
                  <tbody>
                    {topDistricts.map((d) => (
                      <tr key={d.district_id} onClick={() => setSelected(d.district_id)}
                          aria-selected={d.district_id === selected}>
                        <td><b>{d.district_name}</b><div className="tiny muted">{d.state_name}</div></td>
                        <td><span className={`cat-chip cat-${d.category}`}>{CATEGORY_META[d.category].short}</span></td>
                        <td className="num">{fmt(d.corrected_mm)}</td>
                        <td className="num">{pct(d.p_heavy, 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <DistrictRail districtId={selected} date={date} lead={lead} onClose={() => setSelected(null)} />

            <div className="panel">
              <div className="panel-head"><span className="panel-title">Shortcuts · no mouse needed</span></div>
              <div className="panel-body">
                <table className="data small">
                  <tbody>
                    <tr><td><span className="mono">← →</span></td><td>previous / next day</td></tr>
                    <tr><td><span className="mono">1…5</span></td><td>lead time</td></tr>
                    <tr><td><span className="mono">n</span></td><td>next significant day</td></tr>
                    <tr><td><span className="mono">p</span></td><td>previous significant day</td></tr>
                    <tr><td><span className="mono">l</span></td><td>cycle map layer</td></tr>
                    <tr><td><span className="mono">b</span></td><td>bulletin</td></tr>
                    <tr><td><span className="mono">s</span></td><td>story mode (Guided demo)</td></tr>
                    <tr><td><span className="mono">Esc</span></td><td>close panel</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

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
