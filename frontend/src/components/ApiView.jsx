import React, { useEffect, useMemo, useState } from 'react';
import { apiBase } from '../api';

/** Read-only endpoints, with a plain-language line each. Used when the spec is
 *  unreachable (offline build, static file server) so the reference never renders
 *  empty; when the service is up, the real OpenAPI document wins. */
const FALLBACK = [
  ['GET', '/health', 'Service status, loaded models and artifact list'],
  ['GET', '/console', 'Everything one console screen needs for a date and lead day'],
  ['GET', '/timeline', 'Forecast archive: one record per day, regime + confidence'],
  ['GET', '/events', 'Regime spells and the days that carried the most warnings'],
  ['GET', '/district/{district_id}', 'Correction breakdown, confidence and drivers'],
  ['GET', '/bulletin', 'The duty-officer bulletin text (en / hi)'],
  ['GET', '/grid', '0.25° corrected field with raw and bias-corrected baselines'],
  ['GET', '/verification/summary', 'Skill scorecard, significance claims, ensembles'],
  ['GET', '/verification/heavy-events', 'Threshold exceedance reliability and ROC'],
  ['GET', '/verification/regime-value', 'Per-regime CSI improvement and coverage'],
  ['GET', '/verification/report.pdf', 'Generated verification report'],
  ['GET', '/model-card', 'Provenance, archive fingerprint, honesty statement'],
  ['GET', '/case-replays', 'Documented events with known raw-model failures'],
  ['GET', '/explain/{district_id}/{date}', 'Feature contributions behind one correction'],
  ['GET', '/export/districts.csv', 'Current screen as a spreadsheet'],
];

const GROUPS = [
  ['forecast', /^\/(console|timeline|events|grid|bulletin|forecast)/],
  ['district detail', /^\/(district|districts|regime|explain)/],
  ['verification', /^\/verification/],
  ['service', /^\/(health|model-card|case-replays|export)/],
];

const groupOf = (path) => GROUPS.find(([, re]) => re.test(path))?.[0] || 'service';

/** Query strings used by the probe buttons — a real archived date. */
const PROBE_QUERY = {
  '/console': 'date=2020-08-05&lead=1',
  '/district/OD_BBS': 'date=2020-08-05&lead=1',
  '/bulletin': 'date=2020-08-05&lead=1',
  '/grid': 'date=2020-08-05',
  '/events': 'limit=5',
  '/export/districts.csv': 'date=2020-08-05&lead=1',
};

/** Issue one GET and report what came back, timed. Kept outside the component so
 *  the request and its clock are plainly an effect of a click, not of a render. */
async function probeRequest(url) {
  const t0 = performance.now();
  try {
    const res = await fetch(url);
    const text = await res.text();
    return { status: res.status, ms: performance.now() - t0, bytes: text.length };
  } catch {
    return { status: 'network error', ms: performance.now() - t0, bytes: 0 };
  }
}

export default function ApiView() {
  const [spec, setSpec] = useState(null);
  const [offline, setOffline] = useState(false);
  const [result, setResult] = useState({});
  const [busy, setBusy] = useState(null);

  useEffect(() => {
    fetch(`${apiBase()}/openapi.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then(setSpec)
      .catch(() => setOffline(true));
  }, []);

  const routes = useMemo(() => {
    if (!spec?.paths) return FALLBACK.map(([method, path, summary]) => ({ method, path, summary, params: [] }));
    return Object.entries(spec.paths).flatMap(([path, ops]) =>
      Object.entries(ops)
        .filter(([method]) => method === 'get')
        .map(([method, op]) => ({
          method: method.toUpperCase(),
          path,
          summary: op.summary || op.description?.split('.')[0] || '',
          params: (op.parameters || []).map((p) => p.name),
        })),
    );
  }, [spec]);

  const send = async (path) => {
    // Probe each route with a date that exists in the archive, so a probe that
    // returns 200 means the endpoint genuinely works end to end.
    const probe = path.replace('{district_id}', 'OD_BBS').replace('{date}', '2020-08-05');
    const query = PROBE_QUERY[probe];
    const url = `${apiBase()}${probe}${query ? `?${query}` : ''}`;
    setBusy(path);
    const outcome = await probeRequest(url);
    setResult((r) => ({ ...r, [path]: outcome }));
    setBusy(null);
  };

  return (
    <>
      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">Service reference</span>
          <span className="tiny muted" style={{ marginLeft: 8 }}>
            {offline
              ? 'spec unavailable · static list of the published routes'
              : `rendered from the live OpenAPI document · ${routes.length} read endpoints`}
          </span>
        </div>
        <div className="panel-body small muted">
          The console reads forecast state over HTTP; the same endpoints are open, so a state
          emergency operations centre can pull corrected warnings into its own dashboard instead of
          retyping them. Every number on screen is traceable to one of these responses. Send a probe
          to see the status and response time from this browser.
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">Endpoints</span>
          <span className="tiny muted">GET only · all responses JSON except the report PDF and CSV export</span>
        </div>
        <div className="panel-body" style={{ overflowX: 'auto' }}>
          <table className="data">
            <thead>
              <tr>
                <th style={{ width: 54 }}>Method</th>
                <th style={{ width: 300 }}>Path</th>
                <th>What it returns</th>
                <th style={{ width: 190, textAlign: 'right' }}>Probe</th>
              </tr>
            </thead>
            <tbody>
              {GROUP_ORDER(routes).map(([group, items]) => (
                <React.Fragment key={group}>
                  <tr>
                    <td colSpan={4} className="tiny muted"
                        style={{ textTransform: 'uppercase', letterSpacing: '.9px', paddingTop: 12 }}>
                      {group}
                    </td>
                  </tr>
                  {items.map((r) => {
                    const res = result[r.path];
                    return (
                      <tr key={r.path}>
                        <td><span className="badge mono">{r.method}</span></td>
                        <td className="mono" style={{ color: 'var(--ink)' }}>{r.path}</td>
                        <td className="muted">
                          {r.summary}
                          {r.params?.length > 0 && (
                            <div className="tiny mono" style={{ marginTop: 3 }}>
                              {r.params.map((p) => (p.startsWith('{') || r.path.includes(`{${p}}`) ? p : `${p}=`)).join(' · ')}
                            </div>
                          )}
                        </td>
                        <td style={{ textAlign: 'right' }}>
                          <button className="btn" disabled={busy === r.path} onClick={() => send(r.path)}>
                            {busy === r.path ? 'sending…' : 'Send'}
                          </button>
                          {res && (
                            <span className="tiny mono" style={{ marginLeft: 8, color: res.status === 200 ? 'var(--green)' : 'var(--red)' }}>
                              {res.status} · {res.ms.toFixed(0)} ms · {(res.bytes / 1024).toFixed(1)} kB
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function GROUP_ORDER(routes) {
  const buckets = new Map(GROUPS.map(([name]) => [name, []]));
  routes.forEach((r) => buckets.get(groupOf(r.path)).push(r));
  return [...buckets.entries()].filter(([, items]) => items.length);
}
