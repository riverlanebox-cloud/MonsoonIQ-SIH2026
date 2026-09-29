import React, { useState } from 'react';
import Icon from './Icon';
import { apiBase, STATIC } from '../api';

/**
 * API reference, laid out the way SAGAR documents its services: one card per
 * group, a base URL you can copy, and one plain line per endpoint.
 *
 * Nothing here is fetched. The list is written by hand so it reads the same on
 * the live service and on the static site, and so every line says what a
 * response is for rather than what the handler is called.
 */

// Shared payloads are real files on the static site, so they can be opened.
const STATIC_FILE = {
  '/health': '/health.json',
  '/model-card': '/model-card.json',
  '/timeline': '/timeline.json',
  '/events': '/events.json',
  '/grid': '/grid.json',
  '/districts': '/districts.json',
  '/districts/geojson': '/districts/geojson.json',
  '/verification/summary': '/verification/summary.json',
  '/verification/heavy-events': '/verification/heavy-events.json',
  '/verification/regime-value': '/verification/regime-value.json',
  '/verification/report.pdf': '/verification/report.pdf',
  '/case-replays': '/case-replays.json',
};

const GROUPS = [
  {
    id: 'forecast', title: 'Forecast', icon: 'today',
    blurb: 'What the console shows for one date and lead day.',
    endpoints: [
      ['/console', 'Everything on the console screen: warnings, regime, summary and the 5-day outlook', 'date, lead'],
      ['/district/{id}', 'One district in full: correction, percentiles, probabilities, advisory text', 'date, lead'],
      ['/bulletin', 'The bulletin as plain text, English or Hindi', 'date, lead, lang'],
      ['/grid', 'The corrected 0.5° rainfall field with the raw and bias-corrected fields', 'date'],
      ['/export/districts.csv', 'The warning table as a spreadsheet', 'date, lead'],
    ],
  },
  {
    id: 'archive', title: 'Archive', icon: 'method',
    blurb: 'Precomputed once for the whole season, so the console loads them a single time.',
    endpoints: [
      ['/timeline', 'One row per archived day: regime, confidence and warning counts'],
      ['/events', 'The days that carried the most warnings, and the regime spells'],
      ['/case-replays', 'Documented heavy-rain cases with the raw model’s miss'],
      ['/districts', 'The 53 districts with state, zone and coordinates'],
      ['/districts/geojson', 'District boundaries (Census 2011) for the map'],
    ],
  },
  {
    id: 'verification', title: 'Verification', icon: 'skill',
    blurb: 'The numbers behind the Skill lab page.',
    endpoints: [
      ['/verification/summary', 'Scores for every system, with confidence intervals'],
      ['/verification/heavy-events', 'How reliable the heavy-rain probabilities are'],
      ['/verification/regime-value', 'Where regime conditioning helps and where it does not'],
      ['/verification/report.pdf', 'The verification report as a PDF'],
    ],
  },
  {
    id: 'service', title: 'Service', icon: 'api',
    blurb: 'Status and provenance.',
    endpoints: [
      ['/health', 'Whether the models and archive are loaded'],
      ['/model-card', 'Which data, splits and models produced this build'],
      ['/explain/{id}/{date}', 'Which inputs moved one district’s correction (live service only)', 'lead'],
    ],
  },
];

const EXAMPLE = { date: '2024-07-19', lead: 1, id: 'GA_NGA' };

function exampleUrl(path, params) {
  const base = STATIC ? '' : apiBase();
  let p = path.replace('{id}', EXAMPLE.id).replace('{date}', EXAMPLE.date);
  if (STATIC) return STATIC_FILE[path] ? `${apiBase()}${STATIC_FILE[path]}` : null;
  const q = (params || '').split(',').map((s) => s.trim()).filter(Boolean)
    .filter((k) => k !== 'lang')
    .map((k) => `${k}=${EXAMPLE[k] ?? ''}`)
    .join('&');
  return `${base}${p}${q ? `?${q}` : ''}`;
}

export default function ApiView() {
  const [copied, setCopied] = useState(null);
  const origin = typeof window !== 'undefined' ? window.location.origin : '';
  const baseUrl = STATIC ? `${origin}${apiBase()}` : `${origin}${apiBase() || ''}`;

  const copy = (text, key) => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(key);
      setTimeout(() => setCopied(null), 1800);
    }).catch(() => {});
  };

  return (
    <div className="api-page">
      <div className="api-base panel">
        <div>
          <div className="tiny muted" style={{ letterSpacing: '.06em' }}>BASE URL</div>
          <div className="mono" style={{ fontSize: 14, marginTop: 4 }}>{baseUrl}</div>
          <p className="small muted" style={{ marginTop: 8, maxWidth: 640 }}>
            {STATIC
              ? 'This site is a static build: every response was computed from the trained models when the site was built and is served as a file. The endpoints below are the live service’s; the ones marked Open exist here as files.'
              : 'All endpoints are GET and return JSON, except the PDF report and the CSV export. No key is needed.'}
          </p>
        </div>
        <button className="btn" onClick={() => copy(baseUrl, 'base')}>
          <Icon name={copied === 'base' ? 'check' : 'copy'} size={15} />
          {copied === 'base' ? 'Copied' : 'Copy'}
        </button>
      </div>

      <div className="api-grid">
        {GROUPS.map((g) => (
          <section key={g.id} className="panel api-card">
            <div className="panel-head">
              <span className="panel-title"><Icon name={g.icon} size={16} /> {g.title}</span>
              <span className="tiny muted">{g.endpoints.length} endpoints</span>
            </div>
            <div className="panel-body">
              <p className="small muted" style={{ marginBottom: 12 }}>{g.blurb}</p>
              <ul className="api-list">
                {g.endpoints.map(([path, what, params]) => {
                  const url = exampleUrl(path, params);
                  const key = `${g.id}${path}`;
                  return (
                    <li key={path}>
                      <div className="api-line">
                        <span className="badge mono">GET</span>
                        <code className="api-path">{path}</code>
                        {url && (
                          <span className="api-actions">
                            <button className="btn icon" title="Copy example URL" onClick={() => copy(url, key)}>
                              <Icon name={copied === key ? 'check' : 'copy'} size={14} />
                            </button>
                            <a className="btn icon" href={url} target="_blank" rel="noreferrer" title="Open">
                              <Icon name="external" size={14} />
                            </a>
                          </span>
                        )}
                      </div>
                      <div className="api-what">
                        {what}
                        {params && <span className="tiny muted mono"> · {params}</span>}
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          </section>
        ))}
      </div>

      <p className="tiny muted" style={{ padding: '4px 2px' }}>
        Example URLs use {EXAMPLE.date}, Day {EXAMPLE.lead}, North Goa. On the live service the dates
        and leads are free; run it with <code>make serve</code>.
      </p>
    </div>
  );
}
