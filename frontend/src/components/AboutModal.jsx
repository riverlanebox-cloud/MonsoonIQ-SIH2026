import React, { useEffect } from 'react';
import Icon from './Icon';
import { isReal } from '../lib/provenance';

/**
 * "About MonsoonIQ" — the platform overview and data-source acknowledgements,
 * laid out the way SAGAR's About dialog is. Source status follows the archive the
 * API is serving (real IMD + GFS by default; the synthetic simulator on request).
 */
const REAL_SOURCES = [
  {
    name: 'IMD gridded rainfall', tag: 'Observations', tone: 'green',
    org: 'India Meteorological Department, Pune (Pai et al. 2014)',
    text: '0.25° daily gridded rainfall, June–September 2021–2025 — the truth every forecast is scored against.',
  },
  {
    name: 'NOAA GFS', tag: 'Raw forecast', tone: 'green',
    org: 'NOAA / NCEP via AWS Open Data',
    text: '00 UTC runs, Day 1–5 rainfall at 0.5° plus nine dynamical predictors at 1°. Public stand-in for NCMRWF NCUM.',
  },
  {
    name: 'Census 2011 districts', tag: 'Boundaries', tone: 'blue',
    org: 'DataMeet India maps (CC BY 2.5 IN)',
    text: 'Real district boundaries used to aggregate grid rainfall to the 53 modelled districts.',
  },
  {
    name: 'Synthetic simulator', tag: 'Optional', tone: 'blue',
    org: 'Physically-parameterised monsoon generator (src/data)',
    text: 'Kept for method demos: `make synthetic` switches every screen to the seeded archive.',
  },
];

const SYN_SOURCES = [
  {
    name: 'Synthetic archive', tag: 'In use', tone: 'green',
    org: 'Physically-parameterised monsoon simulator (src/data)',
    text: 'Generates the forecast/observation pairs every screen and every skill figure in this build is computed from.',
  },
  {
    name: 'IMD gridded rainfall', tag: 'Available', tone: 'blue',
    org: 'India Meteorological Department, Pune',
    text: '0.25° daily gridded rainfall — `make real` builds the real archive from it.',
  },
  {
    name: 'NOAA GFS', tag: 'Available', tone: 'blue',
    org: 'NOAA / NCEP via AWS Open Data',
    text: 'Day 1–5 forecasts and dynamical predictors used by the real archive.',
  },
];

export default function AboutModal({ onClose, provenance }) {
  const SOURCES = isReal(provenance) ? REAL_SOURCES : SYN_SOURCES;
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="about-title">
      <div className="modal-backdrop" onClick={onClose} />
      <div className="modal about">
        <div className="modal-hero">
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <Icon name="close" size={18} />
          </button>
          <div className="modal-hero-row">
            <img src="/favicon.svg" alt="" className="modal-logo" />
            <div>
              <h2 id="about-title">About MonsoonIQ</h2>
              <p>Regime-aware AI post-processing of monsoon rainfall forecasts</p>
            </div>
          </div>
        </div>

        <div className="modal-body">
          <section>
            <h3><Icon name="globe" size={18} /> What it is</h3>
            <p>
              MonsoonIQ turns a raw rainfall forecast into what a district administration needs:
              a corrected rainfall field, IMD warning levels, the chance of heavy rain, and a
              bulletin ready to send. The correction depends on the monsoon regime of the day,
              because the model's errors differ between, say, a depression day and a break day.
            </p>
            <p className="tiny muted" style={{ marginTop: 8 }}>
              SIH26080 · Ministry of Earth Sciences (NCMRWF) · Disaster Management
            </p>
          </section>

          <section>
            <h3><Icon name="database" size={18} /> Data</h3>
            <div className="source-grid">
              {SOURCES.map((s) => (
                <div key={s.name} className="source-card">
                  <div className="source-head">
                    <h4>{s.name}</h4>
                    <span className={`pill ${s.tone}`}>{s.tag}</span>
                  </div>
                  <div className="source-org">{s.org}</div>
                  <p>{s.text}</p>
                </div>
              ))}
            </div>
          </section>

          <div className="modal-foot">
            Research prototype for Smart India Hackathon 2026 — not an official IMD/NCMRWF product.
            {isReal(provenance) && ' Boundaries © DataMeet contributors, CC BY 2.5 IN.'}
          </div>
        </div>
      </div>
    </div>
  );
}
