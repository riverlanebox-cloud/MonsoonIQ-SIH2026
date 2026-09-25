import React, { useEffect } from 'react';
import Icon from './Icon';

/**
 * "About MonsoonIQ" — the platform overview and data-source acknowledgements,
 * laid out the way SAGAR's About dialog is. Source status is stated plainly:
 * this build runs on the synthetic archive; the real-data loaders exist in
 * src/data/downloader.py but are not what the numbers on screen come from.
 */
const SOURCES = [
  {
    name: 'Synthetic archive', tag: 'In use', tone: 'green',
    org: 'Physically-parameterised monsoon simulator (src/data)',
    text: 'Generates the forecast/observation pairs every screen and every skill figure in this build is computed from.',
  },
  {
    name: 'IMD gridded rainfall', tag: 'Loader', tone: 'blue',
    org: 'India Meteorological Department, Pune',
    text: '0.25° daily gridded rainfall — the verifying observations once real data is loaded.',
  },
  {
    name: 'NOAA GFS', tag: 'Loader', tone: 'blue',
    org: 'National Oceanic and Atmospheric Administration',
    text: 'Global NWP rainfall forecasts at Day 1–5 leads, standing in for the raw dynamical field.',
  },
  {
    name: 'ERA5', tag: 'Loader', tone: 'blue',
    org: 'Copernicus Climate Data Store (ECMWF)',
    text: 'Reanalysis predictors — winds, humidity, pressure — for the regime classifier.',
  },
];

export default function AboutModal({ onClose }) {
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
            <h3><Icon name="globe" size={18} /> Platform overview</h3>
            <p>
              MonsoonIQ takes a dynamical rainfall forecast and turns it into what a district
              administration needs: a bias-corrected rainfall field, IMD-scaled district warnings,
              exceedance probabilities and a ready-to-file bulletin. The correction is conditioned
              on the prevailing monsoon regime — the same raw value is corrected differently on a
              depression day than on a break day, because the model's errors differ.
            </p>
            <p className="tiny muted" style={{ marginTop: 8 }}>
              SIH26080 · Ministry of Earth Sciences (NCMRWF) · Disaster Management
            </p>
          </section>

          <section>
            <h3><Icon name="database" size={18} /> Data sources &amp; acknowledgements</h3>
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
          </div>
        </div>
      </div>
    </div>
  );
}
