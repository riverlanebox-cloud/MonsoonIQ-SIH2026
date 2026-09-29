import React from 'react';
import { FiX, FiTarget, FiCpu, FiShield, FiUsers } from 'react-icons/fi';
import { LogoMark } from './AppHeader';

export default function AboutModal({ onClose }) {
  return (
    <div className="modal-back" onClick={onClose} role="dialog" aria-modal="true">
      <div className="about" onClick={(e) => e.stopPropagation()}>
        <div className="about-hero">
          <button className="icon-btn close" onClick={onClose} aria-label="close"><FiX /></button>
          <LogoMark size={56} />
          <div>
            <h2>About MonsoonIQ</h2>
            <div className="about-sub">Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts</div>
          </div>
        </div>
        <div className="about-body">
          <section>
            <h3><FiTarget /> Problem statement 26080</h3>
            <p>NWP rainfall errors over India change with the weather regime. A single bias correction
              cannot fix a damped orographic extreme on the Western Ghats and an over-forecast break-monsoon
              day over central India at the same time. MonsoonIQ first identifies the regime, then applies
              the correction learned for it — for Ministry of Earth Sciences / NCMRWF.</p>
          </section>
          <section>
            <h3><FiCpu /> What it delivers</h3>
            <ul>
              <li>Weather regime classifier — 7 regimes, calibrated soft posteriors</li>
              <li>Bias-corrected Day 1–5 rainfall — mixture of regime experts</li>
              <li>Heavy-rain probabilities at IMD thresholds (64.5 / 115.6 / 204.5 mm)</li>
              <li>District product — map, table, bulletin, CSV, CAP alerts</li>
              <li>Verification — RMSE, ETS, CSI, POD, FAR, FSS with bootstrap CIs</li>
            </ul>
          </section>
          <section>
            <h3><FiShield /> Honest by design</h3>
            <p>Every artifact carries its data provenance. Negative results (e.g. where the regime-aware system
              does not beat a regime-agnostic learner) are shown on screen, not hidden. The demo archive is a
              reproducible synthetic benchmark; one command rebuilds everything on IMD observations.</p>
          </section>
          <section>
            <h3><FiUsers /> Built for</h3>
            <p>NCMRWF / IMD forecasters, state emergency operations centres and district administrations.
              Works fully offline — no map tiles or cloud services needed on demo day.</p>
          </section>
        </div>
      </div>
    </div>
  );
}
