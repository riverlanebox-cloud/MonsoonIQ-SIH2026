import React, { useEffect, useState } from 'react';

/**
 * Guided demo ("story mode").
 *
 * Eight beats that walk a judge through the product in about 90 seconds without
 * anyone touching the console: what the raw model misses, how the regime changes
 * the correction, what an operator actually files. Each beat moves the console
 * behind the overlay and switches the map layer, so the demo is always live.
 */
const BEATS = [
  {
    id: 'greeting',
    title: 'MonsoonIQ — regime-aware rainfall post-processing',
    body: 'A forecast enters. The system decides which monsoon regime the day belongs to, corrects the rainfall field for that regime, converts it into district warnings, and states honestly how well it works.',
    detail: 'Press → to move. The console behind this overlay is live and will follow each step.',
  },
  {
    id: 'raw_gap',
    title: '1 · The raw model under-forecasts the extremes',
    body: 'Numerical models smooth rainfall. On the heaviest days the raw field is damped by several millimetres to tens of millimetres — exactly where warnings matter most.',
    layer: 'raw',
    detail: 'Map is showing the RAW model field.',
  },
  {
    id: 'corrected',
    title: '2 · The corrected field restores the structure',
    body: 'The same day after regime-conditioned correction. Watch the coastal band and the windward ghats intensify while dry areas stay dry — the correction is not a blanket multiplier.',
    layer: 'corrected',
    detail: 'Map is showing the CORRECTED field.',
  },
  {
    id: 'delta',
    title: '3 · Every change is auditable',
    body: 'Green means the forecast was made wetter, red drier. A forecaster can see at a glance what the statistical layer did to the dynamical forecast, per district.',
    layer: 'adjustment',
    detail: 'Map is showing CORRECTED − RAW.',
  },
  {
    id: 'regime',
    title: '4 · The regime decides how the correction behaves',
    body: 'This is the core idea. A break-monsoon day and a depression day are corrected by different fitted experts, because their error structures are different. The classifier runs on published synoptic criteria.',
    layer: 'regime',
    detail: 'Map is showing the detected REGIME.',
  },
  {
    id: 'probability',
    title: '5 · Warnings carry a probability',
    body: 'Exceedance models give P(≥64.5), P(≥115.6) and P(≥204.5) mm per district. A 70% chance of heavy rain is operationally different from a 20% chance, even if both "look wet".',
    layer: 'p_heavy',
    detail: 'Map is showing P(≥ 64.5 mm).',
  },
  {
    id: 'verification',
    title: '6 · And we show where it fails',
    body: 'On the held-out period the corrected field beats the raw model and the quantile-mapped baseline. Against a strong regime-agnostic learner, the heavy-rain categorical gain is not statistically significant — that is printed on the Skill lab page, in the report, and in the repository.',
    tab: 'skill',
    detail: 'Moving to the Skill lab.',
  },
  {
    id: 'limits',
    title: '7 · What we would need before operations',
    body: 'Real IMD 0.25° gridded rainfall for fitting, real GFS/ECMWF archived forecasts for verification, and IMD gridded data to replace the simplified district geometry. The pipeline, the harness and the console do not change — only the data source does.',
    tab: 'method',
    detail: 'Moving to Method (architecture and model card).',
  },
];

export default function StoryMode({ onClose, onNavigate, currentDate, onLayer, onTab }) {
  const [i, setI] = useState(0);
  const beat = BEATS[i];

  useEffect(() => {
    if (beat.layer && onLayer) onLayer(beat.layer);
    if (beat.tab && onTab) onTab(beat.tab);
  }, [beat, onLayer, onTab]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowRight') setI((v) => Math.min(v + 1, BEATS.length - 1));
      else if (e.key === 'ArrowLeft') setI((v) => Math.max(v - 1, 0));
      else if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const go = (id) => {
    if (id === 'case_konkan') { onNavigate('2019-07-26', 1); setI(1); }
  };

  return (
    <div style={{
      position: 'fixed', left: 16, right: 16, bottom: 16, zIndex: 1200,
      background: 'rgba(10, 25, 41, .97)', border: '1px solid rgba(0, 212, 255, .45)',
      borderRadius: 6, backdropFilter: 'blur(8px)',
      boxShadow: '0 10px 40px rgba(0,0,0,.6), 0 0 26px rgba(0,212,255,.16)', maxWidth: 940, margin: '0 auto',
    }}>
      <div className="panel-head" style={{ borderBottom: '1px solid var(--line-soft)' }}>
        <span className="panel-title">
          Guided demo · step {i + 1} of {BEATS.length}
          <span className="muted small" style={{ marginLeft: 8, fontWeight: 400 }}>{beat.detail}</span>
        </span>
        <div className="cb-group">
          <div className="seg">
            <button aria-pressed onClick={() => setI(Math.max(i - 1, 0))}>◀ prev</button>
            <button aria-pressed onClick={() => setI(Math.min(i + 1, BEATS.length - 1))}>next ▶</button>
          </div>
          <button className="btn" onClick={() => { onNavigate('2019-07-26', 1); }}>
            jump to a coastal case
          </button>
          <button className="btn" onClick={onClose}>close (Esc)</button>
        </div>
      </div>
      <div className="panel-body">
        <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 5, color: '#e9f2fa' }}>{beat.title}</div>
        <div className="small" style={{ maxWidth: 760 }}>{beat.body}</div>
        <div style={{ display: 'flex', gap: 4, marginTop: 10 }}>
          {BEATS.map((b, k) => (
            <button key={b.id} onClick={() => setI(k)} title={b.title}
                    style={{ flex: 1, height: 5, border: 0, borderRadius: 3, padding: 0,
                             background: k <= i ? '#00d4ff' : '#1b3549',
                             boxShadow: k === i ? '0 0 10px rgba(0,212,255,.7)' : 'none' }} />
          ))}
        </div>
      </div>
    </div>
  );
}
