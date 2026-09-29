import React, { useEffect, useState } from 'react';
import { FiDatabase, FiCheckCircle, FiCircle, FiTerminal, FiExternalLink, FiArrowRight } from 'react-icons/fi';
import { api } from '../api';

const FLOW = [
  { k: 'Observe', v: 'IMD 0.25° gridded rain → district mean/max on Census-2011 polygons' },
  { k: 'Label', v: 'Rajeevan CMZ active/break + pressure, vorticity, terrain, coast rules' },
  { k: 'Classify', v: 'Calibrated LightGBM regime posterior from issue-time predictors' },
  { k: 'Correct', v: 'Per-regime quantile mapping + residual experts, blended by posterior' },
  { k: 'Warn', v: 'P(≥64.5/115.6/204.5 mm), P10–P90, IMD category, bulletin, CAP' },
  { k: 'Verify', v: 'RMSE · ETS · CSI · POD · FAR · FSS on held-out seasons, bootstrap CIs' },
];

export default function DataSources() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { api.dataSources().then(setD).catch((e) => setErr(String(e.message || e))); }, []);
  const real = d?.active_profile === 'real';
  const rm = d?.real_archive_metadata;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title"><FiDatabase /> Data Sources</h1>
          <div className="page-sub">Every dataset the system is built on, what it is used for, and how to fetch it.</div>
        </div>
        <span className={`status-pill big ${real ? 'real' : 'synth'}`}>
          Active archive: {real ? 'IMD observed + archived NWP' : 'synthetic benchmark'}
        </span>
      </div>
      {err && <div className="notice err">{err}</div>}

      <div className="flow-strip">
        {FLOW.map((f, i) => (
          <React.Fragment key={f.k}>
            <div className="flow-step" style={{ animationDelay: `${i * 0.08}s` }}>
              <div className="fs-k">{i + 1}. {f.k}</div>
              <div className="fs-v">{f.v}</div>
            </div>
            {i < FLOW.length - 1 && <FiArrowRight className="flow-arrow" />}
          </React.Fragment>
        ))}
      </div>

      <div className="ds-grid">
        {(d?.sources || []).map((s) => (
          <div key={s.id} className={`ds-card ${s.active ? 'active' : ''}`}>
            <div className="ds-top">
              <span className="ds-role">{s.role}</span>
              {s.active ? <FiCheckCircle className="ok" title="used by the active archive" /> : <FiCircle className="muted" title="used by the other profile" />}
            </div>
            <h3>{s.name}</h3>
            <div className="ds-provider">{s.provider}</div>
            <dl>
              <dt>Resolution</dt><dd>{s.resolution}</dd>
              <dt>Period</dt><dd>{s.period}</dd>
              <dt>Access</dt><dd className="mono small">{s.access}</dd>
              <dt>Licence</dt><dd>{s.licence}</dd>
              <dt>Cite</dt><dd>{s.citation}</dd>
            </dl>
            <p className="ds-why">{s.why}</p>
          </div>
        ))}
      </div>

      <div className="two-col">
        <div className="glass">
          <div className="glass-head"><span><FiTerminal /> Build the real archive on your machine</span></div>
          <div className="glass-body">
            <p className="small muted">IMD, Open-Meteo and NOAA need no key; ERA5 needs a free CDS key. Every response is cached, so an interrupted fetch resumes.</p>
            <pre className="code">{`# Linux / macOS
PYTHONPATH=. python scripts/fetch_real_data.py all
MONSOONIQ_PROFILE=real make train evaluate console serve

# Windows PowerShell
.\\scripts\\real_data.ps1

# Operational NCUM-G NetCDF instead of Open-Meteo
python scripts/fetch_real_data.py forecast build --gridded "data/real/raw/ncum/*.nc"`}</pre>
            <p className="small">Config: <span className="mono">configs/data_sources.yaml</span> (years, districts: study | all, model: gfs_seamless | ecmwf_ifs025 | ukmo_global_deterministic_10km).</p>
          </div>
        </div>

        <div className="glass">
          <div className="glass-head"><span>Active archive</span></div>
          <div className="glass-body">
            <table className="kv">
              <tbody>
                <tr><td>Profile</td><td className="mono">{d?.active_profile}</td></tr>
                <tr><td>Archive</td><td className="mono small">{d?.active_archive}</td></tr>
                <tr><td>Train / val / test</td><td className="mono">{d ? `${d.split.train.join(', ')} / ${d.split.val.join(', ')} / ${d.split.test.join(', ')}` : '—'}</td></tr>
                <tr><td>Districts</td><td>{d?.archive_metadata?.districts_count ?? '—'}</td></tr>
                <tr><td>Days</td><td>{d?.archive_metadata?.total_days ?? '—'}</td></tr>
                <tr><td>Type</td><td className="mono small">{d?.archive_metadata?.dataset_type}</td></tr>
              </tbody>
            </table>
            {rm ? (
              <>
                <div className="fp-label" style={{ marginTop: 10 }}>Real archive · regime label counts</div>
                <div className="chip-row">
                  {Object.entries(rm.regime_counts || {}).map(([k, v]) => <span key={k} className="chip2">{k}: {v}</span>)}
                </div>
                <div className="fp-label">IMD-day alignment check (obs vs Day-1 corr.)</div>
                <div className="chip-row">
                  {Object.entries(rm.day_alignment_check || {}).map(([k, v]) => <span key={k} className="chip2 mono">{k}: {v ?? '—'}</span>)}
                </div>
              </>
            ) : (
              <p className="small muted" style={{ marginTop: 10 }}>No real archive built yet on this machine — run the command on the left.</p>
            )}
            <p className="tiny muted" style={{ marginTop: 10 }}>
              <FiExternalLink /> Full dataset notes: docs/DATASETS.md
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
