import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import Today from './components/Today';
import SkillLab from './components/verification/SkillLab';
import Method from './components/verification/Method';
import Landing from './components/Landing';
import ApiView from './components/ApiView';

const TABS = [
  { id: 'today', label: 'Today' },
  { id: 'skill', label: 'Skill lab' },
  { id: 'method', label: 'Method' },
  { id: 'api', label: 'API' },
];

const HOME = { tab: 'landing', date: null, lead: 1 };

function readHash() {
  const h = window.location.hash.replace(/^#/, '');
  if (!h || h === '/' || h === 'home') return HOME;
  const [tab, date, lead] = h.split('/');
  return {
    tab: TABS.some((t) => t.id === tab) ? tab : 'landing',
    date: /^\d{4}-\d{2}-\d{2}$/.test(date || '') ? date : null,
    lead: /^[1-5]$/.test(lead || '') ? Number(lead) : 1,
  };
}

export default function App() {
  const initial = useMemo(() => readHash(), []);
  const [tab, setTab] = useState(initial.tab);
  const [session, setSession] = useState({ date: initial.date, lead: initial.lead });
  const [health, setHealth] = useState(null);
  const [toast, setToast] = useState(null);

  useEffect(() => {
    const poll = () => api.health().then(setHealth).catch(() => setHealth(null));
    poll();
    const t = setInterval(poll, 60000);
    return () => clearInterval(t);
  }, []);

  // The hash is the shareable link: #today/2019-07-26/3 opens that view directly.
  useEffect(() => {
    const next = tab === 'landing'
      ? '#/'
      : `#${tab}${session.date ? `/${session.date}/${session.lead}` : ''}`;
    if (window.location.hash !== next) window.history.replaceState(null, '', next);
  }, [tab, session]);

  useEffect(() => {
    const onHash = () => {
      const h = readHash();
      setTab(h.tab);
      setSession((s) => ({ ...s, date: h.date || s.date, lead: h.lead }));
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  useEffect(() => {
    if (!toast) return undefined;
    const t = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(t);
  }, [toast]);

  const onSession = useCallback((patch) => setSession((s) => ({ ...s, ...patch })), []);

  useEffect(() => {
    const onKey = (e) => {
      const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);
      if (typing) return;
      if (e.key === '?') setToast('Shortcuts: ← → days · 1-5 lead · n/p significant day · l map layer · b bulletin · s story mode · h overview');
      if (e.key === 'h') setTab('landing');
      if (e.key === 'l') {
        window.dispatchEvent(new CustomEvent('monsooniq:cycle-layer'));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const freshness = health?.loaded_utc;

  if (tab === 'landing') {
    return (
      <Landing
        onEnter={(date) => {
          setSession((s) => ({ ...s, date: s.date || date || null, lead: 1 }));
          setTab('today');
        }}
        onMethod={() => setTab('skill')}
      />
    );
  }

  return (
    <div className="app">
      <header className="masthead">
        <div className="masthead-inner">
          <button className="brand" onClick={() => setTab('landing')}
                  title="Back to the overview" style={{ background: 'none', border: 0, padding: 0 }}>
            <span className="brand-mark">MonsoonIQ</span>
            <span className="brand-sub">
              Regime-aware post-processing of NWP rainfall · IMD warning scale
            </span>
          </button>
          <div className="cb-group" style={{ marginLeft: 12 }}>
            <span className="chip live" title={health ? JSON.stringify(health.artifacts) : ''}>
              <span className="dot" />
              {health?.status === 'healthy' ? 'Live · engine ready' : 'Engine degraded'}
            </span>
            <span className="chip" title="data provenance — synthetic archive, not live IMD data">
              {health?.provenance === 'SYNTHETIC_PHYSICALLY_PLAUSIBLE'
                ? 'Provenance: synthetic research archive'
                : `Provenance: ${health?.provenance || 'unknown'}`}
            </span>
            {freshness && <span className="chip">Models loaded {freshness}</span>}
            <button className="btn" onClick={() => setTab('landing')} title="Overview screen (h)">
              Overview
            </button>
          </div>
          <nav className="tabs" role="tablist">
            {TABS.map((t) => (
              <button key={t.id} role="tab" className="tab" aria-selected={tab === t.id}
                      onClick={() => setTab(t.id)}>{t.label}</button>
            ))}
          </nav>
        </div>
      </header>

      {tab === 'today' && (
        <Today session={session} onSession={onSession} onToast={setToast} onTab={setTab} />
      )}
      {tab === 'skill' && <SkillLab />}
      {tab === 'method' && <Method />}
      {tab === 'api' && <ApiView />}

      <footer className="footer">
        Research prototype. Forecast fields are produced by a synthetic physically-plausible archive
        generated inside this repository and are <b>not</b> official IMD/NCMRWF products; do not use for
        public warnings. Verification figures are internal comparisons on that archive.
        {' '}Press <span className="mono">?</span> for shortcuts.
      </footer>

      {toast && (
        <div style={{
          position: 'fixed', bottom: 18, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(10,25,41,.97)', color: 'var(--ink)', padding: '9px 15px', borderRadius: 5,
          border: '1px solid rgba(0,212,255,.4)', boxShadow: '0 8px 30px rgba(0,0,0,.55)',
          fontSize: 12.5, zIndex: 1500, maxWidth: '90vw',
        }}>{toast}</div>
      )}
    </div>
  );
}
