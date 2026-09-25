import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import Today from './components/Today';
import SkillLab from './components/verification/SkillLab';
import Method from './components/verification/Method';
import Landing from './components/Landing';
import ApiView from './components/ApiView';
import AboutModal from './components/AboutModal';
import Icon from './components/Icon';

const TABS = [
  { id: 'today', label: 'Today', icon: 'today' },
  { id: 'skill', label: 'Skill lab', icon: 'skill' },
  { id: 'method', label: 'Method', icon: 'method' },
  { id: 'api', label: 'API', icon: 'api' },
];

// Page headings, SAGAR-style: one large statement of what the screen is for.
const PAGE = {
  today: { title: 'District rainfall warnings', sub: 'Corrected forecast, IMD warning category and exceedance probability for every district.' },
  skill: { title: 'Verification skill lab', sub: 'How much the regime-aware correction actually helps — with confidence intervals, including where it does not.' },
  method: { title: 'Method and architecture', sub: 'How the pipeline is built, how modules talk to each other, and what the models can and cannot do.' },
  api: { title: 'API reference', sub: 'Every number on screen comes from these endpoints. Probe them live.' },
};

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
  const [about, setAbout] = useState(false);

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
          <button className="brand" onClick={() => setTab('landing')} title="Back to the overview">
            <img src="/favicon.svg" alt="" className="brand-logo" />
            <span className="brand-mark">MonsoonIQ</span>
          </button>

          <nav className="tabs" role="tablist" aria-label="Screens">
            {TABS.map((t) => (
              <button key={t.id} role="tab" className="tab" aria-selected={tab === t.id}
                      onClick={() => setTab(t.id)}>
                <Icon name={t.icon} />
                <span>{t.label}</span>
              </button>
            ))}
            <button className="tab" onClick={() => setAbout(true)}>
              <Icon name="info" />
              <span>About</span>
            </button>
          </nav>

          <div className="masthead-right">
            <span className="chip live" title={health ? JSON.stringify(health.artifacts) : ''}>
              <span className="dot" />
              {health?.status === 'healthy' ? 'Live · engine ready' : 'Engine degraded'}
            </span>
            <span className="chip" title="data provenance — synthetic archive, not live IMD data">
              {health?.provenance === 'SYNTHETIC_PHYSICALLY_PLAUSIBLE'
                ? 'Provenance: synthetic research archive'
                : `Provenance: ${health?.provenance || 'unknown'}`}
            </span>
            {freshness && freshness !== '<volatile>' && <span className="chip">Models loaded {freshness}</span>}
            <button className="btn back" onClick={() => setTab('landing')} title="Overview screen (h)">
              <Icon name="back" />
              <span>Overview</span>
            </button>
          </div>
        </div>
      </header>

      {PAGE[tab] && (
        <div className="page-head anim-up" key={tab}>
          <h1>{PAGE[tab].title}</h1>
          <p>{PAGE[tab].sub}</p>
        </div>
      )}

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

      {about && <AboutModal onClose={() => setAbout(false)} />}

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
