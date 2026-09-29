import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import Landing from './components/Landing';
import AppHeader from './components/AppHeader';
import Dashboard from './components/Dashboard';
import ForecastMap from './components/ForecastMap';
import Today from './components/Today';
import SkillLab from './components/verification/SkillLab';
import Method from './components/verification/Method';
import DataSources from './components/DataSources';
import ApiView from './components/ApiView';
import AboutModal from './components/AboutModal';

const TABS = ['dashboard', 'map', 'today', 'skill', 'data', 'method', 'api'];
const HOME = { tab: 'landing', date: null, lead: 1 };

function readHash() {
  const h = window.location.hash.replace(/^#\/?/, '');
  if (!h || h === 'home') return HOME;
  const [tab, date, lead] = h.split('/');
  return {
    tab: TABS.includes(tab) ? tab : 'landing',
    date: /^\d{4}-\d{2}-\d{2}$/.test(date || '') ? date : null,
    lead: /^[1-5]$/.test(lead || '') ? Number(lead) : 1,
  };
}

export default function App() {
  const initial = useMemo(() => readHash(), []);
  const [tab, setTab] = useState(initial.tab);
  const [session, setSession] = useState({ date: initial.date, lead: initial.lead });
  const [mapLayer, setMapLayer] = useState('category');
  const [health, setHealth] = useState(null);
  const [toast, setToast] = useState(null);
  const [about, setAbout] = useState(false);

  useEffect(() => {
    const poll = () => api.health().then(setHealth).catch(() => setHealth(null));
    poll();
    const t = setInterval(poll, 60000);
    return () => clearInterval(t);
  }, []);

  // The hash is the shareable link: #map/2019-07-26/3 opens that view directly.
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

  useEffect(() => { window.scrollTo(0, 0); }, [tab]);

  const onSession = useCallback((patch) => setSession((s) => ({ ...s, ...patch })), []);
  const openMap = useCallback((layer) => { setMapLayer(layer || 'category'); setTab('map'); }, []);

  useEffect(() => {
    const onKey = (e) => {
      const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName);
      if (typing) return;
      if (e.key === '?') setToast('Shortcuts: ← → days · 1-5 lead · n/p significant day · l map layer · b bulletin · s story mode · h home · g dashboard · m map');
      if (e.key === 'h') setTab('landing');
      if (e.key === 'g') setTab('dashboard');
      if (e.key === 'm') setTab('map');
      if (e.key === 'l') window.dispatchEvent(new CustomEvent('monsooniq:cycle-layer'));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  if (tab === 'landing') {
    return (
      <Landing
        onEnter={(target) => { setSession((s) => ({ ...s, lead: 1 })); setTab(target || 'dashboard'); }}
        onTab={setTab}
      />
    );
  }

  return (
    <div className={`app2 tab-${tab}`}>
      <AppHeader tab={tab} onTab={setTab} health={health} onAbout={() => setAbout(true)} />

      <main className="app-main">
        {tab === 'dashboard' && (
          <Dashboard session={session} onSession={onSession} onTab={setTab} onOpenMap={openMap} />
        )}
        {tab === 'map' && (
          <ForecastMap session={session} onSession={onSession} initialLayer={mapLayer} onTab={setTab} />
        )}
        {tab === 'today' && (
          <div className="page wide">
            <Today session={session} onSession={onSession} onToast={setToast} onTab={setTab} />
          </div>
        )}
        {tab === 'skill' && <div className="page wide"><SkillLab /></div>}
        {tab === 'method' && <div className="page wide"><Method /></div>}
        {tab === 'data' && <DataSources />}
        {tab === 'api' && <div className="page wide"><ApiView /></div>}
      </main>

      {tab !== 'map' && (
        <footer className="footer2">
          {health?.profile === 'real'
            ? 'Forecasts verified against IMD 0.25° gridded rainfall on held-out seasons. '
            : 'Demo archive: reproducible synthetic benchmark (not official IMD/NCMRWF data) — see Data Sources to switch to IMD observations. '}
          Research prototype for SIH 2026 · PS 26080 (MoES / NCMRWF); not for public warnings.
          {' '}Press <span className="mono">?</span> for shortcuts.
        </footer>
      )}

      {about && <AboutModal onClose={() => setAbout(false)} />}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
