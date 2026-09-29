import React from 'react';
import {
  FiHome, FiGrid, FiMap, FiActivity, FiBarChart2, FiDatabase, FiBookOpen, FiCode, FiInfo,
} from 'react-icons/fi';

export const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: FiGrid },
  { id: 'map', label: 'Forecast Map', icon: FiMap },
  { id: 'today', label: 'Operations', icon: FiActivity },
  { id: 'skill', label: 'Verification', icon: FiBarChart2 },
  { id: 'data', label: 'Data Sources', icon: FiDatabase },
  { id: 'method', label: 'Method', icon: FiBookOpen },
  { id: 'api', label: 'API', icon: FiCode },
];

/** Fixed, blurred top bar with icon navigation (one row, collapses to icons on narrow screens). */
export default function AppHeader({ tab, onTab, health, onAbout }) {
  const real = health?.profile === 'real';
  return (
    <header className="app-header">
      <div className="app-header-inner">
        <button className="logo" onClick={() => onTab('landing')} title="Home">
          <LogoMark />
          <span className="logo-text">Monsoon<span>IQ</span></span>
        </button>

        <nav className="app-nav" aria-label="Main">
          <button className="nav-item" onClick={() => onTab('landing')} title="Home">
            <FiHome /><span>Home</span>
          </button>
          {NAV.map((n) => (
            <button key={n.id} className={`nav-item${tab === n.id ? ' active' : ''}`}
                    onClick={() => onTab(n.id)} aria-current={tab === n.id ? 'page' : undefined}
                    title={n.label}>
              <n.icon /><span>{n.label}</span>
            </button>
          ))}
          <button className="nav-item" onClick={onAbout} title="About"><FiInfo /><span>About</span></button>
        </nav>

        <div className="header-status">
          <span className={`status-pill ${health?.status === 'healthy' ? 'ok' : 'bad'}`}>
            <span className="dot" />{health?.status === 'healthy' ? 'Engine live' : 'Engine offline'}
          </span>
          <span className={`status-pill ${real ? 'real' : 'synth'}`}
                title={real ? 'IMD-observed archive with archived NWP forecasts'
                  : 'Reproducible synthetic benchmark - switch with MONSOONIQ_PROFILE=real'}>
            {real ? 'IMD observed data' : 'Synthetic benchmark'}
          </span>
        </div>
      </div>
    </header>
  );
}

export function LogoMark({ size = 30 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="logo-mark">
      <defs>
        <linearGradient id="lg-cloud" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#00d4ff" />
          <stop offset="1" stopColor="#20b2aa" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="8" fill="#0d1b2a" stroke="url(#lg-cloud)" strokeWidth="1.5" />
      <path d="M9 17.5a4.5 4.5 0 0 1 1.2-8.8A6 6 0 0 1 21.6 10a4 4 0 0 1 .9 7.5Z" fill="url(#lg-cloud)" />
      <path d="M11 21l-1.2 3M16 21l-1.2 3M21 21l-1.2 3" stroke="#00d4ff" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}
