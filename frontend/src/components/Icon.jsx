import React from 'react';

/**
 * Minimal line icons (Feather-style: 24px grid, 2px round stroke), inlined so the
 * console needs no icon package. Same visual family SAGAR uses via react-icons/fi.
 */
const PATHS = {
  today: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  skill: <><path d="M3 3v18h18" /><path d="M7 15l4-4 3 3 5-6" /></>,
  method: <><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" /></>,
  api: <><path d="M16 18l6-6-6-6" /><path d="M8 6l-6 6 6 6" /></>,
  info: <><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" /></>,
  back: <><path d="M19 12H5" /><path d="M12 19l-7-7 7-7" /></>,
  arrow: <><path d="M5 12h14" /><path d="M12 5l7 7-7 7" /></>,
  close: <><path d="M18 6L6 18M6 6l12 12" /></>,
  globe: <><circle cx="12" cy="12" r="10" /><path d="M2 12h20" /><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" /></>,
  database: <><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></>,
  rain: <><path d="M20 16.2A4.5 4.5 0 0 0 17.5 8h-1.8A7 7 0 1 0 4 14.9" /><path d="M8 19v2M8 13v2M16 19v2M16 13v2M12 21v2M12 15v2" /></>,
};

export default function Icon({ name, size = 16, className = '', style }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      className={`icon ${className}`} style={style} aria-hidden="true" focusable="false"
    >
      {PATHS[name]}
    </svg>
  );
}
