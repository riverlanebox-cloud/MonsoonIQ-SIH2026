// Entry point used only by the jsdom smoke test (see scripts/smoke.mjs).
import { createRoot } from 'react-dom/client';
import React from 'react';
import App from './src/App.jsx';

export async function mount(el) {
  createRoot(el).render(React.createElement(App));
}

// Exposed for the harness: the console mirrors one backend rule (which criterion
// set a district's warning category), and the smoke run checks the mirror against
// the API's own labels rather than trusting it.
import * as format from './src/lib/format.js';
window.__format = format;
