import React, { useState } from 'react';
import { FiSearch, FiCornerDownLeft, FiLoader } from 'react-icons/fi';

export const SUGGESTIONS = [
  'Heavy rain in Kerala tomorrow',
  'Red warnings day 3',
  'What regime is driving today?',
  'How much rain in Mumbai',
  'Where did the correction add most rain on the west coast',
  'Very heavy rain in Uttarakhand day 2',
];

/** Bottom-centred query bar (SAGAR-style) for plain-language questions over the forecast. */
export default function AskBar({ onAsk, busy }) {
  const [q, setQ] = useState('');
  const submit = (text) => {
    const v = (text ?? q).trim();
    if (!v) return;
    setQ(v);
    onAsk(v);
  };
  return (
    <div className="askbar">
      <div className="ask-suggest">
        {SUGGESTIONS.map((s) => (
          <button key={s} onClick={() => submit(s)}>{s}</button>
        ))}
      </div>
      <form className="ask-input" onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <FiSearch className="ask-icon" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask MonsoonIQ — e.g. “heavy rain in Odisha day 2”, “what regime is driving today?”"
          aria-label="Ask MonsoonIQ"
        />
        <button type="submit" className="ask-go" disabled={busy}>
          {busy ? <FiLoader className="spin" /> : <FiCornerDownLeft />} Ask
        </button>
      </form>
    </div>
  );
}
