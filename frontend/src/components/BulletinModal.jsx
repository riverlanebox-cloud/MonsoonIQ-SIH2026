import React, { useEffect, useState } from 'react';
import { api } from '../api';

/** Copy-pasteable SDMA bulletin — the deliverable a duty forecaster actually files. */
export default function BulletinModal({ date, lead, onClose }) {
  const [lang, setLang] = useState('en');
  const [data, setData] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let alive = true;
    api.bulletin(date, lead, lang).then((d) => alive && setData(d)).catch(() => alive && setData(null));
    return () => { alive = false; };
  }, [date, lead, lang]);

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const copy = async () => {
    if (!data?.text) return;
    try {
      await navigator.clipboard.writeText(data.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch { /* clipboard blocked - the textarea still allows manual copy */ }
  };

  return (
    <div className="bulletin-overlay" style={{
      position: 'fixed', inset: 0, background: 'rgba(4,10,16,.72)', backdropFilter: 'blur(3px)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
    }} onClick={onClose}>
      <div className="panel bulletin" style={{ width: 'min(880px, 100%)', maxHeight: '88vh', display: 'flex', flexDirection: 'column' }}
           onClick={(e) => e.stopPropagation()}>
        <div className="panel-head">
          <span className="panel-title">District warning bulletin · {date} · Day {lead}</span>
          <div className="cb-group">
            <div className="seg">
              <button aria-pressed={lang === 'en'} onClick={() => setLang('en')}>English</button>
              <button aria-pressed={lang === 'hi'} onClick={() => setLang('hi')}>हिंदी</button>
            </div>
            <button className="btn primary" onClick={copy}>{copied ? 'Copied' : 'Copy text'}</button>
            <button className="btn" onClick={() => window.print()}>Print</button>
            <button className="btn" onClick={onClose}>close</button>
          </div>
        </div>
        <div className="panel-body" style={{ overflow: 'auto' }}>
          {data ? (
            <>
              <div className="small muted" style={{ marginBottom: 6 }}>
                {data.districts_included} districts reach a warning category. Wording follows the
                IMD impact-based warning scale; deviation from raw model value is stated so the
                duty forecaster can judge the correction.
              </div>
              <pre className="mono" style={{ whiteSpace: 'pre-wrap', fontSize: 12, margin: 0 }}>{data.text}</pre>
            </>
          ) : <div className="skeleton" style={{ height: 200 }} />}
        </div>
      </div>
    </div>
  );
}
