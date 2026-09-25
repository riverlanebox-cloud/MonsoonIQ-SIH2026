import React, { useMemo, useState } from 'react';
import { CATEGORY_META, categoryBasis, fmt, pct } from '../lib/format';

const COLUMNS = [
  { id: 'district_name', label: 'District', type: 'text' },
  { id: 'state_name', label: 'State', type: 'text' },
  { id: 'category', label: 'Category', type: 'cat', rank: { green: 0, yellow: 1, orange: 2, red: 3 } },
  { id: 'regime', label: 'Regime', type: 'text' },
  { id: 'raw_mm', label: 'Raw', type: 'num', digits: 1, unit: 'mm' },
  { id: 'corrected_mm', label: 'Corrected', type: 'num', digits: 1, unit: 'mm' },
  { id: 'adjustment_mm', label: 'Δ', type: 'num', digits: 1, unit: 'mm', signed: true },
  { id: 'p90_mm', label: 'P90', type: 'num', digits: 0, unit: 'mm' },
  { id: 'p_heavy', label: 'P(≥64.5)', type: 'num', digits: 2 },
  { id: 'p_very_heavy', label: 'P(≥115.6)', type: 'num', digits: 2 },
];

export default function DistrictTable({ districts, selectedId, onSelect, filter, onFilterChange }) {
  const [sort, setSort] = useState({ id: 'category', dir: 'desc' });

  const arranged = useMemo(() => {
    const rows = [...(districts || [])];
    const col = COLUMNS.find((c) => c.id === sort.id);
    rows.sort((a, b) => {
      let av = a[sort.id];
      let bv = b[sort.id];
      if (col?.rank) { av = col.rank[av]; bv = col.rank[bv]; }
      if (typeof av === 'string') return sort.dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      const d = (bv ?? -Infinity) - (av ?? -Infinity);
      return sort.dir === 'asc' ? -d : d;
    });
    return rows;
  }, [districts, sort]);

  const toggle = (id) => setSort((s) => ({ id, dir: s.id === id && s.dir === 'desc' ? 'asc' : 'desc' }));

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">
          Warning table · {arranged.length} districts
          <span className="muted small" style={{ marginLeft: 6, fontWeight: 400 }}>
            sorted by {COLUMNS.find((c) => c.id === sort.id)?.label.toLowerCase()} ({sort.dir})
          </span>
        </span>
        <div className="cb-group">
          <input type="search" value={filter} placeholder="filter district or state"
                 onChange={(e) => onFilterChange(e.target.value)} />
        </div>
      </div>
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th key={c.id} className={c.type === 'num' ? 'num' : ''} onClick={() => toggle(c.id)}>
                  {c.label}{sort.id === c.id ? (sort.dir === 'desc' ? ' ▼' : ' ▲') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {arranged.map((d) => (
              <tr key={d.district_id} aria-selected={d.district_id === selectedId}
                  onClick={() => onSelect(d.district_id)}>
                <td><b>{d.district_name}</b></td>
                <td className="muted">{d.state_name}</td>
                <td>
                  <span className={`cat-chip cat-${d.category}`}
                        title={`${CATEGORY_META[d.category].label} — ${CATEGORY_META[d.category].action}\nWhy: ${categoryBasis(d).text}`}>
                    {CATEGORY_META[d.category].short}
                  </span>
                </td>
                <td className="small">{d.regime}</td>
                <td className="num">{fmt(d.raw_mm)}</td>
                <td className="num"><b>{fmt(d.corrected_mm)}</b></td>
                <td className="num" style={{ color: d.adjustment_mm > 0 ? 'var(--green)' : d.adjustment_mm < 0 ? 'var(--red)' : undefined }}>
                  {d.adjustment_mm > 0 ? '+' : ''}{fmt(d.adjustment_mm)}
                </td>
                <td className="num">{fmt(d.p90_mm, 0)}</td>
                <td className="num">{pct(d.p_heavy, 0)}</td>
                <td className="num">{pct(d.p_very_heavy, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
