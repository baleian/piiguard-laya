(function () {
  const P = window.P;
  if (!P.init('dashboard')) return;
  const S = P.S, el = P.el, td = P.td;
  const names = P.models().sort((a, b) => String(S.models[b].model.benched_at || '').localeCompare(String(S.models[a].model.benched_at || '')));
  if (!names.length) { P.empty(document.querySelector('.wrap'), '평가한 모델이 없습니다. make bench MODEL=laya-multilingual'); return; }
  const OPS = ['recall95', 'fpr05', 'default'].filter((o) => names.some((n) => S.models[n].calibration.thresholds[o] != null));
  document.getElementById('ops-desc').textContent = '운영 기준: ' + OPS.map((o) => `${P.opName(o)} — ${P.opDesc(o)}`).join(' · ');

  const modelCell = (n) => {
    const d = el('div'), a = el('a', null, n); a.href = 'evals.html#model=' + encodeURIComponent(n);
    d.append(a, el('small', 'd', P.kindLabel(S.models[n]) + (P.baseOf(n) ? ' · 기준 ' + P.baseOf(n) : '')));
    return d;
  };
  // 열마다 가장 좋은 값 강조
  const best = (vals, up) => { const xs = vals.filter((v) => v != null && !isNaN(v)); return xs.length > 1 ? (up ? Math.max(...xs) : Math.min(...xs)) : null; };
  const tint = (v, lo, hi, up) => {
    if (v == null || hi === lo) return '';
    const g = up ? (v - lo) / (hi - lo) : (hi - v) / (hi - lo);
    return `color-mix(in srgb, var(--accent) ${Math.round(4 + g * 22)}%, transparent)`;
  };

  // 주요 지표
  const cols = [['auroc', null, 'AUROC', 'score', true]];
  OPS.forEach((o) => { cols.push(['recall', o, '탐지율', 'rate', true], ['fpr', o, '오탐률', 'rate', false], ['accuracy', o, '정확도', 'rate', true]); });
  cols.push(['top1', null, '유형 맞힘', 'rate', true], ['p50', null, '지연 p50', 'ms', false]);
  const val = (n, [f, o]) => {
    const m = S.models[n], A = m.pooled.ALL;
    if (f === 'auroc') return A && A.threshold_free.auroc;
    if (f === 'top1') return m.classification && m.classification.top1_accuracy;
    if (f === 'p50') { const l = m.latency && m.latency.results && m.latency.results[0]; return l ? l.p50_ms : null; }
    return A && A.ops[o] ? A.ops[o][f] : null;
  };
  const t = document.getElementById('main');
  const h1 = el('tr'), h2 = el('tr');
  const th = (txt, span, rs) => { const x = el('th', null, txt); if (span) x.colSpan = span; if (rs) x.rowSpan = rs; return x; };
  h1.append(th('모델', 0, 2), th('AUROC', 0, 2));
  OPS.forEach((o) => { const x = th(P.opName(o), 3); h1.append(x); });
  h1.append(th('유형 맞힘', 0, 2), th('지연 p50', 0, 2));
  OPS.forEach(() => h2.append(th('탐지율'), th('오탐률'), th('정확도')));
  const thead = el('thead'); thead.append(h1, h2);
  const bests = cols.map((c) => best(names.map((n) => val(n, c)), c[4]));
  const tbody = el('tbody');
  names.forEach((n) => {
    const tr = el('tr'); tr.append(td(modelCell(n)));
    cols.forEach((c, i) => {
      const v = val(n, c), cell = td(c[3] === 'ms' ? (v == null ? '–' : (+v).toFixed(1) + ' ms') : P.fmt(v, c[3]), 'num');
      if (v != null && bests[i] != null && v === bests[i]) cell.classList.add('best');
      tr.append(cell);
    });
    tbody.append(tr);
  });
  t.replaceChildren(thead, tbody);

  // 데이터셋 목록 (모델들이 평가한 것의 합집합)
  const dsNames = [...new Set(names.flatMap((n) => Object.keys(S.models[n].datasets)))];
  const dsHead = (first) => {
    const tr = el('tr'); tr.append(el('th', null, first));
    dsNames.forEach((d) => { const x = el('th', null, P.dsTitle(d)); x.append(el('small', null, d)); tr.append(x); });
    const h = el('thead'); h.append(tr); return h;
  };
  const heatTable = (id, getter, up, fmt) => {
    const tb = el('tbody');
    const ranges = dsNames.map((d) => { const xs = names.map((n) => getter(n, d)).filter((v) => v != null); return [Math.min(...xs), Math.max(...xs)]; });
    names.forEach((n) => {
      const tr = el('tr'); tr.append(td(modelCell(n)));
      dsNames.forEach((d, i) => {
        const v = getter(n, d), c = td(v == null ? '–' : fmt(v), 'num');
        if (v != null && names.length > 1) c.style.background = tint(v, ranges[i][0], ranges[i][1], up);
        tr.append(c);
      });
      tb.append(tr);
    });
    document.getElementById(id).replaceChildren(dsHead('모델'), tb);
  };
  heatTable('ds-auroc', (n, d) => { const m = S.models[n].datasets[d]; return m && m.n_neg > 0 ? m.threshold_free.auroc : null; }, true, P.f3);

  // 운영 기준별 데이터셋 탐지율·오탐률
  const box = document.getElementById('by-op');
  const parts = [];
  OPS.forEach((o) => {
    const h = el('h2'); h.append(`데이터셋별 탐지율 · 오탐률 `, el('span', 'small muted', '· ' + P.opName(o)));
    const wrap = el('div', 'tw'), tbl = el('table');
    const tb = el('tbody');
    names.forEach((n) => {
      const tr = el('tr'); tr.append(td(modelCell(n)));
      dsNames.forEach((d) => {
        const m = S.models[n].datasets[d], c = el('td', 'num');
        if (!m) c.textContent = '–';
        else {
          const b = m.ops[o] || {}, x = el('div', 'cmp');
          x.append(el('span', null, '탐지 ' + P.pct(b.recall)));
          x.append(el('span', 'b', m.n_neg > 0 ? '오탐 ' + P.pct(b.fpr) : '오탐 해당 없음'));
          c.append(x);
        }
        tr.append(c);
      });
      tb.append(tr);
    });
    tbl.append(dsHead('모델'), tb); wrap.append(tbl);
    parts.push(h, wrap);
  });
  box.replaceChildren(...parts);

  // 유형별
  const catKeys = S.labels.category_order.filter((c) => c !== 'none' && names.some((n) => S.models[n].by_category[c]));
  const ch = el('tr'); ch.append(el('th', null, '모델'));
  catKeys.forEach((c) => ch.append(el('th', null, P.catName(c))));
  const chead = el('thead'); chead.append(ch);
  const cb = el('tbody');
  const op0 = OPS[0];
  const cr = catKeys.map((c) => { const xs = names.map((n) => { const r = S.models[n].by_category[c]; return r && r.ops[op0] ? r.ops[op0].recall : null; }).filter((v) => v != null); return [Math.min(...xs), Math.max(...xs)]; });
  names.forEach((n) => {
    const tr = el('tr'); tr.append(td(modelCell(n)));
    catKeys.forEach((c, i) => {
      const r = S.models[n].by_category[c], v = r && r.ops[op0] ? r.ops[op0].recall : null;
      const cell = td(v == null ? '–' : P.pct(v), 'num');
      if (v != null && names.length > 1) cell.style.background = tint(v, cr[i][0], cr[i][1], true);
      tr.append(cell);
    });
    cb.append(tr);
  });
  document.getElementById('cats').replaceChildren(chead, cb);
})();
