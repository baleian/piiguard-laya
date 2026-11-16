(function () {
  const P = window.P;
  if (!P.init('evals')) return;
  const S = P.S, el = P.el, td = P.td, $ = (id) => document.getElementById(id);
  const byLatest = (a, c) => String(S.models[c].model.benched_at || '').localeCompare(String(S.models[a].model.benched_at || ''));
  const names = P.models().sort(byLatest);
  if (!names.length) { P.empty(document.querySelector('.wrap'), '평가한 모델이 없습니다. make bench MODEL=base/laya-multilingual'); return; }
  const msel = $('m'), bsel = $('b');
  P.options(msel, names.map((n) => [n, n]));
  const h = P.hash();
  let m = names.includes(h.model) ? h.model : names[0];
  const cardBase = (n) => P.baseOf(n);
  const defaultBase = (n) => { const x = cardBase(n); return x && S.models[x] && x !== n ? x : ''; };
  let b = h.base !== undefined && (h.base === '' || (S.models[h.base] && h.base !== m)) ? h.base : defaultBase(m);

  function fillBase() {
    P.options(bsel, [['', '(비교 안 함)'], ...names.filter((n) => n !== m).map((n) => [n, n + (n === cardBase(m) ? ' · 모델 카드의 기준 모델' : '')])], b);
    const cb = cardBase(m);
    $('bnote').textContent = cb && !S.models[cb] ? `모델 카드의 기준 모델 ${cb} 은 아직 평가되지 않았습니다 (make bench MODEL=${cb}).`
      : cb ? (b === cb ? '모델 카드에 기록된 기준 모델과 비교합니다.' : '') : '';
  }
  msel.value = m;
  fillBase();
  msel.onchange = () => { m = msel.value; b = defaultBase(m); fillBase(); render(); };
  bsel.onchange = () => { b = bsel.value; fillBase(); render(); };
  window.addEventListener('hashchange', () => {
    const x = P.hash();
    if (x.model && S.models[x.model] && (x.model !== m || (x.base || '') !== b)) { m = x.model; b = x.base && S.models[x.base] ? x.base : ''; msel.value = m; fillBase(); render(); }
  });

  // ------------------------------------------------------------------ 도우미
  function bigCmp(bv, tv, kind, up) {
    const box = el('div', 'big');
    if (tv == null) { box.append(el('span', 'to', '–')); return box; }
    if (bv != null) box.append(el('span', 'from', P.fmt(bv, kind)), el('span', 'from', '→'));
    box.append(el('span', 'to', P.fmt(tv, kind)));
    if (bv != null) box.append(P.chip(tv - bv, kind, up));
    return box;
  }
  function heroCard(label, desc, bv, tv, kind, up) {
    const c = el('div', 'card'); c.append(el('span', 'label', label), el('span', 'desc', desc), bigCmp(bv, tv, kind, up)); return c;
  }
  const g = (o, ...ks) => ks.reduce((a, k) => (a == null ? null : a[k]), o);
  function bell(name, bv, tv, up, tip, host) {
    const box = el('div'), head = el('div', 'bell-head'), vals = el('span', 'vals');
    if (bv != null && tv != null) vals.append(el('span', 'muted', P.pct(bv)), el('span', 'muted', '→'), el('b', null, P.pct(tv)), P.chip(tv - bv, 'rate', up));
    else vals.append(el('b', null, P.pct(tv != null ? tv : bv)));
    head.append(el('span', null, name), vals);
    const tr = el('div', 'track');
    if (bv != null && tv != null) { const s = el('div', 'seg'); s.style.left = Math.min(bv, tv) * 100 + '%'; s.style.width = Math.abs(tv - bv) * 100 + '%'; tr.append(s); }
    if (bv != null) { const k = el('div', 'mk base'); k.style.left = bv * 100 + '%'; tr.append(k); }
    if (tv != null) { const k = el('div', 'mk tuned'); k.style.left = tv * 100 + '%'; tr.append(k); }
    tr.setAttribute('role', 'img');
    tr.setAttribute('aria-label', `${name}: ` + (bv != null ? `기준 모델 ${P.pct(bv)}, ` : '') + `이 모델 ${P.pct(tv)}`);
    const show = (ev) => {
      tip.replaceChildren(el('div', null, name));
      if (bv != null) tip.append(el('div', 'muted', `기준 모델 ${b}: ${P.pct(bv)}`));
      tip.append(el('div', null, `${m}: ${P.pct(tv)}`));
      tip.hidden = false;
      const r = host.getBoundingClientRect();
      tip.style.left = Math.max(0, Math.min(ev.clientX - r.left + 12, r.width - tip.offsetWidth)) + 'px';
      tip.style.top = (ev.clientY - r.top + 14) + 'px';
    };
    tr.addEventListener('pointermove', show); tr.addEventListener('pointerenter', show);
    tr.addEventListener('pointerleave', () => { tip.hidden = true; });
    box.append(head, tr);
    return box;
  }
  function opHead(OPS, first) {
    const tr = el('tr'); tr.append(el('th', null, first));
    OPS.forEach((o) => { const x = el('th', null, P.opName(o)); x.append(el('small', null, P.opDesc(o))); tr.append(x); });
    const hd = el('thead'); hd.append(tr); return hd;
  }
  function grpRow(text, span) { const tr = el('tr', 'grp'), c = td(text); c.colSpan = span; tr.append(c); return tr; }
  function lblCell(title, sub) { const d = el('td', 'lbl', title); if (sub) d.append(el('small', null, sub)); return d; }

  // ------------------------------------------------------------------ 그리기
  function render() {
    P.setHash({ model: m, base: b });
    const M = S.models[m], B = b ? S.models[b] : null, OPS = P.ops(M);
    $('title').textContent = `모델 비교 — ${m}`;
    const A = M.pooled.ALL || {};
    $('lede').textContent = `평가 문장 ${P.int(A.n)}건(정상 문장이 있는 데이터셋 ${Object.values(M.datasets).filter((d) => d.n_neg > 0).length}개)` +
      (B ? `에서 ${m} 을 기준 모델 ${b} 와 같은 기준으로 비교합니다.` : `에서 ${m} 의 결과를 보여 줍니다. 비교하려면 기준 모델을 고르세요.`) +
      ' 모든 숫자는 학습과 임계값 결정에 쓰지 않은 평가 문장(test)으로 계산했습니다.';
    $('pred-link').href = 'predictions.html#model=' + encodeURIComponent(m);
    const lg = $('legend');
    lg.replaceChildren();
    if (B) { const a = el('span'); a.append(el('span', 'dot base'), `기준 모델 (${b})`); lg.append(a); }
    const t1 = el('span'); t1.append(el('span', 'dot tuned'), `이 모델 (${m})`); lg.append(t1);
    if (B) { const c = el('span'); c.append('변화 표시: ', P.chip(0.1, 'rate', true), ' ', P.chip(-0.1, 'rate', true), ' ', P.chip(0, 'rate', true)); lg.append(c); }

    // 판정
    const gate = B ? g(S.compare, m, b) : null;
    const v = $('verdict');
    v.replaceChildren();
    if (gate) {
      const failed = gate.checks.filter((c) => !c.pass);
      v.append(el('span', 'pill ' + (gate.pass ? 'ok' : 'bad'), gate.pass ? '회귀 검사 통과' : '회귀 검사 미달'),
        el('span', 'muted', `기준 모델 대비 검사 ${gate.checks.length}개 중 ${gate.checks.length - failed.length}개 통과` + (failed.length ? ` · ${failed.length}개 미달 (아래 '회귀 검사')` : '')));
    }

    // 모델 정보
    const mi = M.model, card = mi.card || null, kv = $('info');
    kv.replaceChildren();
    const row = (k, val) => { kv.append(el('dt', null, k)); const d = el('dd'); if (val instanceof Node) d.append(val); else d.textContent = val == null || val === '' ? '–' : val; kv.append(d); };
    row('종류', P.kindLabel(M));
    row('위치', mi.path);
    if (mi.kind === 'base') row('출처', P.sourceText(mi.source) + (mi.license ? ` (${mi.license})` : ''));
    if (card) {
      row('기준 모델', card.base ? card.base.id || card.base.name : '–');
      row('계보', [...(card.lineage || []), m].join(' → '));
      row('학습 데이터', (card.training_data || []).map((t) => `${t.name} · 사용 ${P.int(t.used_rows)}건 / 전체 ${P.int(t.rows)}건`).join(', '));
      row('평가 문장 겹침 제외', P.int(g(card, 'leakage_check', 'excluded_overlap_with_eval') || 0) + '건');
      row('만든 날', P.date(card.created_at));
    }
    row('평가한 날', P.date(mi.benched_at));
    row('평가 장치', [g(mi, 'runtime', 'device'), g(mi, 'runtime', 'gpu')].filter(Boolean).join(' · '));
    row('임계값', OPS.map((o) => `${P.opName(o)} ${P.thr(M.calibration.thresholds[o])}`).join(' · '));
    const te = $('train-eval');
    te.replaceChildren(el('span', 'label', '학습 중 평가'));
    const tev = card && card.train_eval;
    if (tev && tev.before && tev.after) {
      te.append(el('span', 'desc', `${tev.source} ${P.int(tev.rows || tev.items)}문장. 유형(개인정보 없음 포함)을 맞힌 비율과 확률 보정 오차(ECE · Brier, 낮을수록 좋음)입니다.`));
      const tb = el('table'), hd = el('thead'), hr = el('tr');
      ['', '학습 전', '학습 후'].forEach((x) => hr.append(el('th', null, x))); hd.append(hr);
      const body = el('tbody');
      [['정답률', 'accuracy', true], ['ECE', 'ece', false], ['Brier', 'brier', false]].forEach(([n, k, up]) => {
        const r = el('tr'); r.append(td(n), td(k === 'accuracy' ? P.pct(tev.before[k]) : P.f3(tev.before[k]), 'num'));
        const c = el('td', 'num'); c.append(k === 'accuracy' ? P.pct(tev.after[k]) : P.f3(tev.after[k]), ' ', P.chip(tev.after[k] - tev.before[k], k === 'accuracy' ? 'rate' : 'score', up)); r.append(c);
        body.append(r);
      });
      tb.append(hd, body); te.append(tb);
      if (card.epoch_loss) te.append(el('span', 'desc', '에폭별 학습 손실: ' + card.epoch_loss.map((x) => (+x).toFixed(3)).join(' → ')));
    } else te.append(el('span', 'desc', mi.kind === 'base' ? 'Base 모델이라 학습 기록이 없습니다.' : '학습 기록이 없습니다.'));

    // 핵심 결과
    const hero = [];
    const pa = (X, ...ks) => (X ? g(X.pooled.ALL, ...ks) : null);
    hero.push(heroCard('구분 능력 (AUROC)', '개인정보 문장을 정상 문장보다 높게 점수 매기는 능력', pa(B, 'threshold_free', 'auroc'), pa(M, 'threshold_free', 'auroc'), 'score', true));
    if (OPS.includes('recall95')) hero.push(heroCard('오탐률 · 누락 최소화 기준', '개인정보의 95%를 잡도록 맞췄을 때 정상 문장을 잘못 막는 비율', pa(B, 'ops', 'recall95', 'fpr'), pa(M, 'ops', 'recall95', 'fpr'), 'rate', false));
    if (OPS.includes('fpr05')) hero.push(heroCard('탐지율 · 오탐 최소화 기준', '정상 문장 오차단을 5% 수준으로 묶었을 때 잡아내는 개인정보 비율', pa(B, 'ops', 'fpr05', 'recall'), pa(M, 'ops', 'fpr05', 'recall'), 'rate', true));
    hero.push(heroCard('개인정보 유형 맞힘', '개인정보 문장에서 종류(주민번호·계좌·연락처 등)를 맞힌 비율', B ? g(B, 'classification', 'top1_accuracy') : null, g(M, 'classification', 'top1_accuracy'), 'rate', true));
    $('hero').replaceChildren(...hero);

    // 운영 기준 카드
    $('ops').replaceChildren(...OPS.map((o) => {
      const c = el('div', 'card');
      c.append(el('h3', null, P.opName(o)), el('span', 'desc', P.opDesc(o)));
      c.append(el('span', 'small muted', '임계값 · ' + (B ? `기준 ${P.thr(B.calibration.thresholds[o])} · ` : '') + `이 모델 ${P.thr(M.calibration.thresholds[o])}`));
      const bells = el('div', 'bells'), tip = el('div', 'tip'); tip.hidden = true;
      [['accuracy', '정확도'], ['recall', '탐지율'], ['fpr', '오탐률'], ['precision', '정밀도']].forEach(([f, name]) =>
        bells.append(bell(name, pa(B, 'ops', o, f), pa(M, 'ops', o, f), P.up[f], tip, bells)));
      bells.append(tip);
      c.append(bells);
      return c;
    }));

    // 언어별
    const lt = el('tbody');
    [['ko', '한국어'], ['en', '영어']].forEach(([k, name]) => {
      const X = M.pooled[k]; if (!X) return;
      lt.append(grpRow(`${name} · 평가 문장 ${P.int(X.n)}건`, OPS.length + 1));
      [['accuracy', '정확도'], ['recall', '탐지율'], ['fpr', '오탐률']].forEach(([f, fn]) => {
        const r = el('tr'); r.append(lblCell(fn));
        OPS.forEach((o) => r.append(td(P.cmp(B ? g(B.pooled[k], 'ops', o, f) : null, g(X, 'ops', o, f), 'rate', P.up[f]))));
        lt.append(r);
      });
    });
    $('lang').replaceChildren(opHead(OPS, '언어 · 지표'), lt);

    // 데이터셋별
    const dsNames = Object.keys(M.datasets);
    ['recall', 'fpr'].forEach((f) => {
      const tb = el('tbody');
      dsNames.forEach((d) => {
        const X = M.datasets[d], Y = B && B.datasets[d];
        const r = el('tr');
        r.append(lblCell(X.title || d, `${d} · ${f === 'recall' ? '개인정보 문장 ' + P.int(X.n_pos) : '정상 문장 ' + P.int(X.n_neg)}건`));
        OPS.forEach((o) => r.append(td(f === 'fpr' && X.n_neg === 0 ? el('span', 'na', '해당 없음') : P.cmp(g(Y, 'ops', o, f), g(X, 'ops', o, f), 'rate', P.up[f]))));
        tb.append(r);
      });
      $('ds-' + f).replaceChildren(opHead(OPS, '데이터셋'), tb);
    });

    // 유형별
    const ct = el('tbody');
    S.labels.category_order.filter((c) => c !== 'none' && M.by_category[c]).forEach((c) => {
      const X = M.by_category[c], Y = B && B.by_category[c];
      const r = el('tr'); r.append(lblCell(P.catName(c), `개인정보 문장 ${P.int(X.n)}건`));
      OPS.forEach((o) => r.append(td(P.cmp(g(Y, 'ops', o, 'recall'), g(X, 'ops', o, 'recall'), 'rate', true))));
      ct.append(r);
    });
    if (M.classification && M.classification.n) {
      const r = el('tr'); r.append(lblCell('유형 맞힘 (운영 기준과 무관)', `개인정보 문장 ${P.int(M.classification.n)}건`));
      const c = td(P.cmp(B ? g(B, 'classification', 'top1_accuracy') : null, M.classification.top1_accuracy, 'rate', true)); c.colSpan = OPS.length;
      r.append(c); ct.append(r);
    }
    $('cats').replaceChildren(opHead(OPS, '개인정보 유형'), ct);

    // 변조
    const order = Object.keys(S.labels.variants);
    const vkeys = [...new Set([...Object.keys(M.by_variant || {}), ...Object.keys((B && B.by_variant) || {})])];
    $('var-sec').hidden = !vkeys.length;
    $('vars').replaceChildren(...['ko', 'en'].filter((lang) => vkeys.some((k) => k.startsWith(lang + '/'))).map((lang) => {
      const box = el('div', 'sec'), tw = el('div', 'tw'), tbl = el('table'), tb = el('tbody');
      [['pii', '개인정보 문장 · 탐지율 (높을수록 좋음)', true], ['negative', '정상 문장 · 오탐률 (낮을수록 좋음)', false]].forEach(([kind, title, up]) => {
        const ks = vkeys.filter((k) => k.startsWith(lang + '/') && k.endsWith('/' + kind)).sort((x, y) => order.indexOf(x.split('/')[1]) - order.indexOf(y.split('/')[1]));
        if (!ks.length) return;
        tb.append(grpRow(title, OPS.length + 1));
        ks.forEach((k) => {
          const X = (M.by_variant || {})[k], Y = B && (B.by_variant || {})[k], vn = k.split('/')[1];
          const r = el('tr'); r.append(lblCell(P.varName(vn), `${P.int((X || Y).n)}건`));
          OPS.forEach((o) => r.append(td(P.cmp(Y ? Y.ops[o] : null, X ? X.ops[o] : null, 'rate', up))));
          tb.append(r);
        });
      });
      tbl.append(opHead(OPS, '변형'), tb); tw.append(tbl);
      box.append(el('h3', null, lang === 'ko' ? '한국어' : '영어'), tw);
      return box;
    }));

    // 회귀 검사
    const rules = S.gate || {};
    $('gate-desc').textContent = `기준 모델보다 어느 영역에서도 나빠지지 않았는지 검사합니다. 기준: 전체 탐지율 ${(rules.max_recall_drop || 0) * 100}%p, 데이터셋별 ${(rules.max_dataset_recall_drop || 0) * 100}%p, 변조 유형별 ${(rules.max_variant_recall_drop || 0) * 100}%p 이상 떨어지지 않고, 전체 오탐률이 ${(rules.max_fpr_increase || 0) * 100}%p 이상 늘지 않을 것. 비교는 ${P.opName(gate ? gate.op : rules.op)} 기준에서 각 모델의 임계값으로 합니다.`;
    const gbox = $('gate'), gall = $('gate-all');
    const RATE = /^(overall_recall|overall_fpr|dataset_recall|variant_recall)/;
    let curCheck = '';
    const fmtv = (x) => (x == null ? '–' : RATE.test(curCheck) ? P.pct(x) : curCheck === 'overall_auroc' ? P.f3(x) : typeof x === 'number' ? P.int(x) : String(x));
    const fmtc = (c, x) => { curCheck = c.check; return fmtv(x); };
    const vname = (c) => (c.check.startsWith('variant_recall:') ? (() => { const [l, vv] = c.check.split(':')[1].split('/'); return `변조 입력 탐지율 유지: ${l === 'ko' ? '한국어' : '영어'} · ${P.varName(vv)}`; })() : c.name);
    if (gate) {
      const failed = gate.checks.filter((c) => !c.pass);
      const head = el('div', 'row');
      head.append(el('span', 'pill ' + (gate.pass ? 'ok' : 'bad'), gate.pass ? '통과' : '미달'), el('span', null, `검사 ${gate.checks.length}개 중 ${gate.checks.length - failed.length}개 통과`));
      if (gate.mcnemar) head.append(el('span', 'small muted', `McNemar exact p=${(+gate.mcnemar.p_value).toPrecision(3)} (기준 모델만 맞음 ${P.int(gate.mcnemar.a_only_correct)}건, 이 모델만 맞음 ${P.int(gate.mcnemar.b_only_correct)}건)`));
      const parts = [head];
      if (failed.length) {
        const ul = el('ul', 'list');
        failed.forEach((c) => { const li = el('li'); li.append(vname(c) + ': ', fmtc(c, c.baseline), ' → ', el('b', null, fmtc(c, c.candidate))); ul.append(li); });
        parts.push(el('p', 'small muted', '미달 항목 (기준 모델 → 이 모델):'), ul);
      }
      gbox.replaceChildren(...parts);
      const hd = el('thead'), hr = el('tr'); ['검사', '결과', '기준 모델', '이 모델', '기준'].forEach((x) => hr.append(el('th', null, x))); hd.append(hr);
      const tb = el('tbody');
      [...gate.checks].sort((x, y) => x.pass - y.pass).forEach((c) => { const r = el('tr'); r.append(td(vname(c)), td(c.pass ? '통과' : '미달', c.pass ? '' : 'best'), td(fmtc(c, c.baseline), 'num'), td(fmtc(c, c.candidate), 'num'), td(c.rule)); tb.append(r); });
      gall.replaceChildren(hd, tb);
    } else { P.empty(gbox, '비교할 기준 모델을 고르면 회귀 검사 결과를 보여 줍니다.'); gall.replaceChildren(); }

    // 속도
    const lat = el('tbody'), lh = el('thead'), lr = el('tr');
    ['모델', '장치', 'p50', 'p95', 'p99', '측정 요청 수'].forEach((x) => lr.append(el('th', null, x))); lh.append(lr);
    [[b, B], [m, M]].filter(([, X]) => X).forEach(([n, X]) => {
      const L = X.latency && X.latency.p50_ms != null ? X.latency : null;
      const r = el('tr'); r.append(td(n), td(L ? [L.gpu, L.device].filter(Boolean).join(' · ') : '–'));
      ['p50_ms', 'p95_ms', 'p99_ms'].forEach((k) => r.append(td(L ? (+L[k]).toFixed(1) + ' ms' : '–', 'num')));
      r.append(td(L ? P.int(L.n) : '–', 'num')); lat.append(r);
    });
    $('lat').replaceChildren(lh, lat);

    // 전문가용
    const dh = el('thead'), dr = el('tr');
    ['모델', '범위', '기준', '임계값', '정확도 (95% CI)', '탐지율 (95% CI)', '오탐률 (95% CI)', '정밀도', 'F1', 'F2', 'TP', 'FN', 'FP', 'TN'].forEach((x) => dr.append(el('th', null, x))); dh.append(dr);
    const db = el('tbody');
    const ci = (x, k) => `${P.pct(x[k])} (${P.pct(x[k + '_lo'])}–${P.pct(x[k + '_hi'])})`;
    [[m, M], [b, B]].filter(([, X]) => X).forEach(([n, X]) => {
      [['ALL', '전체'], ['ko', '한국어'], ['en', '영어']].forEach(([sk, sn]) => {
        const Q = X.pooled[sk]; if (!Q) return;
        P.ops(X).forEach((o) => {
          const z = Q.ops[o]; if (!z) return;
          const r = el('tr');
          [n, sn, P.opName(o), P.thr(X.calibration.thresholds[o]), ci(z, 'accuracy'), ci(z, 'recall'), ci(z, 'fpr'), P.pct(z.precision), P.f3(z.f1), P.f3(z.f2),
            P.int(z.tp), P.int(z.fn), P.int(z.fp), P.int(z.tn)].forEach((x, i) => r.append(td(x, i > 2 ? 'num' : '')));
          db.append(r);
        });
      });
    });
    $('detail').replaceChildren(dh, db);

    // 출처
    const used = new Set([...dsNames, ...(B ? Object.keys(B.datasets) : [])]);
    const src = S.datasets.filter((d) => used.has(d.name)).map((d) => `${d.title}: ${d.source || '–'} (${d.license || '라이선스 표기 없음'})`);
    $('foot').textContent = '데이터 출처 — ' + src.join(' · ');

    renderExamples(M, B, OPS);
  }

  // ------------------------------------------------------------------ 실제 사례 (문장 데이터는 이때 읽는다)
  let token = 0;
  function renderExamples(M, B, OPS) {
    const box = $('examples'), my = ++token;
    box.replaceChildren(el('p', 'muted', '사례를 불러오는 중…'));
    Promise.all([P.loadItems(m), B ? P.loadItems(b) : Promise.resolve(null)]).then(([pm, pb]) => {
      if (my !== token) return;
      const xs = P.rowsOf(pm), base = pb ? Object.fromEntries(P.rowsOf(pb).map((r) => [r.id, r])) : null;
      const tm = M.calibration.thresholds, tb = B ? B.calibration.thresholds : null;
      const groups = [];
      if (B && OPS.includes('recall95')) groups.push(['고쳐진 오탐 — 기준 모델은 막았지만 이 모델은 통과시킨 정상 문장', 'recall95',
        (x, y) => x.label === 0 && y && y.score >= tb.recall95 && x.score < tm.recall95, (x, y) => y.score - x.score]);
      if (B && OPS.includes('fpr05')) groups.push(['고쳐진 미검출 — 기준 모델은 놓쳤지만 이 모델은 잡아낸 개인정보', 'fpr05',
        (x, y) => x.label === 1 && y && y.score < tb.fpr05 && x.score >= tm.fpr05, (x, y) => x.score - y.score]);
      if (OPS.includes('recall95')) groups.push(['이 모델이 놓치는 개인정보 (누락 최소화 기준에서도 통과, 확신하고 놓친 순)', 'recall95',
        (x) => x.label === 1 && x.score < tm.recall95, (x) => -x.score]);
      if (OPS.includes('fpr05')) groups.push(['이 모델이 잘못 막는 정상 문장 (오탐 최소화 기준에서도 차단, 확신하고 막은 순)', 'fpr05',
        (x) => x.label === 0 && x.score >= tm.fpr05, (x) => x.score]);
      const parts = groups.map(([title, op, cond, key]) => {
        const cands = xs.filter((x) => cond(x, base && base[x.id])).map((x) => [key(x, base && base[x.id]), x]).sort((p, q) => q[0] - p[0]);
        const picked = [], per = {}, seen = new Set();
        for (const [, x] of cands) {
          if (seen.has(x.text) || (per[x.dataset] || 0) >= 2) continue;
          picked.push(x); seen.add(x.text); per[x.dataset] = (per[x.dataset] || 0) + 1;
          if (picked.length >= 10) break;
        }
        const sec = el('div', 'sec'), hd = el('h3', null, title);
        hd.append(el('span', 'small muted', ` · ${P.opName(op)} 기준 · 전체 ${P.int(cands.length)}건 중 ${picked.length}건`));
        sec.append(hd);
        if (!picked.length) { sec.append(el('p', 'muted small', '해당하는 사례가 없습니다.')); return sec; }
        const tw = el('div', 'tw'), tbl = el('table'), th = el('thead'), tr = el('tr');
        ['문장', '정답 (데이터셋 라벨)', B ? '판정 (기준 → 이 모델)' : '판정', '데이터셋'].forEach((x) => tr.append(el('th', null, x))); th.append(tr);
        const body = el('tbody');
        const dec = (who, x, t) => { const s = el('span', 'line'); s.append(el('span', 'muted', who), el('span', 'tag ' + (x.score >= t ? 'block' : 'pass'), x.score >= t ? '차단' : '통과'), el('span', 'muted num', (+x.score).toFixed(3))); return s; };
        picked.forEach((x) => {
          const y = base && base[x.id], r = el('tr');
          r.append(td(x.text, 'text'));
          const gold = el('div'); gold.append(el('div', null, x.label ? '개인정보 포함' : '개인정보 없음'));
          if (x.categories && x.categories.length) gold.append(el('small', 'd', x.categories.map(P.catName).join(', ')));
          r.append(td(gold));
          const d = el('div', 'cmp');
          if (y) d.append(dec('기준', y, tb[op]));
          d.append(dec('이 모델', x, tm[op]));
          if (x.choice) d.append(el('small', 'd', `고른 유형: ${P.catName(x.choice)}${x.choice_prob != null ? ' ' + P.pct(x.choice_prob) : ''}`));
          r.append(td(d));
          const ds = el('div'); ds.append(el('div', null, P.dsTitle(x.dataset))); if (x.variant && x.variant !== 'original') ds.append(el('small', 'd', P.varName(x.variant)));
          r.append(td(ds));
          body.append(r);
        });
        tbl.append(th, body); tw.append(tbl); sec.append(tw);
        return sec;
      });
      box.replaceChildren(...parts);
    }).catch((e) => { if (my === token) box.replaceChildren(el('p', 'muted', e.message)); });
  }

  render();  // 위의 도우미(const·let)가 모두 초기화된 뒤에 그린다
})();
