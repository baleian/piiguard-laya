/* piiguard-laya 결과 페이지 공용 도우미 */
(function () {
  const S = window.PIIG || null;
  const OP_ORDER = ['recall95', 'fpr05', 'default', 'fpr10', 'f1max'];
  const P = {
    S,
    pct: (v, d = 1) => (v == null || v === '' || isNaN(+v) ? '–' : (+v * 100).toFixed(d) + '%'),
    f3: (v) => (v == null || v === '' || isNaN(+v) ? '–' : (+v).toFixed(3)),
    f4: (v) => (v == null || v === '' || isNaN(+v) ? '–' : (+v).toFixed(4)),
    // 임계값: 정상 문장의 점수가 1.0 에 몰려 있으면 목표 오탐률을 맞출 임계값이 1 을 넘는다 → 그 기준에서는 아무것도 차단하지 않음
    thr: (v) => (v == null || isNaN(+v) ? '–' : +v > 1 ? '1 초과 (차단 없음)' : (+v).toFixed(4)),
    int: (v) => (v == null || v === '' ? '–' : (+v).toLocaleString('ko-KR')),
    date: (s) => (s ? new Date(s).toLocaleString('ko-KR', { dateStyle: 'medium', timeStyle: 'short' }) : '–'),
    el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; },
    td(content, cls) { const e = P.el('td', cls); if (content instanceof Node) e.append(content); else if (content != null) e.textContent = content; return e; },
    up: { accuracy: true, recall: true, fpr: false, precision: true, auroc: true, top1: true, rate: true },
    fmt: (v, kind) => (kind === 'score' ? P.f3(v) : P.pct(v)),
    // 변화 배지. up = 값이 오를수록 좋은 지표인지
    chip(d, kind, up) {
      if (d == null || isNaN(+d)) return P.el('span', 'chip flat', '–');
      const v = +d;
      const text = (v >= 0 ? '+' : '−') + (kind === 'score' ? Math.abs(v).toFixed(3) : Math.abs(v * 100).toFixed(1) + '%p');
      return P.el('span', 'chip ' + (Math.abs(v) < 0.005 ? 'flat' : (v > 0) === up ? 'good' : 'bad'), text);
    },
    // 비교 칸: 기준 → 대상 + 변화. 기준이 없으면 대상만
    cmp(b, t, kind, up) {
      const box = P.el('div', 'cmp');
      const has = (x) => x != null && x !== '' && !isNaN(+x);
      if (!has(t) && !has(b)) { box.append(P.el('span', 'na', '해당 없음')); return box; }
      const line = P.el('span', 'line');
      if (has(b) && has(t)) {
        line.append(P.el('span', 'b', P.fmt(b, kind)), P.el('span', 'b', '→'), P.el('span', 't', P.fmt(t, kind)));
        box.append(line, P.chip(t - b, kind, up));
      } else {
        line.append(P.el('span', 't', P.fmt(has(t) ? t : b, kind)));
        box.append(line);
      }
      return box;
    },
    ops(model) {
      const t = (model && model.calibration && model.calibration.thresholds) || {};
      return OP_ORDER.filter((o) => t[o] != null);
    },
    opName: (op) => ((S && S.labels.ops[op]) || {}).name || op,
    opDesc: (op) => ((S && S.labels.ops[op]) || {}).desc || '',
    catName: (k) => (S && S.labels.categories[k]) || k,
    varName: (k) => (S && S.labels.variants[k]) || k,
    dsTitle(name) { const d = S && S.datasets.find((x) => x.name === name); return d ? d.title : name; },
    models: () => (S ? Object.keys(S.models) : []),
    latest() {
      const ms = P.models();
      return ms.sort((a, b) => String(S.models[b].model.benched_at || '').localeCompare(String(S.models[a].model.benched_at || '')))[0];
    },
    hash() { return Object.fromEntries(location.hash.replace(/^#/, '').split('&').filter(Boolean).map((kv) => kv.split('=').map(decodeURIComponent))); },
    setHash(o) {
      const s = '#' + Object.entries(o).filter(([, v]) => v != null && v !== '').map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(v)).join('&');
      if (location.hash === s) return;
      // file:// 에서는 브라우저에 따라 replaceState 가 막힌다 → 해시만 바꾸는 이동으로 대신한다 (주소 갱신은 부가 기능)
      try { history.replaceState(null, '', s); } catch (e) { try { location.replace(s); } catch (e2) { /* 무시 */ } }
    },
    options(sel, items, value) { sel.replaceChildren(...items.map(([v, t]) => new Option(t, v))); if (value != null) sel.value = value; },
    _loading: {},
    loadItems(name) {
      window.PIIG_ITEMS = window.PIIG_ITEMS || {};
      if (window.PIIG_ITEMS[name]) return Promise.resolve(window.PIIG_ITEMS[name]);
      if (!P._loading[name]) {
        P._loading[name] = new Promise((res, rej) => {
          const s = document.createElement('script');
          s.src = '_site/data/items/' + encodeURIComponent(name) + '.js';
          s.onload = () => res(window.PIIG_ITEMS[name]);
          s.onerror = () => rej(new Error('예측 데이터를 읽지 못했습니다: ' + name));
          document.head.append(s);
        });
      }
      return P._loading[name];
    },
    rowsOf(pack) { const c = pack.cols; return pack.rows.map((r) => Object.fromEntries(c.map((k, i) => [k, r[i]]))); },
    kindLabel(m) {
      const k = m && m.model && m.model.kind;
      return k === 'trained' ? '직접 학습' : k === 'hf' ? 'Hugging Face' : '';
    },
    baseOf(name) { const c = S.models[name] && S.models[name].model.card; return c && c.base ? c.base.name : null; },
    empty(container, msg) { container.replaceChildren(P.el('div', 'empty', msg)); },
    init(active) {
      document.querySelectorAll('.nav a.link').forEach((a) => a.classList.toggle('on', a.dataset.page === active));
      const g = document.querySelector('.nav .gen');
      if (g) g.textContent = S ? '생성 ' + P.date(S.generated_at) : '';
      if (!S) {
        const w = document.querySelector('.wrap');
        if (w) P.empty(w, '데이터가 없습니다. `make bench MODEL=laya-multilingual` 로 평가를 실행하면 이 페이지가 채워집니다.');
        return false;
      }
      return true;
    },
  };
  window.P = P;
})();
