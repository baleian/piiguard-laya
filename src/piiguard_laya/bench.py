"""모델 평가 (`make bench MODEL=... [DATASET=...]`).

    reports/<모델>/
        model.json                  모델 정보 (출처, 모델 카드 요약, 실행 환경)
        calibration.json            모델 임계값과 선택된 질문 문구 (calib 행에서 결정)
        latency.json                요청 1건 처리 시간
        _calibration/<데이터셋>.jsonl  보정에 쓴 calib 행의 원시 예측
        <데이터셋>/predictions.jsonl  test 행마다 원문 · 정답 · 모델 점수 · 모델이 고른 유형
        <데이터셋>/metrics.json       지표 요약

순서
 1. 모델 확인 (HF 모델은 HF_HOME 에 내려받음, 있으면 건너뜀)
 2. 보정: 평가용 데이터셋의 calib 행으로 임계값을 정한다. 이미 했고 데이터가 그대로면 다시 하지 않는다.
 3. 평가: 지정한 데이터셋(없으면 전부)의 test 행을 판정한다. 이미 평가했고 데이터·모델이 그대로면 건너뛴다.
 4. 요약: 이 모델의 모든 데이터셋 지표를 현재 임계값으로 다시 계산한다.
 5. 결과 페이지(reports/*.html)를 다시 만든다.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import datasets as D
from . import evaluate as E
from . import paths, registry
from .detector import LayaDetector, QuestionSet


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(home: Path) -> Dict[str, Any]:
    return yaml.safe_load(paths.config(home, "bench.yaml").read_text(encoding="utf-8"))


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stratified_sample(rows: List[Dict], k: Optional[int]) -> List[Dict]:
    """label × variant 층별 비례 추출. sha256(id) 순서라 결정적."""
    if not k or len(rows) <= k:
        return rows
    strata: Dict[Any, List[Dict]] = defaultdict(list)
    for r in rows:
        strata[(r["label"], r.get("variant", "original"))].append(r)
    out = []
    for _, xs in sorted(strata.items(), key=lambda kv: str(kv[0])):
        xs = sorted(xs, key=lambda r: _h(r["id"]))
        out.extend(xs[:max(1, round(k * len(xs) / len(rows)))])
    return out


def read_jsonl(p: Path) -> List[Dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows: List[Dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _item(r: Dict, o: Dict) -> Dict:
    it = {k: r[k] for k in ("id", "text", "lang", "label", "categories", "variant") if k in r}
    for k in ("group", "parent_id", "source"):
        if r.get(k):
            it[k] = r[k]
    it.update({"scores": o["scores"], "choice": o["categories"], "truncated": o.get("truncated", False)})
    return it


class _Lazy:
    """추론이 필요할 때만 모델을 올린다 (전부 캐시면 GPU 를 쓰지 않음)."""

    def __init__(self, **kw):
        self.kw, self.det = kw, None

    def __call__(self) -> LayaDetector:
        if self.det is None:
            self.det = LayaDetector(**self.kw)
        return self.det

    def infer(self, rows: List[Dict]) -> List[Dict]:
        outs = self().predict([r["text"] for r in rows], [r["lang"] for r in rows])
        return [_item(r, o) for r, o in zip(rows, outs)]


def _rel(p: Path, home: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(home).resolve()))
    except ValueError:
        return str(p)


def _cache_ok(meta_p: Path, key: Dict) -> bool:
    if not meta_p.exists():
        return False
    try:
        return json.loads(meta_p.read_text(encoding="utf-8")).get("key") == key
    except json.JSONDecodeError:
        return False


def run(home: Path, model: str, datasets: Optional[List[str]] = None, limit: Optional[int] = None, device: str = "auto",
        backend: str = "torch", batch_size: Optional[int] = None, recalibrate: bool = False, force: bool = False,
        build_site: bool = True) -> Path:
    cfg = load_config(home)
    ref = registry.resolve(model, home)
    qs_path = (home / cfg["question_set"]).resolve()
    qs = QuestionSet.load(qs_path)
    qs_sha = qs.rendered_sha256()
    if backend == "mlx":
        source = ref.mlx_source()
        fp = f"mlx:{source}:{ref.revision or ''}" if ref.kind == "hf" else "mlx:" + ref.fingerprint()
    else:
        source = str(ref.local_dir())
        fp = ref.fingerprint()

    out = paths.reports_dir(home) / ref.name
    mj = out / "model.json"
    if mj.exists():
        old = json.loads(mj.read_text(encoding="utf-8"))
        if old.get("fingerprint") != fp or old.get("question_set", {}).get("sha256") != qs_sha:
            print(f"[bench] {ref.name}: 모델이나 질문 문구가 바뀌어 이전 결과(reports/{ref.name})를 지우고 다시 평가합니다.")
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    lazy = _Lazy(model=source, device=device, question_set=qs_path, batch_size=batch_size or cfg.get("batch_size", 32),
                 backend=backend)

    # ---------------- 2. 보정 (calib 행)
    eval_all = D.list_eval(home)
    if not eval_all:
        raise SystemExit("datasets/ 에 평가용 데이터셋이 없습니다. 먼저 `make prepare` 를 실행하세요.")
    ccfg = cfg.get("calibration", {})
    names = ccfg.get("datasets", "auto")
    cal_sets = [d for d in eval_all if (names == "auto" or d.name in names)]
    cal_dir = out / "_calibration"
    cal_items, cal_used = [], {}
    changed = recalibrate
    cal_limit = ccfg.get("limit_per_dataset")
    if limit:  # 빠른 확인(LIMIT)일 때는 보정 행도 같은 수로 줄인다
        cal_limit = min(cal_limit or limit, limit)
    for d in cal_sets:
        rows = [r for r in d.rows() if r["split"] == "calib"]
        if not rows:
            continue
        rows = stratified_sample(rows, cal_limit)
        key = {"dataset_sha256": d.sha256(), "limit": cal_limit, "n": len(rows)}
        if recalibrate or not _cache_ok(cal_dir / f"{d.name}.json", key):
            print(f"[bench] 보정용 판정: {d.name} calib {len(rows)}건")
            write_jsonl(cal_dir / f"{d.name}.jsonl", lazy.infer(rows))
            (cal_dir / f"{d.name}.json").write_text(json.dumps({"key": key, "at": now()}, ensure_ascii=False), encoding="utf-8")
            changed = True
        cal_items += read_jsonl(cal_dir / f"{d.name}.jsonl")
        cal_used[d.name] = len(rows)
    for stale in cal_dir.glob("*.json*") if cal_dir.exists() else []:
        if stale.stem not in cal_used:
            stale.unlink()
            changed = True
    if not cal_items:
        raise SystemExit("calib 행이 있는 평가용 데이터셋이 없어 임계값을 정할 수 없습니다.")
    cal_p = out / "calibration.json"
    if changed or not cal_p.exists():
        calib = E.calibrate(cal_items, qs, cfg["operating_points"])
        calib.update({"datasets": cal_used, "at": now()})
        cal_p.write_text(json.dumps(calib, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[bench] 임계값: {calib['thresholds']} (점수 문구 {calib['score_variant']})")
    calib = json.loads(cal_p.read_text(encoding="utf-8"))

    # ---------------- 3. 평가 (test 행)
    targets = D.get(datasets) if datasets else eval_all
    for d in targets:
        rows = [r for r in d.rows() if r["split"] == "test"]
        rows = stratified_sample(rows, limit)
        key = {"dataset_sha256": d.sha256(), "limit": limit, "n": len(rows)}
        dd = out / d.name
        if not force and (dd / "predictions.jsonl").exists() and _cache_ok(dd / "meta.json", key):
            print(f"[bench] {d.name}: 이미 평가됨 — 건너뜀 (다시 하려면 FORCE=1)")
            continue
        print(f"[bench] 평가: {d.name} test {len(rows)}건")
        write_jsonl(dd / "predictions.jsonl", lazy.infer(rows))
        (dd / "meta.json").write_text(json.dumps({"key": key, "at": now(), "dataset": d.meta}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    # ---------------- 4. 요약 (이 모델의 모든 데이터셋)
    for dd in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not (dd / "predictions.jsonl").exists():
            continue
        items = read_jsonl(dd / "predictions.jsonl")
        meta = json.loads((dd / "meta.json").read_text(encoding="utf-8")) if (dd / "meta.json").exists() else {}
        m = E.summarize(items, calib)
        m.update({"dataset": dd.name, "model": ref.name, "title": meta.get("dataset", {}).get("title", dd.name),
                  "description": meta.get("dataset", {}).get("description", ""), "evaluated_at": meta.get("at"),
                  "score_variant": calib["score_variant"], "category_variant": calib["category_variant"],
                  "thresholds": calib["thresholds"], "limit": meta.get("key", {}).get("limit")})
        (dd / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 지연시간 (이번에 모델을 올렸을 때만 새로 잰다). 이 모델의 모든 데이터셋 문장에서 표본을 뽑는다
    lat_p = out / "latency.json"
    if lazy.det is not None or not lat_p.exists():
        texts = []
        for p in sorted(out.glob("*/predictions.jsonl")):
            texts += [(r["text"], r["lang"]) for r in read_jsonl(p)]
        random.Random(0).shuffle(texts)
        texts = texts[:cfg.get("latency_requests", 200)]
        if texts:
            only = sorted(qs.variants) if calib["score_variant"] == E.ENS else sorted({calib["score_variant"], calib["category_variant"]})
            print(f"[bench] 지연시간 측정: 요청 {len(texts)}건")
            lat = lazy().latency([t for t, _ in texts], [l for _, l in texts], only=only)
            info = lazy().info()
            lat_p.write_text(json.dumps({"at": now(), "device": info.get("device"), "gpu": info.get("gpu"),
                                         "results": list(lat.values())}, ensure_ascii=False, indent=1), encoding="utf-8")

    info = lazy.det.info() if lazy.det else (json.loads(mj.read_text(encoding="utf-8")).get("runtime") if mj.exists() else {})
    mj.write_text(json.dumps({**ref.describe(), "fingerprint": fp, "backend": backend, "benched_at": now(),
                              "question_set": {"path": str(cfg["question_set"]), "version": qs.version, "sha256": qs_sha},
                              "runtime": info, "host": {"platform": platform.platform(), "python": platform.python_version()}},
                             ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[bench] 완료 → reports/{ref.name}/")
    if build_site:
        from . import site

        print(f"[bench] 결과 페이지 → {_rel(site.build(home), home)}")
    return out
