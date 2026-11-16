"""운영 피드백 수집: 로그 → 검토 큐(CSV) → 검토 결과를 학습용 데이터셋에 추가.

    piiguard-laya review-export --log logs.jsonl --model trained/r1 --out feedback/queue.csv
    (검토자가 CSV 의 review_* 열과 reviewer 를 채운다. review_values 에 문장 속 개인정보 값을 적으면
     학습 때 그 값을 같은 형식의 합성값으로 바꾼다)
    piiguard-laya review-import feedback/queue.csv --into feedback-2026-11

로그 형식 (jsonl 한 줄 = 문장 하나): {"text": ..., "id": (선택), "lang": (선택), "ts": (선택), "score"·"choice": (선택)}
score(모델 점수)가 없으면 지정한 모델로 채점한다. 판정 임계값은 그 모델의 reports/<모델 id>/calibration.json 을 쓴다
(먼저 make bench 로 평가해 둔 모델이어야 한다).

뽑는 순서: 임계값 근처(불확실) → 차단된 문장 → 통과된 문장. 잘한 판정과 잘못한 판정을 모두 모으기 위해서다.
검토 결과는 training/<이름>/data.jsonl 에 source="review" 로 쌓인다. 원문이 들어 있으므로 접근 권한을 관리한다.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from . import datasets as D
from . import paths, registry
from . import taxonomy as T

COLUMNS = ["id", "lang", "reason", "model", "model_score", "model_decision", "model_choice", "text",
           "review_is_pii", "review_categories", "review_values", "review_confidence", "reviewer", "note"]


def _hid(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def export_queue(home: Path, log: Path, model: str, out: Path, op: str = "fpr05", band: float = 0.1, limit: int = 500,
                 device: str = "auto", backend: str = "torch") -> Dict:
    ref = registry.resolve(model, home)
    cal_p = paths.reports_dir(home) / ref.id / "calibration.json"
    if not cal_p.exists():
        raise SystemExit(f"{ref.id} 의 임계값이 없습니다. 먼저 `make bench MODEL={ref.id}` 를 실행하세요.")
    calib = json.loads(cal_p.read_text(encoding="utf-8"))
    rows = []
    with open(log, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                rows.append({"id": str(r.get("id") or f"log-{_hid(r['text'])}"), "text": r["text"], "lang": r.get("lang"),
                             "score": r.get("score"), "choice": r.get("choice")})
    from .detector import detect_lang

    for r in rows:
        r["lang"] = r["lang"] or detect_lang(r["text"])
    todo = [r for r in rows if r.get("score") is None]
    if todo:
        from .bench import load_config
        from .detector import LayaDetector

        cfg = load_config(home)
        det = LayaDetector(model=ref.mlx_source() if backend == "mlx" else str(ref.local_dir()), device=device,
                           question=(home / cfg["question"]).resolve(), backend=backend)
        for r, o in zip(todo, det.predict([r["text"] for r in todo])):
            r["score"], r["choice"] = o["score"], o["choice"]
    t = calib["thresholds"][op]
    for r in rows:
        r["score"] = float(r["score"])
        r["block"] = int(r["score"] >= t)
        r["choice_name"] = T.CATEGORY_KO.get(r.get("choice"), r.get("choice") or "")
    groups = {"uncertain": sorted((r for r in rows if abs(r["score"] - t) <= band), key=lambda r: abs(r["score"] - t)),
              "blocked": sorted((r for r in rows if r["block"] and abs(r["score"] - t) > band), key=lambda r: _hid(r["id"])),
              "passed": sorted((r for r in rows if not r["block"] and abs(r["score"] - t) > band), key=lambda r: _hid(r["id"]))}
    picked, seen, i = [], set(), 0
    while len(picked) < limit and any(i < len(v) for v in groups.values()):
        for reason, xs in groups.items():
            if i < len(xs) and len(picked) < limit and xs[i]["id"] not in seen:
                picked.append((reason, xs[i]))
                seen.add(xs[i]["id"])
        i += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for reason, r in picked:
            w.writerow([r["id"], r["lang"], reason, ref.id, round(r["score"], 4), "차단" if r["block"] else "통과", r["choice_name"],
                        r["text"], "", "", "", "", "", ""])
    from collections import Counter

    return {"out": str(out), "n": len(picked), "by_reason": dict(Counter(x for x, _ in picked)), "threshold": t, "op": op}


def value_spans(text: str, values: str, categories) -> tuple:
    """review_values ("010-2345-6789|contact:hong@x.kr") → 문장 속 위치 스팬. 학습 때 이 위치의 값을 합성값으로 바꾼다.

    값 앞에 "유형:" 을 붙이면 그 유형으로, 아니면 review_categories 의 대표 유형으로 본다. 같은 값이 여러 번 나오면 모두 표시한다.
    """
    spans, missing = [], []
    default = T.primary_category(categories) if categories else ""
    for raw in str(values or "").split("|"):
        raw = raw.strip()
        if not raw:
            continue
        cat, val = default, raw
        head, sep, rest = raw.partition(":")
        if sep and head.strip() in T.PII_CATEGORIES and rest.strip():
            cat, val = head.strip(), rest.strip()
        start, found = text.find(val), False
        while start != -1:
            spans.append({"start": start, "end": start + len(val), "type": cat or T.NONE, "value_source": "review_values"})
            found = True
            start = text.find(val, start + len(val))
        if not found:
            missing.append(val)
    return spans, missing


def import_reviews(home: Path, file: Path, into: str, require_reviewer: bool = True) -> Dict:
    if not registry.NAME_RE.match(into):
        raise SystemExit(f"데이터셋 이름 '{into}' 은 영문·숫자·. _ - 만 쓸 수 있습니다.")
    with open(file, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f)) if file.suffix == ".csv" else [json.loads(l) for l in f if l.strip()]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new, skipped, warn = [], {"no_review": 0, "no_reviewer": 0}, {"pii_without_values": 0, "values_not_in_text": 0}
    for r in rows:
        v = str(r.get("review_is_pii", "")).strip().lower()
        if v in ("", "none", "null"):
            skipped["no_review"] += 1
            continue
        if require_reviewer and not str(r.get("reviewer", "")).strip():
            skipped["no_reviewer"] += 1
            continue
        cats = [c for c in str(r.get("review_categories") or "").replace(",", "|").split("|") if c.strip()]
        conf = str(r.get("review_confidence") or "").strip()
        label = 1 if v in ("1", "true", "yes", "y", "o", "예", "pii") else 0
        spans, missing = value_spans(r["text"], r.get("review_values"), cats) if label else ([], [])
        warn["values_not_in_text"] += len(missing)
        if label and not spans:
            warn["pii_without_values"] += 1
        new.append({"id": r.get("id") or f"review-{_hid(r['text'])}", "text": r["text"], "lang": r.get("lang") or None,
                    "label": label, "categories": cats, **({"spans": spans} if spans else {}),
                    "confidence": float(conf) if conf else None, "reviewer": r.get("reviewer"), "note": r.get("note") or "",
                    "reviewed_at": now, "source": "review", "origin": {"model": r.get("model"), "reason": r.get("reason"),
                                                                       "model_score": r.get("model_score")}})
    for i, r in enumerate(new):
        D.normalize(r, into, i)  # 형식 검사 (유형 이름 오타 등)
    root = paths.training_dir(home)
    existing = D.get([into], kind="train", home=home)[0].rows() if (root / into / "data.jsonl").exists() else []
    latest = {r["id"]: r for r in existing}
    for r in new:
        latest[r["id"]] = r  # 같은 문장을 다시 검토하면 마지막 판정을 쓴다
    meta = D.Dataset(into, root / into, "train").meta if (root / into / "dataset.json").exists() else {
        "title": into, "description": "운영 로그 검토 결과", "source": "piiguard-laya review-import"}
    D.write(root, into, list(latest.values()), meta)
    return {"into": f"training/{into}", "added": len(new), "total": len(latest), "skipped": skipped, "warnings": warn}
