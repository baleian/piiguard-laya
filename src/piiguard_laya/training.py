"""모델 학습 (`make train BASE=... NAME=... DATA=...`).

    BASE   기준 모델 id. base/<이름> (models/base/) 또는 trained/<이름> (models/trained/), hf:org/repo[:하위폴더]
    NAME   새 모델 이름 → models/trained/<NAME>/ (모델 id trained/<NAME>)
    DATA   학습용 데이터셋 이름 (training/<이름>/), 쉼표로 여러 개

학습 방식은 Laya 의 RLCD 다. 검토 라벨(정답)을 적정 점수 규칙(proper scoring rule) 보상으로 쓰는
GRPO 방식 강화학습이며, 학습 후 온도(temperature)를 다시 보정한다.

학습 데이터 처리
  - 평가용 데이터셋(datasets/)과 원문이 같은 행은 뺀다 (평가 누수 방지). 뺀 수는 모델 카드에 남는다.
  - source 가 "review"(검토자가 넣은 운영 데이터)이고 spans 가 있으면 실제 값을 같은 형식의 합성값으로 바꾼 뒤 학습한다.
  - weight 필드(정수)가 있으면 그만큼 반복해 가중한다.
  - confidence(0.5~1) 가 1 미만이면 soft label 로 넣는다.

models/trained/<NAME>/MODEL_CARD.json 에 기준 모델·계보·학습 데이터(sha256)·설정·학습 중 평가 지표가 기록된다.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from . import datasets as D
from . import paths, registry
from . import taxonomy as T
from .detector import Question
from .surrogate import surrogate


def make_row(text: str, is_pii: bool, categories: Sequence[str], q: Question, confidence: Optional[float] = None) -> Dict:
    """Laya 학습 행 {state, questions, expected|gold}. 정답은 대표 유형(개인정보가 없으면 none)이다.

    confidence(0.5~1) 가 1 미만이면 그만큼만 확신하는 soft label 로 넣는다.
    개인정보라는 것만 알고 유형을 모르면 '개인정보 없음' 확률만 낮추고 유형은 고르게 둔다.
    """
    primary = T.primary_category(categories) if is_pii else T.NONE
    conf = 1.0 if confidence is None or float(confidence) >= 0.999 else max(0.5, float(confidence))
    pii = T.PII_CATEGORIES
    row: Dict = {"state": text, "questions": q.questions()}
    if is_pii and primary == T.NONE:
        row["gold"] = {q.id: {"probabilities": {**{c: conf / len(pii) for c in pii}, T.NONE: 1 - conf}}}
    elif conf == 1.0:
        row["expected"] = {q.id: primary}
    elif is_pii:
        row["gold"] = {q.id: {"probabilities": {primary: conf, T.NONE: 1 - conf}}}
    else:
        row["gold"] = {q.id: {"probabilities": {T.NONE: conf, **{c: (1 - conf) / len(pii) for c in pii}}}}
    return row


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_jsonl(p: Path, rows: List[Dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


EVAL_SOURCE = "평가용 데이터셋 calib 행 표본 (학습에는 쓰지 않음)"


def _replace_work_paths(report: Path) -> None:
    """laya 학습 리포트에 남은 임시 입력 파일 경로를 설명 문구로 바꾼다."""
    if not report.exists():
        return
    t = json.loads(report.read_text(encoding="utf-8"))
    if "eval_source" in t:
        t["eval_source"] = EVAL_SOURCE
    if isinstance(t.get("training"), dict) and "eval_data" in t["training"]:
        t["training"]["eval_data"] = EVAL_SOURCE
    report.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")


def _lineage(ref: registry.ModelRef) -> List[str]:
    if ref.kind == "trained" and ref.card:
        return list(ref.card.get("lineage") or []) + [ref.id]
    return [ref.id]


def train(home: Path, base: str, name: str, data: List[str], device: str = "auto", overwrite: bool = False,
          build_site: bool = True, **overrides) -> Path:
    from laya.train import TrainConfig, finetune

    tcfg = yaml.safe_load(paths.config(home, "train.yaml").read_text(encoding="utf-8"))
    tcfg.update({k: v for k, v in overrides.items() if v is not None})
    bcfg = yaml.safe_load(paths.config(home, "bench.yaml").read_text(encoding="utf-8"))
    q = Question.load((home / bcfg["question"]).resolve())

    out = registry.validate_new_name(name, home, overwrite)
    base_ref = registry.resolve(base, home)
    base_dir = base_ref.local_dir()
    sets = D.get(data, kind="train", home=home)
    if not sets:
        raise SystemExit("학습용 데이터셋을 지정하세요 (예: DATA=seed). 목록: piiguard-laya datasets")

    eval_sets = D.list_eval(home)
    eval_texts = {_h(r["text"]) for d in eval_sets for r in d.rows()}
    rows, stats, data_info = [], Counter(), []
    for d in sets:
        used = 0
        for r in d.rows():
            if _h(r["text"]) in eval_texts:
                stats["excluded_overlap_with_eval"] += 1
                continue
            text = r["text"]
            if r.get("source") == "review" and r.get("spans"):
                sr = surrogate(text, r["spans"], r.get("lang") or "ko", seed_key=f"{name}|{r['id']}")
                if sr["unlocated"]:
                    stats["excluded_unlocated_span"] += 1
                    continue
                text = sr["text"]
                stats["surrogated"] += 1
            row = make_row(text, bool(r["label"]), r["categories"], q, r.get("confidence"))
            rows.extend([row] * max(1, int(r.get("weight", 1) or 1)))
            used += 1
        desc = d.describe()
        data_info.append({"name": d.name, "title": desc["title"], "sha256": d.sha256(), "rows": desc["n"], "used_rows": used,
                          "by_label": dict(Counter(str(r["label"]) for r in d.rows()))})
    if not rows:
        raise SystemExit(f"학습할 행이 없습니다 ({dict(stats)})")
    rows.sort(key=lambda r: _h(name + json.dumps(r, ensure_ascii=False)))

    # 학습 중 평가: 평가용 데이터셋의 calib 행에서 라벨 균형 표본 (gradient 에는 쓰지 않음)
    cal = sorted((r for d in eval_sets for r in d.rows() if r["split"] == "calib"), key=lambda r: _h("eval" + r["id"]))
    n_eval = int(tcfg.get("eval_rows", 1500))
    pos = [r for r in cal if r["label"] == 1][: n_eval // 2]
    neg = [r for r in cal if r["label"] == 0][: n_eval - len(pos)]
    eval_rows = [make_row(r["text"], bool(r["label"]), r["categories"], q) for r in pos + neg]

    if out.exists() and overwrite:
        shutil.rmtree(out)
    # 학습 입력 파일은 임시 폴더에 만들고 학습이 끝나면 지운다 (모델 폴더에는 모델과 카드만 남긴다)
    with tempfile.TemporaryDirectory(prefix="piiguard-laya-train-") as tmp:
        work = Path(tmp)
        _write_jsonl(work / "train.jsonl", rows)
        _write_jsonl(work / "eval.jsonl", eval_rows)
        cfg = TrainConfig(epochs=int(tcfg["epochs"]), micro_batch=int(tcfg["micro_batch"]), grad_accum=int(tcfg["grad_accum"]),
                          encoder_lr=float(tcfg["encoder_lr"]), head_lr=float(tcfg["head_lr"]), loss=tcfg["loss"],
                          shuffle_options=("choice",), freeze_encoder=bool(tcfg.get("freeze_encoder", False)), seed=int(tcfg["seed"]),
                          max_len=int(tcfg["max_len"]), head_max_len=q.head_max_len, eval_data=str(work / "eval.jsonl"))
        print(f"[train] 기준 모델 {base_ref.id} → 새 모델 trained/{name} · 학습 행 {len(rows)} · 평가 행 {len(eval_rows)} · {dict(stats)}")
        summary = finetune(data=str(work / "train.jsonl"), model_dir=str(base_dir), output_dir=str(out), config=cfg, device=device)
    shutil.rmtree(out / "checkpoint_latest", ignore_errors=True)  # 에폭 중간 저장본 (최종 모델과 별도)
    _replace_work_paths(out / "train_report.json")

    import torch
    import laya

    tr = summary.get("train_report") or {}
    card = {
        "name": name, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": f"{base_ref.id} 에서 {', '.join(d.name for d in sets)} 로 추가 학습",
        "base": base_ref.describe(), "lineage": _lineage(base_ref),
        "training_data": data_info, "train_rows": len(rows), "data_stats": dict(stats),
        "leakage_check": {"excluded_overlap_with_eval": stats.get("excluded_overlap_with_eval", 0),
                          "rule": "평가용 데이터셋(datasets/) 원문과 같은 행은 학습에서 제외"},
        "question": q.describe(str(bcfg["question"])),
        "objective": tcfg["loss"],
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_") and k != "eval_data"},  # eval_data 는 지운 임시 파일
        "train_eval": {"source": EVAL_SOURCE, "rows": len(eval_rows), "items": tr.get("eval_items"),
                       "before": tr.get("before"), "after": tr.get("after"), "comparison": tr.get("comparison"),
                       "note": tr.get("note")},
        "epoch_loss": summary.get("epoch_loss"), "temperature": summary.get("temperature"),
        "calibration": summary.get("calibration"),
        "env": {"laya": getattr(laya, "__version__", "?"), "torch": torch.__version__, "device": device,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "platform": platform.platform()},
    }
    (out / "MODEL_CARD.json").write_text(json.dumps(card, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[train] 완료 → models/trained/{name}/ (평가: make bench MODEL=trained/{name})")
    if build_site:
        from . import site

        site.build(home)
    return out
