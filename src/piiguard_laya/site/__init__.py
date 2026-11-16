"""결과 페이지 생성 (`make site`, bench·train 뒤에 자동 실행).

    reports/index.html         홈: 모델·데이터셋 목록, 사용법
    reports/dashboard.html     모든 모델의 주요 지표 요약
    reports/evals.html         모델 하나의 결과, 다른 모델(기본: 학습된 모델이면 그 기준 모델)과 비교
    reports/predictions.html   문장별 예측 결과 조회 (원문 · 정답 · 모델 점수 · 판정 · 모델이 고른 유형)
    reports/_site/             공용 스타일·스크립트와 데이터 파일

페이지는 정적 파일이라 서버 없이 브라우저로 바로 연다. 데이터는 <script src> 로 읽는다
(브라우저가 file:// 에서 fetch 를 막기 때문). predictions.js 의 문장 원문은 모델을 고를 때만 읽는다.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .. import datasets as D
from .. import paths, registry
from .. import taxonomy as T
from ..summary import compare, list_reports, model_summary

ASSETS = Path(__file__).parent / "assets"
PAGES = ("index.html", "dashboard.html", "evals.html", "predictions.html")
VARIANTS = {"original": "원문 (변형 없음)", "fullwidth": "전각 문자 (０１０)", "zwsp": "보이지 않는 공백 삽입",
            "spaced": "글자 사이 띄어쓰기 (0 1 0)", "digit_words": "숫자를 단어로 (공일공 / zero)", "homoglyph": "비슷한 모양 문자 (O·l·S)",
            "dots": "점으로 구분 (010.1234)", "split": "값을 둘로 나눠 쓰기", "email_spelled": "이메일 풀어쓰기 (골뱅이 / at)",
            "wrapper": "'개인정보 아님' 위장 문구"}
OPS = {"recall95": {"name": "누락 최소화", "desc": "개인정보 문장의 95% 이상을 잡도록 임계값을 맞춤"},
       "fpr05": {"name": "오탐 최소화", "desc": "정상 문장을 잘못 막는 비율이 5% 이하가 되도록 임계값을 맞춤"},
       "default": {"name": "기본 기준 (0.5)", "desc": "점수 0.5 이상이면 차단. 별도 보정 없이 쓸 때"},
       "fpr10": {"name": "오탐 10% 이하", "desc": "정상 문장 차단을 10% 이하로"},
       "f1max": {"name": "F1 최대", "desc": "보정용 문장에서 F1 이 가장 높은 임계값"}}


def _js(var: str, obj: Any) -> str:
    return f"window.{var} = " + json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str) + ";\n"


def build(home: Path) -> Path:
    root = paths.reports_dir(home)
    root.mkdir(parents=True, exist_ok=True)
    site = root / "_site"
    data = site / "data"
    shutil.rmtree(data, ignore_errors=True)
    (data / "items").mkdir(parents=True, exist_ok=True)
    for f in ASSETS.iterdir():
        if f.suffix in (".css", ".js"):
            shutil.copy2(f, site / f.name)
        elif f.suffix == ".html":
            shutil.copy2(f, root / f.name)

    bcfg = yaml.safe_load(paths.config(home, "bench.yaml").read_text(encoding="utf-8"))
    rules = bcfg.get("gate", {})
    names = list_reports(home)
    summaries, items = {}, {}
    for n in names:
        s, it = model_summary(home, n, with_items=True)
        summaries[n], items[n] = s, it
        rows = []
        for d, xs in it.items():
            for x in xs:
                c = x.get("choice")
                rows.append([x["id"], d, x["lang"], x.get("variant", "original"), x["label"], x["categories"], x["text"],
                             round(x["score"], 4), c, (x.get("probabilities") or {}).get(c)])
        (data / "items" / f"{n.replace('/', '--')}.js").write_text(  # 모델 id 의 / → -- (파일 이름)
            "window.PIIG_ITEMS = window.PIIG_ITEMS || {};\n" + f"window.PIIG_ITEMS[{json.dumps(n)}] = "
            + json.dumps({"cols": ["id", "dataset", "lang", "variant", "label", "categories", "text", "score", "choice", "choice_prob"],
                          "rows": rows}, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    pairs = {a: {b: compare(summaries[a], summaries[b], rules, items[a], items[b]) for b in names if b != a} for a in names}
    local = registry.list_local(home)  # models/base/ · models/trained/ 에 있는 모델 (평가 전인 것 포함)
    for x in local:
        x["evaluated"] = x["id"] in summaries
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": summaries, "compare": pairs, "local": local, "kind_labels": registry.KIND_LABEL,
        "datasets": [d.describe() for d in D.list_eval(home)],
        "training": [d.describe() for d in D.list_train(home)],
        "labels": {"categories": T.CATEGORY_KO, "category_order": T.CATEGORIES, "variants": VARIANTS, "ops": OPS},
        "gate": rules, "config": {"operating_points": bcfg.get("operating_points"), "calibration": bcfg.get("calibration")},
    }
    (data / "summary.js").write_text(_js("PIIG", payload), encoding="utf-8")
    return root / "index.html"
