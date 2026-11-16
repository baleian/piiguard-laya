"""데이터셋 읽기·쓰기. 평가용(datasets/)과 학습용(training/)이 같은 형식을 쓴다.

폴더 하나가 데이터셋 하나다.

    datasets/<이름>/data.jsonl      (또는 data.csv)
    datasets/<이름>/dataset.json    (선택) 제목·설명·언어·라이선스·출처

행 형식 (jsonl 한 줄 = 문장 하나):
    text        (필수) 문장
    label       (필수) 개인정보 포함 여부. 1/0, true/false, "pii"/"none"
    categories  (선택) 들어 있는 개인정보 유형 목록. none, national_id, financial, credential, sensitive, contact,
                identity, quasi_identifier 중에서. CSV 에서는 "contact|identity" 처럼 | 로 구분
    id          (선택) 없으면 "<데이터셋>-<줄번호>"
    lang        (선택) ko / en. 없으면 글자로 추정
    split       (선택, 평가용) calib / test. calib 행은 모델별 임계값 결정에만 쓰고 test 행으로 성능을 낸다.
                없으면 모두 test 로 본다.
    variant     (선택) 변형 이름 (변조 입력 데이터셋 등). 없으면 original
    spans       (선택) [{"start", "end", "type"}] 개인정보 위치 (학습 시 대체값 치환에 쓴다)
    그 밖의 필드는 그대로 보존된다.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import paths
from . import taxonomy as T
from .detector import detect_lang

_TRUE = {"1", "true", "yes", "y", "pii", "개인정보", "o"}
_FALSE = {"0", "false", "no", "n", "none", "정상", "x", ""}


@dataclass
class Dataset:
    name: str
    path: Path
    kind: str  # "eval" | "train"

    @property
    def file(self) -> Optional[Path]:
        for f in ("data.jsonl", "data.csv"):
            if (self.path / f).exists():
                return self.path / f
        return None

    @property
    def meta(self) -> Dict[str, Any]:
        p = self.path / "dataset.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.file, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def rows(self) -> List[Dict[str, Any]]:
        return [normalize(r, self.name, i) for i, r in enumerate(read_raw(self.file))]

    def describe(self) -> Dict[str, Any]:
        rows = self.rows()
        m = self.meta
        return {"name": self.name, "kind": self.kind, "title": m.get("title", self.name), "description": m.get("description", ""),
                "language": m.get("language") or "/".join(sorted({r["lang"] for r in rows})), "license": m.get("license", ""),
                "source": m.get("source", ""), "n": len(rows),
                "by_split_label": {f"{s}/{l}": c for (s, l), c in sorted(Counter((r["split"], r["label"]) for r in rows).items())}}


def read_raw(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _label(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v > 0)
    s = str(v).strip().lower()
    if s in _TRUE:
        return 1
    if s in _FALSE:
        return 0
    raise ValueError(f"label 값을 해석할 수 없습니다: {v!r}")


def normalize(r: Dict[str, Any], name: str, i: int) -> Dict[str, Any]:
    text = r.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{name} {i + 1}번째 행: text 가 비어 있습니다")
    cats = r.get("categories") or []
    if isinstance(cats, str):
        cats = [c for c in cats.replace(",", "|").split("|") if c.strip()]
    cats = [c.strip() for c in cats if c and c.strip() and c.strip() != T.NONE]
    unknown = [c for c in cats if c not in T.CATEGORIES]
    if unknown:
        raise ValueError(f"{name} {i + 1}번째 행: 알 수 없는 개인정보 유형 {unknown} (사용 가능: {T.PII_CATEGORIES})")
    label = _label(r.get("label", 1 if cats else 0))
    split = (r.get("split") or "test").strip()
    out = dict(r)
    out.update({"id": str(r.get("id") or f"{name}-{i + 1}"), "text": text, "label": label, "categories": sorted(set(cats)),
                "lang": r.get("lang") or detect_lang(text), "split": split if split in ("calib", "test", "train") else "test",
                "variant": r.get("variant") or "original", "dataset": name})
    out["primary"] = T.primary_category(out["categories"])
    return out


def _list(root: Path, kind: str) -> List[Dataset]:
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith((".", "_")):
            ds = Dataset(d.name, d, kind)
            if ds.file is not None:
                out.append(ds)
    return out


def list_eval(home: Optional[Path] = None) -> List[Dataset]:
    return _list(paths.datasets_dir(home), "eval")


def list_train(home: Optional[Path] = None) -> List[Dataset]:
    return _list(paths.training_dir(home), "train")


def get(names: Optional[Iterable[str]], kind: str = "eval", home: Optional[Path] = None) -> List[Dataset]:
    all_ = list_eval(home) if kind == "eval" else list_train(home)
    if not names:
        return all_
    by = {d.name: d for d in all_}
    missing = [n for n in names if n not in by]
    if missing:
        where = "datasets/" if kind == "eval" else "training/"
        raise SystemExit(f"{where} 에 없는 데이터셋: {missing}. 있는 것: {sorted(by)}")
    return [by[n] for n in names]


def write(root: Path, name: str, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "data.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = dict(meta)
    meta.setdefault("title", name)
    meta["n"] = len(rows)
    meta["by_split_label"] = {f"{s}/{l}": c for (s, l), c in sorted(Counter((r.get("split", "test"), r["label"]) for r in rows).items())}
    (d / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
