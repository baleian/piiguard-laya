"""데이터 준비: 원본 다운로드 → 통합 형식 변환 → 평가용·학습용으로 나눠 저장.

    datasets/<이름>/data.jsonl + dataset.json     평가용 (split = calib / test)
    training/seed/data.jsonl + dataset.json       학습용 시작 데이터 (평가용과 겹치지 않는 train 분할에서 표본)
    datasets/.downloads/                          내려받은 원본 파일

분할은 id(합성 데이터는 문장 틀)의 sha256 으로 결정한다: calib 20% / test 30% / train 50%.
calib 은 모델별 임계값 결정에, test 는 성능 측정에 쓰고, train 은 평가용 데이터셋에 넣지 않는다.
합성 데이터는 문장 틀 단위로 나눠 학습용과 평가용에 같은 틀이 들어가지 않는다.

오프라인 환경: 인터넷이 되는 곳에서 `piiguard-laya prepare --download-only` 로 datasets/.downloads 를 만든 뒤
그 폴더를 옮겨 `piiguard-laya prepare --offline` 으로 실행한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from .. import datasets as D
from .. import paths
from .. import taxonomy as T
from . import perturb, synth

GRETEL_REV = "7b844d16738527a04264f50214cb426a4cea0897"  # gretelai/synthetic_pii_finance_multilingual
KGD_REV = "7e98c2fa2903b020e9c3c1826be37e687e48f3c6"     # skan0779/korean-guardrail-dataset (opt-in 번역본만)

# 기본 소스
SOURCES = {
    "PII_dataset_V3.json": {
        "url": "https://zenodo.org/api/records/16759166/files/PII_dataset_V3.json/content",
        "md5": "419921a5e153e8346b6d320d5f084a4a",
        "license": "CC-BY-4.0",
        "origin": "KDPII DATASET REVISED — Yonsei Univ. HamSaeM Kim's Lab, TSCIENTIFIC Co.,Ltd. "
                  "doi:10.5281/zenodo.16759166"},
    "English_test-00000-of-00001.parquet": {
        "url": f"https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual/resolve/{GRETEL_REV}/data/English_test-00000-of-00001.parquet",
        "license": "Apache-2.0", "origin": "gretelai/synthetic_pii_finance_multilingual (English test split)"},
}
# 선택 소스 (--with-gretel-ko): Gretel 원본은 Apache-2.0 이며, 이 한국어 번역본에는 별도 라이선스가 명시되어 있지 않다.
OPTIONAL_SOURCES = {
    "synthetic_pii_finance_multilingual-kr.jsonl": {
        "url": f"https://raw.githubusercontent.com/skan0779/korean-guardrail-dataset/{KGD_REV}/data/translated/synthetic_pii_finance_multilingual-kr.jsonl",
        "license": "원본 Apache-2.0 / 번역본 라이선스 미명시",
        "origin": "gretelai/synthetic_pii_finance_multilingual 의 한국어 번역 (skan0779/korean-guardrail-dataset)"},
}

SPLITS = (("calib", 20), ("test", 50), ("train", 100))


def locate_forms(text: str, forms: List[Dict]) -> List[Dict]:
    """값(form)만 있는 라벨(KDPII, Gretel-KO)을 원문 위치로 바꾼다. 못 찾으면 start=None.
    스팬 위치는 피드백 루프의 대체값(surrogate) 치환에 쓰인다."""
    out, used = [], []
    for f in forms:
        val, pos, start = f["form"], 0, -1
        while val:
            start = text.find(val, pos)
            if start < 0 or not any(start < e and s_ < start + len(val) for s_, e in used):
                break
            pos = start + 1
        if val and start >= 0:
            used.append((start, start + len(val)))
            out.append({"type": f["label"], "start": start, "end": start + len(val)})
        else:
            out.append({"type": f["label"], "start": None, "end": None})
    return out


def split_of(key: str) -> str:
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    for name, upper in SPLITS:
        if h < upper:
            return name
    return "train"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path, offline: bool = False, optional: bool = False) -> Dict[str, Dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    sources = dict(SOURCES, **(OPTIONAL_SOURCES if optional else {}))
    for name, src in sources.items():
        dst = raw_dir / name
        if not dst.exists():
            if offline:
                print(f"[prepare] 없음(오프라인): {dst} — 해당 데이터셋은 건너뜁니다")
                continue
            print(f"[prepare] download {src['url']}")
            tmp = dst.with_suffix(dst.suffix + ".part")
            req = urllib.request.Request(src["url"], headers={"User-Agent": "piiguard-laya-prepare"})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, dst)
        if src.get("md5") and _md5(dst) != src["md5"]:
            raise SystemExit(f"[prepare] {dst} md5 불일치 — 원본이 바뀌었거나 손상됐습니다 (기대값 {src['md5']})")
        info[name] = {**src, "sha256": sha256_file(dst), "bytes": dst.stat().st_size}
    return info


def _with_cat(spans, map_source):
    return [{**s, "category": T.map_label(s["type"], map_source)} for s in spans]


def _row(id_, suite, lang, text, label, types, source, license_, split, variant="original", map_source=None, **extra):
    cats = sorted({c for c in (T.map_label(t, map_source) for t in types) if c}) if map_source else []
    if "spans" in extra and map_source:
        extra["spans"] = _with_cat(extra["spans"], map_source)
    return {"id": id_, "suite": suite, "lang": lang, "text": text, "label": int(label), "categories": cats,
            "primary": T.primary_category(cats), "types": sorted(set(types)), "variant": variant, "split": split,
            "source": source, "license": license_, **extra}


def load_kdpii(path: Path) -> List[Dict]:
    """KDPII 원본(Zenodo, CC BY 4.0): 대화 목록 → 문장 단위 행. PNE(개인정보 개체)의 begin/end 를 스팬으로 쓴다."""
    rows = []
    for dialog in json.loads(path.read_text(encoding="utf-8")):
        for sent in dialog.get("sentence", []):
            text = sent.get("form") or ""
            if not text.strip():
                continue
            spans, types = [], []
            for e in sent.get("PNE", []) or []:
                b, en, lab = e.get("begin"), e.get("end"), e.get("label")
                types.append(lab)
                if isinstance(b, int) and isinstance(en, int) and text[b:en] == e.get("form"):
                    spans.append({"type": lab, "start": b, "end": en})
                else:  # 오프셋이 맞지 않으면 값으로 다시 찾는다
                    spans.extend(locate_forms(text, [{"form": e.get("form", ""), "label": lab}]))
            sid = f"kdpii-{sent['id']}"
            rows.append(_row(sid, "kdpii_ko", "ko", text, 1 if types else 0, types, "kdpii", "CC-BY-4.0",
                             split_of(sid), map_source="kdpii", spans=spans))
    return rows


def load_gretel_ko(path: Path) -> List[Dict]:
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1  # 원본 파일에 이스케이프가 깨진 줄이 3개 있다
                continue
            types = [a["label"] for a in r.get("answer", [])]
            row = _row(f"gretel-ko-{r['id']}", "gretel_fin_ko", "ko", r["query"], 1, types, "gretel-ko", "Apache-2.0",
                       split_of(r["id"]), map_source="gretel", spans=locate_forms(r["query"], r.get("answer", [])))
            if row["categories"]:  # company/date/time 만 있는 문서는 '개인정보' 양성이 아니므로 제외
                rows.append(row)
    if bad:
        print(f"[prepare] gretel_fin_ko: JSON 파싱 불가 {bad}줄 제외")
    return rows


def load_gretel_en(path: Path) -> List[Dict]:
    import pyarrow.parquet as pq

    tbl = pq.read_table(path).to_pylist()
    rows = []
    for i, r in enumerate(tbl):
        text = r.get("generated_text") or ""
        spans = r.get("pii_spans")
        if isinstance(spans, str):
            try:
                spans = json.loads(spans)
            except json.JSONDecodeError:
                spans = []
        spans = [s for s in (spans or []) if isinstance(s, dict) and "label" in s]
        types = [s["label"] for s in spans]
        sp = [{"type": s["label"], "start": s.get("start"), "end": s.get("end")} for s in spans
              if isinstance(s.get("start"), int) and isinstance(s.get("end"), int) and 0 <= s["start"] < s["end"] <= len(text)]
        rid = f"gretel-en-{i:05d}"
        row = _row(rid, "gretel_fin_en", "en", text, 1, types, "gretel-en", "Apache-2.0", split_of(rid),
                   map_source="gretel", document_type=r.get("document_type"), spans=sp)
        if text.strip() and row["categories"]:
            rows.append(row)
    return rows


def synth_rows(seed: int = 0) -> List[Dict]:
    out = []
    for r in synth.generate(seed=seed):
        suite = f"synth_fin_{r['lang']}"
        out.append(_row(r["id"], suite, r["lang"], r["text"], r["label"], [s["type"] for s in r["spans"]], "synth",
                        "Apache-2.0 (generated)", split_of(r["template"]), map_source="synth",
                        template=r["template"], group=r["group"], spans=r["spans"]))
    return out


def evasion_from(synth_all: List[Dict]) -> List[Dict]:
    out = []
    for r in perturb.evasion_rows(synth_all):
        lang = r["lang"]
        out.append({**r, "suite": f"evasion_{lang}", "source": "synth+perturb"})
    return out


DATASET_INFO = {
    "kdpii_ko": {"title": "KDPII 한국어 대화", "language": "ko", "license": "CC BY 4.0",
                 "description": "공개 한국어 대화 데이터. 이름·연락처·직장 등 33종 개인정보 라벨",
                 "source": "KDPII DATASET REVISED — Yonsei University HamSaeM Kim's Lab, TSCIENTIFIC Co.,Ltd. doi:10.5281/zenodo.16759166"},
    "synth_fin_ko": {"title": "금융 상담 문장 (한국어)", "language": "ko", "license": "Apache-2.0",
                     "description": "개인정보 문장 + 개인정보처럼 보이는 정상 문장 + 코드·로그 (합성)", "source": "piiguard-laya synth.py"},
    "synth_fin_en": {"title": "금융 상담 문장 (영어)", "language": "en", "license": "Apache-2.0",
                     "description": "위와 같은 구성의 영어 문장 (합성)", "source": "piiguard-laya synth.py"},
    "evasion_ko": {"title": "변조 입력 (한국어)", "language": "ko", "license": "Apache-2.0",
                   "description": "개인정보 값의 겉모양만 바꾼 문장 (전각, 보이지 않는 공백, 숫자 단어화 등)", "source": "piiguard-laya perturb.py"},
    "evasion_en": {"title": "변조 입력 (영어)", "language": "en", "license": "Apache-2.0",
                   "description": "위와 같은 방식의 영어 문장", "source": "piiguard-laya perturb.py"},
    "gretel_fin_en": {"title": "금융 문서 (영어, Gretel)", "language": "en", "license": "Apache-2.0",
                      "description": "계약서·명세서 같은 긴 문서. 개인정보 문서만 있음",
                      "source": "gretelai/synthetic_pii_finance_multilingual (English test split)"},
    "gretel_fin_ko": {"title": "금융 문서 (한국어 번역, Gretel)", "language": "ko", "license": "원본 Apache-2.0, 번역본 라이선스 명시 없음",
                      "description": "위 문서의 한국어 번역. 개인정보 문서만 있음",
                      "source": "skan0779/korean-guardrail-dataset (gretelai 번역본)"},
}
KEEP = ("id", "text", "label", "categories", "lang", "split", "variant", "group", "spans", "types", "source", "parent_id", "template")


def _dedupe(rows: List[Dict]) -> List[Dict]:
    """같은 문장은 첫 행만 남긴다 (값 칸이 없는 합성 템플릿, 짧은 대화 응답 등). 분할을 넘나드는 중복도 함께 사라진다."""
    seen, out = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append(r)
    return out


def _slim(r: Dict) -> Dict:
    return {k: r[k] for k in KEEP if k in r and r[k] not in (None, [], "")}


def _seed_sample(rows: List[Dict], k: int, salt: str) -> List[Dict]:
    """결정적 표본: 정상 문장을 k/2 건까지 먼저 뽑고 나머지를 개인정보 문장으로 채운다 (정상 문장이 적으면 개인정보 쪽이 많아진다)."""
    key = lambda r: hashlib.sha256((salt + r["id"]).encode()).hexdigest()
    pos = sorted((r for r in rows if r["label"] == 1), key=key)
    neg = sorted((r for r in rows if r["label"] == 0), key=key)
    if not neg:
        return pos[:k]
    half = k // 2
    take_neg = neg[:max(half, k - len(pos))]
    return pos[:k - len(take_neg)] + take_neg


# 데이터셋별 원본 파일 (합성·변조 데이터셋은 원본 파일 없이 seed 로 생성)
RAW_OF = {"kdpii_ko": "PII_dataset_V3.json", "gretel_fin_en": "English_test-00000-of-00001.parquet",
          "gretel_fin_ko": "synthetic_pii_finance_multilingual-kr.jsonl"}


def prepare(home: Path, raw_dir: Optional[Path] = None, offline: bool = False, download_only: bool = False,
            seed: int = 0, with_gretel_ko: bool = False, seed_per_dataset: int = 1500) -> Dict:
    raw_dir = raw_dir or paths.downloads_dir(home)
    src_info = download(raw_dir, offline=offline, optional=with_gretel_ko)
    if download_only:
        return {"raw": src_info}

    suites: Dict[str, List[Dict]] = {}
    if (raw_dir / "PII_dataset_V3.json").exists():
        suites["kdpii_ko"] = load_kdpii(raw_dir / "PII_dataset_V3.json")
    if with_gretel_ko and (raw_dir / "synthetic_pii_finance_multilingual-kr.jsonl").exists():
        suites["gretel_fin_ko"] = load_gretel_ko(raw_dir / "synthetic_pii_finance_multilingual-kr.jsonl")
    if (raw_dir / "English_test-00000-of-00001.parquet").exists():
        suites["gretel_fin_en"] = load_gretel_en(raw_dir / "English_test-00000-of-00001.parquet")
    syn = _dedupe(synth_rows(seed))
    for r in syn:
        suites.setdefault(r["suite"], []).append(r)
    for r in evasion_from(syn):
        suites.setdefault(r["suite"], []).append(r)
    suites = {k: _dedupe(v) for k, v in suites.items()}

    seed_rows, report = [], {}
    for name, rows in sorted(suites.items()):
        info = DATASET_INFO.get(name, {"title": name})
        evals = [_slim(r) for r in rows if r["split"] in ("calib", "test")]
        train = [_slim(r) for r in rows if r["split"] == "train"]
        raw = {k: {"sha256": v["sha256"]} for k, v in src_info.items() if k == RAW_OF.get(name)}
        D.write(paths.datasets_dir(home), name, evals, {**info, "prepared_by": "piiguard-laya prepare", "seed": seed,
                                                       **({"raw": raw} if raw else {})})
        picked = _seed_sample(train, seed_per_dataset, f"seed{seed}")
        for r in picked:
            seed_rows.append({**r, "id": f"{name}:{r['id']}", "split": "train", "source_dataset": name, "source": "prepare"})
        report[name] = {"eval": len(evals), "train_pool": len(train), "seed": len(picked)}
        print(f"[prepare] {name:15s} 평가용 {len(evals):6d}건 · 학습용 표본 {len(picked):5d}건 (train 분할 {len(train)}건 중)")
    seed_rows = _dedupe(seed_rows)
    D.write(paths.training_dir(home), "seed", seed_rows, {
        "title": "기본 학습 데이터 (seed)", "language": "ko/en",
        "description": f"평가용 데이터셋과 겹치지 않는 train 분할에서 데이터셋마다 최대 {seed_per_dataset}건 (정상 문장을 절반까지 먼저 뽑고 나머지는 개인정보 문장)",
        "license": "원본 데이터셋 라이선스를 따름 (KDPII CC BY 4.0 등)", "source": "piiguard-laya prepare"})
    print(f"[prepare] training/seed {len(seed_rows)}건")
    return {"raw": src_info, "datasets": report, "seed": len(seed_rows)}
