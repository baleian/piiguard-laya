"""평가·학습 대상 모델 찾기.

모델은 models/ 아래 경로로 지정한다 (`MODEL=` / `BASE=`). 이 경로가 모델 id 이고, 결과도 같은 경로에 쌓인다.
    base/<이름>                models/base/<이름>/     Base 모델 (Hugging Face 에서 받은 원본)  → reports/base/<이름>/
    trained/<이름>             models/trained/<이름>/  학습된 모델 (make train 결과)             → reports/trained/<이름>/
    hf:org/repo[:하위폴더]      Hugging Face 저장소를 models/base/org--repo[--하위폴더]/ 에 받아 base/<그 이름> 으로 쓴다

base/laya-multilingual, base/laya-english 는 models/base/ 에 없으면 처음 쓸 때 Hugging Face 에서 받는다.
받은 저장소·커밋은 그 폴더의 MODEL_CARD.json 에 남는다. 체크포인트 폴더를 models/base/<이름>/ 에 직접 넣어도 된다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths

ALIASES: Dict[str, Dict[str, Any]] = {
    "laya-multilingual": {"repo": "convaiinnovations/laya", "subfolder": "multilingual",
                          "description": "Laya multilingual (한국어·영어 등 100+ 언어)"},
    "laya-english": {"repo": "convaiinnovations/laya", "subfolder": None, "description": "Laya 영어 전용"},
}
KIND_LABEL = {"base": "Base 모델", "trained": "학습된 모델"}
DEFAULT_BASE = "base/laya-multilingual"
CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
CHECKPOINT_ENTRIES = ("rl_agent_config.json", "model.safetensors", "tokenizer", "encoder")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def is_checkpoint(d: Path) -> bool:
    return d.is_dir() and (d / "rl_agent_config.json").exists() and (d / "model.safetensors").exists()


def report_name(spec: str) -> str:
    return re.sub(r"[/:]+", "--", spec.strip().strip("/"))


def _read_card(d: Path) -> Optional[Dict[str, Any]]:
    p = d / "MODEL_CARD.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _rel(p: Path) -> str:
    try:  # 프로젝트 기준 상대 경로 (컨테이너 · 호스트 어디서 봐도 같은 값)
        return str(Path(p).resolve().relative_to(paths.home().resolve()))
    except ValueError:
        return str(p)


@dataclass
class ModelRef:
    spec: str
    name: str
    kind: str                                  # "base" | "trained"
    path: Path                                 # models/base/<이름> 또는 models/trained/<이름>
    card: Optional[Dict[str, Any]] = None      # 폴더의 MODEL_CARD.json
    source: Optional[Dict[str, Any]] = None    # 아직 받지 않은 Base 모델의 Hugging Face 위치 {repo, subfolder, revision}
    description: str = ""

    @property
    def id(self) -> str:
        """모델 id = models/ 아래 경로 (base/<이름> 또는 trained/<이름>). 결과 폴더 reports/<id>/ 도 같다."""
        return f"{self.kind}/{self.name}"

    def local_dir(self) -> Path:
        """체크포인트 폴더. Base 모델이 아직 없으면 이때 Hugging Face 에서 받는다."""
        if not is_checkpoint(self.path):
            if self.kind != "base" or not self.source:
                raise SystemExit(f"{self.spec}: 체크포인트(rl_agent_config.json, model.safetensors)가 없습니다 → {_rel(self.path)}")
            download_base(self)
        return self.path

    def mlx_source(self) -> str:
        """MLX 백엔드용 체크포인트. 처음 쓸 때 <모델 폴더>/mlx/ 로 변환해 둔다 (다시 학습하면 폴더째 지워진다)."""
        d = self.local_dir() / "mlx"
        if not (d.is_dir() and any(d.iterdir())):
            export_mlx(self)
        return str(d)

    def fingerprint(self) -> str:
        """모델 파일이 바뀌면(다시 학습·다시 받음) 달라지는 값 (리포트 캐시 무효화용)."""
        d = self.local_dir()
        h = hashlib.sha256()
        for f in ("rl_agent_config.json", "model.safetensors"):
            st = (d / f).stat()
            h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
        return h.hexdigest()[:16]

    def describe(self) -> Dict[str, Any]:
        out = {"id": self.id, "name": self.name, "kind": self.kind, "kind_label": KIND_LABEL[self.kind],
               "description": self.description, "path": _rel(self.path)}
        if self.kind == "base":
            c = self.card or {}
            out.update({"source": c.get("source") or self.source, "license": c.get("license")})
        elif self.card:
            out["card"] = card_summary(self.card)
        return out


def card_summary(card: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("name", "created_at", "base", "lineage", "training_data", "question", "objective", "config", "train_eval",
            "leakage_check", "epoch_loss", "env")
    return {k: card.get(k) for k in keys if k in card}


def download_base(ref: ModelRef) -> Path:
    """Hugging Face 저장소(하위 폴더)의 Laya 체크포인트를 models/base/<이름>/ 에 받고 MODEL_CARD.json 을 쓴다."""
    from huggingface_hub import HfApi, snapshot_download

    src, token = ref.source or {}, os.environ.get("HF_TOKEN") or None
    where = src["repo"] + (f" ({src['subfolder']})" if src.get("subfolder") else "")
    try:
        info = HfApi().model_info(src["repo"], revision=src.get("revision"), token=token)
    except Exception as e:  # noqa: BLE001 — 오프라인, 권한, 이름 오류
        raise SystemExit(f"{ref.spec}: {_rel(ref.path)}/ 에 모델이 없고 Hugging Face {where} 에서 받지 못했습니다 ({type(e).__name__}).\n"
                         f"  인터넷이 안 되는 환경이면 체크포인트 폴더를 {_rel(ref.path)}/ 에 직접 넣으세요.")
    prefix = f"{src['subfolder']}/" if src.get("subfolder") else ""
    tmp = ref.path.parent / f".{ref.name}.download"
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[model] Hugging Face {where} @ {info.sha[:10]} → {_rel(ref.path)}/")
    snapshot_download(src["repo"], revision=info.sha, token=token, local_dir=tmp,
                      allow_patterns=[prefix + p for p in CHECKPOINT_FILES])
    got = tmp / src["subfolder"] if src.get("subfolder") else tmp
    if not is_checkpoint(got):
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"{ref.spec}: {where} 에서 Laya 체크포인트(rl_agent_config.json, model.safetensors)를 찾지 못했습니다.")
    ref.path.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_ENTRIES:
        if (got / name).exists():
            shutil.move(str(got / name), str(ref.path / name))
    shutil.rmtree(tmp, ignore_errors=True)
    license_ = getattr(info.card_data, "license", None)
    ref.card = write_base_card(ref.path, ref.name, src["repo"], src.get("subfolder"), info.sha, ref.description, license_)
    return ref.path


def write_base_card(d: Path, name: str, repo: str, subfolder: Optional[str], commit: str, description: str,
                    license_: Optional[str] = None) -> Dict[str, Any]:
    card = {"name": name, "kind": "base", "description": description,
            "source": {"type": "huggingface", "repo": repo, "subfolder": subfolder, "commit": commit},
            "license": license_, "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (d / "MODEL_CARD.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    return card


def export_mlx(ref: ModelRef, dtype: str = "float16") -> Path:
    """모델 폴더(torch 체크포인트) → MLX 변환본 <모델 폴더>/mlx/ (Apple Silicon, laya-mlx 필요)."""
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("laya_mlx") is None:
        raise SystemExit("laya-mlx 가 없습니다. Apple Silicon 맥에서 `make mac-setup` 으로 설치하세요.")
    src = ref.local_dir()
    dst = src / "mlx"
    shutil.rmtree(dst, ignore_errors=True)
    # 지금 실행 중인 Python 의 laya-mlx 를 쓴다 (.venv 를 activate 하지 않고 make 로 실행해도 찾도록)
    cmd = [sys.executable, "-m", "laya_mlx", "convert", "--model", str(src), "--dtype", dtype, "--output", str(dst)]
    print("[mlx] 변환:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dst


def _local(kind: str, name: str, d: Path, spec: Optional[str] = None) -> ModelRef:
    card = _read_card(d)
    return ModelRef(spec=spec or f"{kind}/{name}", name=name, kind=kind, path=d, card=card,
                    description=(card or {}).get("description") or ((ALIASES.get(name) or {}).get("description") or KIND_LABEL[kind]))


def _root(kind: str, home: Path) -> Path:
    return paths.base_dir(home) if kind == "base" else paths.trained_dir(home)


def resolve(spec: str, home: Optional[Path] = None, revision: Optional[str] = None) -> ModelRef:
    """base/<이름> · trained/<이름> · hf:org/repo[:하위폴더] → ModelRef. 다른 형식은 받지 않는다."""
    home = home or paths.home()
    spec = (spec or "").strip().strip("/")
    rev = revision or os.environ.get("LAYA_REVISION") or None
    if spec.startswith("hf:"):
        repo, _, sub = spec[3:].partition(":")
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise SystemExit(f"Hugging Face 저장소는 hf:org/repo 또는 hf:org/repo:하위폴더 로 지정하세요 (받은 값: '{spec}')")
        name = report_name(repo + (f":{sub}" if sub else ""))
        d = paths.base_dir(home) / name
        if is_checkpoint(d):
            return _local("base", name, d, spec=spec)
        return ModelRef(spec=spec, name=name, kind="base", path=d, source={"repo": repo, "subfolder": sub or None, "revision": rev},
                        description=f"Hugging Face {repo}" + (f" ({sub})" if sub else ""))
    kind, sep, name = spec.partition("/")
    if not sep or kind not in KIND_LABEL or not NAME_RE.match(name):
        found = [f"{k}/{spec}" for k in KIND_LABEL if NAME_RE.match(spec) and is_checkpoint(_root(k, home) / spec)]
        hint = f" 혹시 {' 또는 '.join(found)} 인가요?" if found else ""
        raise SystemExit(f"모델은 base/<이름> 또는 trained/<이름> 으로 지정하세요 (받은 값: '{spec}').{hint} 목록: make models")
    d = _root(kind, home) / name
    if is_checkpoint(d):
        return _local(kind, name, d)
    if kind == "base" and name in ALIASES:  # 기본 Base 모델: 처음 쓸 때 받는다
        a = ALIASES[name]
        return ModelRef(spec=spec, name=name, kind="base", path=d, description=a["description"],
                        source={"repo": a["repo"], "subfolder": a["subfolder"], "revision": rev})
    have = ", ".join(m["id"] for m in list_local(home)) or "없음"
    extra = " Hugging Face 저장소는 MODEL=hf:org/repo[:하위폴더] 로 받는다." if kind == "base" else ""
    raise SystemExit(f"models/{spec}/ 에 모델이 없습니다. 있는 모델: {have}.{extra}")


def list_local(home: Optional[Path] = None) -> List[Dict[str, Any]]:
    """models/base/ 와 models/trained/ 에 있는 모델."""
    home = home or paths.home()
    out = []
    for kind in KIND_LABEL:
        root = _root(kind, home)
        for d in sorted(root.glob("*")) if root.exists() else []:
            if d.name.startswith(".") or not is_checkpoint(d):
                continue
            ref = _local(kind, d.name, d)
            card = ref.card or {}
            base = card.get("base") or {}
            out.append({**ref.describe(), "created_at": card.get("created_at") or card.get("downloaded_at"),
                        "base": base.get("id") or base.get("name"),
                        "training_data": [t.get("name") for t in card.get("training_data") or []]})
    return out


def list_models(home: Optional[Path] = None) -> List[Dict[str, Any]]:
    """받은·학습한 모델 + 아직 받지 않은 기본 Base 모델."""
    local = list_local(home)
    have = {m["id"] for m in local}
    pending = [{"id": f"base/{k}", "name": k, "kind": "base", "kind_label": KIND_LABEL["base"], "description": v["description"],
                "downloaded": False, "source": {"repo": v["repo"], "subfolder": v["subfolder"]}}
               for k, v in ALIASES.items() if f"base/{k}" not in have]
    return [{**m, "downloaded": True} for m in local] + pending


def validate_new_name(name: str, home: Optional[Path] = None, overwrite: bool = False) -> Path:
    home = home or paths.home()
    if not NAME_RE.match(name or ""):
        raise SystemExit(f"모델 이름 '{name}' 은 영문·숫자·. _ - 만 쓸 수 있습니다.")
    d = paths.trained_dir(home) / name
    if d.exists() and not overwrite:
        raise SystemExit(f"models/trained/{name} 이 이미 있습니다. 다른 이름을 쓰거나 OVERWRITE=1 로 덮어쓰세요.")
    return d
