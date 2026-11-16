"""평가·학습 대상 모델 찾기.

모델 지정 방법 (`MODEL=` / `BASE=`):
    laya-multilingual            미리 등록된 별칭 → Hugging Face convaiinnovations/laya (multilingual)
    laya-english                 미리 등록된 별칭 → Hugging Face convaiinnovations/laya (영어 전용)
    r1                           models/trained/r1/ 에 있는 직접 학습한 모델
    org/repo                     Hugging Face 저장소 (루트에 Laya 체크포인트)
    org/repo:subfolder           Hugging Face 저장소의 하위 폴더

Hugging Face 모델은 처음 쓸 때 HF_HOME(기본 ~/.cache/huggingface)에 내려받는다. 이미 있으면 다시 받지 않는다.
리포트 폴더 이름(reports/<이름>/)은 별칭·학습 모델은 그 이름, HF 저장소는 "/" 와 ":" 를 "--" 로 바꾼 이름이다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths

ALIASES: Dict[str, Dict[str, Any]] = {
    "laya-multilingual": {"repo": "convaiinnovations/laya", "subfolder": "multilingual", "mlx": "aac6fef/laya-multilingual-mlx",
                          "description": "Laya multilingual 기본 모델 (한국어·영어 등 100+ 언어)"},
    "laya-english": {"repo": "convaiinnovations/laya", "subfolder": None, "mlx": "aac6fef/laya-mlx",
                     "description": "Laya 영어 전용 기본 모델"},
}
CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def is_checkpoint(d: Path) -> bool:
    return d.is_dir() and (d / "rl_agent_config.json").exists()


def report_name(spec: str) -> str:
    return re.sub(r"[/:]+", "--", spec.strip().strip("/"))


@dataclass
class ModelRef:
    spec: str
    name: str
    kind: str                      # "hf" | "trained"
    repo: Optional[str] = None
    subfolder: Optional[str] = None
    revision: Optional[str] = None
    mlx: Optional[str] = None
    path: Optional[Path] = None    # 학습 모델 폴더, 또는 내려받은 HF 스냅샷 폴더
    card: Optional[Dict[str, Any]] = None
    commit: Optional[str] = None
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def local_dir(self) -> Path:
        """체크포인트가 있는 로컬 폴더. HF 모델은 이때 HF_HOME 에 내려받는다(있으면 건너뜀)."""
        if self.path is not None:
            return self.path
        from huggingface_hub import snapshot_download

        prefix = f"{self.subfolder}/" if self.subfolder else ""
        root = Path(snapshot_download(self.repo, revision=self.revision, token=os.environ.get("HF_TOKEN") or None,
                                      allow_patterns=[prefix + p for p in CHECKPOINT_FILES]))
        self.commit = root.name if root.parent.name == "snapshots" else None
        self.path = root / self.subfolder if self.subfolder else root
        if not is_checkpoint(self.path):
            raise SystemExit(f"{self.spec}: Laya 체크포인트(rl_agent_config.json)를 찾지 못했습니다 → {self.path}")
        return self.path

    def mlx_source(self) -> str:
        """MLX 백엔드용 체크포인트. 학습 모델은 처음 쓸 때 <모델>/mlx/ 로 변환해 둔다 (다시 학습하면 폴더째 지워진다)."""
        if self.kind == "trained":
            d = self.path / "mlx"
            if not (d.is_dir() and any(d.iterdir())):
                export_mlx(self)
            return str(d)
        return self.mlx or self.repo

    def fingerprint(self) -> str:
        """같은 이름으로 다시 학습했거나 HF 리비전이 바뀌면 달라지는 값 (리포트 캐시 무효화용)."""
        d = self.local_dir()
        h = hashlib.sha256()
        for f in ("rl_agent_config.json", "model.safetensors"):
            p = d / f
            st = p.stat()
            h.update(f"{f}:{st.st_size}:{int(st.st_mtime) if self.kind == 'trained' else 0}:{self.commit or ''}".encode())
        return h.hexdigest()[:16]

    def describe(self) -> Dict[str, Any]:
        out = {"name": self.name, "spec": self.spec, "kind": self.kind, "description": self.description}
        if self.kind == "hf":
            out.update({"repo": self.repo, "subfolder": self.subfolder, "revision": self.revision, "commit": self.commit})
        else:
            try:  # 프로젝트 기준 상대 경로 (컨테이너 · 호스트 어디서 봐도 같은 값)
                rel = str(Path(self.path).resolve().relative_to(paths.home().resolve()))
            except ValueError:
                rel = str(self.path)
            out.update({"path": rel})
        if self.card:
            out["card"] = card_summary(self.card)
        return out


def card_summary(card: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("name", "created_at", "base", "lineage", "training_data", "question_set", "objective", "config", "train_eval",
            "leakage_check", "env")
    return {k: card.get(k) for k in keys if k in card}


def export_mlx(ref: "ModelRef", dtype: str = "float16") -> Path:
    """학습 모델(torch) → MLX 변환본 models/trained/<이름>/mlx/ (Apple Silicon, laya-mlx 필요)."""
    import importlib.util
    import shutil
    import subprocess
    import sys

    if ref.kind != "trained":
        raise SystemExit("MLX 변환은 models/trained/ 의 학습 모델에만 씁니다 (기본 모델은 공개된 MLX 변환본을 바로 씁니다).")
    if importlib.util.find_spec("laya_mlx") is None:
        raise SystemExit("laya-mlx 가 없습니다. Apple Silicon 맥에서 `make mac-setup` 으로 설치하세요.")
    dst = ref.path / "mlx"
    shutil.rmtree(dst, ignore_errors=True)
    # 지금 실행 중인 Python 의 laya-mlx 를 쓴다 (.venv 를 activate 하지 않고 make 로 실행해도 찾도록)
    cmd = [sys.executable, "-m", "laya_mlx", "convert", "--model", str(ref.path), "--dtype", dtype, "--output", str(dst)]
    print("[mlx] 변환:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dst


def resolve(spec: str, home: Optional[Path] = None, revision: Optional[str] = None) -> ModelRef:
    home = home or paths.home()
    spec = (spec or "").strip()
    if not spec:
        raise SystemExit("모델을 지정하세요 (예: MODEL=laya-multilingual). 목록: piiguard-laya models")
    trained = paths.trained_dir(home) / spec
    if is_checkpoint(trained):
        card_p = trained / "MODEL_CARD.json"
        card = json.loads(card_p.read_text(encoding="utf-8")) if card_p.exists() else None
        return ModelRef(spec=spec, name=spec, kind="trained", path=trained, card=card,
                        description=(card or {}).get("description", "직접 학습한 모델"))
    if spec in ALIASES:
        a = ALIASES[spec]
        return ModelRef(spec=spec, name=spec, kind="hf", repo=a["repo"], subfolder=a["subfolder"], mlx=a["mlx"],
                        revision=revision or os.environ.get("LAYA_REVISION") or None, description=a["description"])
    if "/" in spec:
        repo, _, sub = spec.partition(":")
        return ModelRef(spec=spec, name=report_name(spec), kind="hf", repo=repo, subfolder=sub or None,
                        revision=revision or os.environ.get("LAYA_REVISION") or None,
                        description=f"Hugging Face {repo}" + (f" ({sub})" if sub else ""))
    names = ", ".join([*ALIASES, *[m["name"] for m in list_trained(home)]])
    raise SystemExit(f"모델 '{spec}' 을 찾지 못했습니다. 사용 가능: {names} 또는 Hugging Face 'org/repo[:하위폴더]'")


def list_trained(home: Optional[Path] = None) -> List[Dict[str, Any]]:
    home = home or paths.home()
    out = []
    root = paths.trained_dir(home)
    for d in sorted(root.glob("*")) if root.exists() else []:
        if not is_checkpoint(d):
            continue
        card_p = d / "MODEL_CARD.json"
        card = json.loads(card_p.read_text(encoding="utf-8")) if card_p.exists() else {}
        out.append({"name": d.name, "kind": "trained", "path": str(d), "created_at": card.get("created_at"),
                    "base": (card.get("base") or {}).get("name"),
                    "training_data": [t.get("name") for t in card.get("training_data") or []]})
    return out


def list_models(home: Optional[Path] = None) -> List[Dict[str, Any]]:
    out = [{"name": k, "kind": "hf", "repo": v["repo"], "subfolder": v["subfolder"], "description": v["description"]}
           for k, v in ALIASES.items()]
    return out + list_trained(home)


def validate_new_name(name: str, home: Optional[Path] = None, overwrite: bool = False) -> Path:
    if not NAME_RE.match(name or ""):
        raise SystemExit(f"모델 이름 '{name}' 은 영문·숫자·. _ - 만 쓸 수 있습니다.")
    if name in ALIASES:
        raise SystemExit(f"'{name}' 은 기본 모델 별칭이라 학습 모델 이름으로 쓸 수 없습니다.")
    d = paths.trained_dir(home or paths.home()) / name
    if d.exists() and not overwrite:
        raise SystemExit(f"models/trained/{name} 이 이미 있습니다. 다른 이름을 쓰거나 OVERWRITE=1 로 덮어쓰세요.")
    return d
