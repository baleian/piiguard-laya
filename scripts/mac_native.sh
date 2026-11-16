#!/usr/bin/env bash
# Apple Silicon 네이티브 환경 구성 (macOS 14+, Python 3.11+)
#
#   추론(벤치마크): laya-mlx  — MLX(Metal) 로 upstream 가중치를 그대로 실행. M3 Max 기준 짧은 질문 ~7 ms (multilingual)
#   학습(파인튜닝): torch MPS — upstream laya.train (RLCD) 가 MPS 를 지원
#   모델(models/base/, models/trained/)은 make bench 가 처음 평가할 때 <모델 폴더>/mlx/ 로 변환한다 (make export-mlx MODEL=trained/<이름>)
#
# 사용:  scripts/mac_native.sh            (.venv 생성 + 설치)
#        .venv/bin/piiguard-laya env           (mlx / mps 확인)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Apple Silicon macOS 전용 스크립트입니다. 다른 플랫폼은 Docker(make bench)를 쓰세요." >&2
  exit 1
fi

PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null 2>&1 || PY=python3.11
command -v "$PY" >/dev/null 2>&1 || { echo "python3.11+ 가 필요합니다 (brew install python@3.12 또는 uv)" >&2; exit 1; }

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PY" .venv
  PIP="uv pip install --python .venv/bin/python"
else
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -U pip
  PIP=".venv/bin/python -m pip install"
fi

# macOS 용 torch 기본 wheel 에 MPS 가 들어 있다 (CUDA 인덱스 불필요)
$PIP "torch==${TORCH_VERSION:-2.14.0}"
$PIP -r docker/requirements.txt
$PIP "laya-mlx==${LAYA_MLX_VERSION:-0.6.0}"
$PIP --no-deps -e .

echo
.venv/bin/piiguard-laya env
cat <<'EOF'

다음 단계:
  make prepare                     # 기본 데이터셋 → datasets/, training/seed
  make bench                       # MLX 백엔드로 평가 (PLATFORM=mac 자동 감지)
  make train NAME=r1 DATA=seed     # MPS 로 학습
  make bench MODEL=trained/r1
EOF
