#!/usr/bin/env bash
# 실행 플랫폼 자동 판별: cuda | mac | cpu
#   cuda : Linux/WSL2 + nvidia-smi + Docker NVIDIA 런타임
#   mac  : macOS Apple Silicon → 네이티브 실행 (MLX 추론 / MPS 학습). Docker 는 Metal GPU 를 못 쓴다.
#   cpu  : 그 외 (Intel 맥, GPU 없는 Linux, ARM 서버) → Docker CPU 이미지
set -euo pipefail
os="$(uname -s)"; arch="$(uname -m)"
if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
  echo mac; exit 0
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  if docker info 2>/dev/null | grep -qiE 'runtimes:.*nvidia|nvidia'; then
    echo cuda; exit 0
  fi
fi
echo cpu
