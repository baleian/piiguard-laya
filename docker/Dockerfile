# syntax=docker/dockerfile:1
# piiguard-laya: Laya 기반 PII 가드레일 벤치마크 · 피드백 루프 이미지
#
# 하나의 Dockerfile 로 모든 Linux 타깃을 만든다.
#   CPU  amd64 / arm64 : --build-arg TORCH_INDEX=cpu      (기본값, Apple Silicon 의 Docker 도 여기)
#   NVIDIA amd64/arm64 : --build-arg TORCH_INDEX=cu130    (RTX 50xx Blackwell 은 cu128 이상 필요)
#
# Apple Silicon 의 Metal(MPS) GPU 는 Docker 컨테이너 안에서 쓸 수 없다(가상화된 Linux VM).
# 맥에서 GPU 가속이 필요하면 scripts/mac_native.sh 로 같은 코드를 네이티브로 실행한다.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv --upgrade-deps /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG TORCH_INDEX=cpu
ARG TORCH_VERSION=2.14.0
RUN pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

# 의존성 레이어를 소스와 분리해 캐시한다 (소스만 바뀌면 이 레이어는 재사용).
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-deps . && pip check

FROM ${PYTHON_IMAGE} AS runtime
ARG TORCH_INDEX=cpu
LABEL org.opencontainers.image.title="piiguard-laya" \
      org.opencontainers.image.description="Laya-based PII guardrail benchmark and feedback loop" \
      org.opencontainers.image.licenses="Apache-2.0" \
      piiguard-laya.torch_index="${TORCH_INDEX}"

# torch 2.14 는 일부 CUDA 연산을 첫 추론 때 Triton 으로 컴파일하는데 slim 이미지엔 C 컴파일러가 없다.
# (laya Dockerfile #365 와 동일한 회피책: 기본 커널을 쓰며 결과·속도 동일)
ENV PATH="/opt/venv/bin:$PATH" \
    TORCH_DISABLE_NATIVE_JIT=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    USE_TF=0 USE_TORCH=1 \
    HF_HOME=/hf \
    PIIGUARD_LAYA_HOME=/workspace \
    PIIGUARD_LAYA_TORCH_INDEX=${TORCH_INDEX}

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
COPY --from=build /opt/venv /opt/venv
WORKDIR /workspace
COPY --chown=app:app configs/ /workspace/configs/
COPY --chown=app:app tests/ /opt/piiguard-laya/tests/
RUN mkdir -p /workspace/datasets /workspace/training /workspace/models/trained /workspace/reports /workspace/feedback /hf \
    && chown -R app:app /workspace /hf
USER app

ENTRYPOINT ["piiguard-laya"]
CMD ["--help"]
