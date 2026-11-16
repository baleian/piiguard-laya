# piiguard-laya — 플랫폼 자동 감지 실행기
#
#   make prepare
#   make bench MODEL=laya-multilingual [DATASET=kdpii_ko,synth_fin_ko] [LIMIT=500]
#   make train BASE=laya-multilingual NAME=r1 DATA=seed
#   make site
#
# PLATFORM=cuda|cpu|mac 로 강제할 수 있다 (기본: scripts/platform.sh 자동 감지)
#   cuda : Docker + NVIDIA (Windows WSL2, Linux)            → compose 서비스 cuda
#   cpu  : Docker CPU (linux/amd64, linux/arm64, 맥의 Docker) → compose 서비스 cpu
#   mac  : Apple Silicon 네이티브 (.venv, MLX 추론 / MPS 학습)  → make mac-setup 으로 먼저 설치

PLATFORM ?= $(shell scripts/platform.sh)
MODEL    ?= laya-multilingual
export PIIGUARD_LAYA_UID := $(shell id -u)
export PIIGUARD_LAYA_GID := $(shell id -g)
# Hugging Face 모델은 시스템의 HF_HOME (기본 ~/.cache/huggingface) 에 받는다. Docker 에서도 같은 폴더를 쓴다.
export PIIGUARD_LAYA_HF_HOME := $(or $(HF_HOME),$(HOME)/.cache/huggingface)

ifeq ($(PLATFORM),mac)
  BACKEND ?= mlx
  DEVICE  ?= mps
  TRAIN_DEVICE ?= mps
  CLI = PIIGUARD_LAYA_HOME=$(CURDIR) .venv/bin/piiguard-laya
else ifeq ($(PLATFORM),cuda)
  BACKEND ?= torch
  DEVICE  ?= cuda
  TRAIN_DEVICE ?= cuda
  CLI = docker compose run --rm cuda
else
  BACKEND ?= torch
  DEVICE  ?= cpu
  TRAIN_DEVICE ?= cpu
  CLI = docker compose run --rm cpu
endif

.PHONY: help platform build build-cpu build-cuda build-multiarch mac-setup dirs env prepare prepare-download \
        datasets models bench train site compare review-export review-import inspect export-mlx test clean-reports

help:
	@echo "PLATFORM=$(PLATFORM) BACKEND=$(BACKEND) DEVICE=$(DEVICE)"
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  make %-16s %s\n", $$1, $$2}'

platform: ## 감지된 플랫폼 출력
	@echo $(PLATFORM)

build: ## 감지된 플랫폼용 이미지/환경 준비
ifeq ($(PLATFORM),mac)
	scripts/mac_native.sh
else ifeq ($(PLATFORM),cuda)
	docker compose --profile cuda build cuda
else
	docker compose build cpu
endif

build-cpu: ## CPU 이미지 (현재 아키텍처)
	docker compose build cpu

build-cuda: ## NVIDIA 이미지 (torch cu130)
	docker compose --profile cuda build cuda

build-multiarch: ## CPU 이미지 amd64+arm64 (IMAGE=repo/piiguard-laya:cpu PUSH=1 로 레지스트리에 올림)
	docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile --build-arg TORCH_INDEX=cpu \
	  -t $${IMAGE:-piiguard-laya:cpu-multiarch} $${PUSH:+--push} .

mac-setup: ## (Apple Silicon) 네이티브 .venv: torch(MPS) + laya + laya-mlx
	scripts/mac_native.sh

dirs:
	@mkdir -p datasets training models/trained reports feedback "$(PIIGUARD_LAYA_HF_HOME)"

env: dirs ## 장치·백엔드 확인
	$(CLI) env

prepare: dirs ## 기본 데이터셋 내려받기·변환 → datasets/, training/seed (오프라인: OFFLINE=1)
	$(CLI) prepare $(if $(OFFLINE),--offline) $(if $(WITH_GRETEL_KO),--with-gretel-ko) $(if $(SEED_PER_DATASET),--seed-per-dataset $(SEED_PER_DATASET))

prepare-download: dirs ## 원본 파일만 datasets/.downloads 에 받기 (오프라인 환경으로 옮길 때)
	$(CLI) prepare --download-only $(if $(WITH_GRETEL_KO),--with-gretel-ko)

datasets: dirs ## 평가용·학습용 데이터셋 목록
	$(CLI) datasets

models: dirs ## 쓸 수 있는 모델 목록
	$(CLI) models

bench: dirs ## 평가: MODEL=이름 [DATASET=a,b] [LIMIT=N] [FORCE=1] [RECALIBRATE=1]
	$(CLI) bench --model $(MODEL) --backend $(BACKEND) --device $(DEVICE) $(if $(DATASET),--dataset $(DATASET)) \
	  $(if $(LIMIT),--limit $(LIMIT)) $(if $(FORCE),--force) $(if $(RECALIBRATE),--recalibrate) $(if $(BATCH),--batch-size $(BATCH))

train: dirs ## 학습: BASE=기준모델 NAME=새이름 DATA=학습데이터[,..] [EPOCHS=2] [OVERWRITE=1]
	@test -n "$(NAME)" || (echo "NAME=새 모델 이름 이 필요합니다 (예: make train BASE=laya-multilingual NAME=r1 DATA=seed)"; exit 1)
	@test -n "$(DATA)" || (echo "DATA=학습용 데이터셋 이름 이 필요합니다 (make datasets 로 확인)"; exit 1)
	$(CLI) train --base $(or $(BASE),laya-multilingual) --name $(NAME) --data $(DATA) --device $(TRAIN_DEVICE) \
	  $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(MICRO_BATCH),--micro-batch $(MICRO_BATCH)) $(if $(GRAD_ACCUM),--grad-accum $(GRAD_ACCUM)) \
	  $(if $(OVERWRITE),--overwrite)

site: dirs ## 결과 페이지 다시 만들기 → reports/index.html (bench·train 뒤에는 자동)
	$(CLI) site

compare: dirs ## 회귀 검사: MODEL=후보 BASE=기준
	$(CLI) compare --model $(MODEL) --base $(BASE)

review-export: dirs ## 운영 로그 → 검토 큐: LOG=feedback/logs.jsonl MODEL=이름 [OUT=feedback/queue.csv] [LIMIT=500]
	@test -n "$(LOG)" || (echo "LOG=로그 파일 이 필요합니다 (feedback/ 아래에 두세요. 예: LOG=feedback/logs.jsonl)"; exit 1)
	$(CLI) review-export --log $(LOG) --model $(MODEL) --out $(or $(OUT),feedback/queue.csv) --device $(DEVICE) --backend $(BACKEND) \
	  $(if $(LIMIT),--limit $(LIMIT))

review-import: dirs ## 검토 결과 → 학습용 데이터셋: FILE=feedback/queue.csv INTO=이름
	@test -n "$(FILE)" -a -n "$(INTO)" || (echo "FILE=검토한 CSV INTO=학습용 데이터셋 이름 이 필요합니다 (예: FILE=feedback/queue.csv INTO=review-2026-11)"; exit 1)
	$(CLI) review-import $(FILE) --into $(INTO)

inspect: dirs ## 질문 문구 토큰 예산 점검
	$(CLI) inspect-questions --model $(MODEL) --device $(DEVICE)

export-mlx: dirs ## (Apple Silicon) 학습 모델 → MLX: NAME=이름
	$(CLI) export-mlx $(NAME)

test: dirs ## 단위 테스트 (모델 다운로드 없음)
ifeq ($(PLATFORM),mac)
	.venv/bin/python -m pytest -q tests
else
	docker compose $(if $(filter cuda,$(PLATFORM)),--profile cuda) run --rm --entrypoint python $(if $(filter cuda,$(PLATFORM)),cuda,cpu) -m pytest -q -p no:cacheprovider /opt/piiguard-laya/tests
endif

clean-reports: ## 평가 결과 지우기: MODEL=이름 (그 모델만) 또는 ALL=1 (전부)
ifeq ($(ALL),1)
	rm -rf reports/*
	@echo "reports/ 를 비웠습니다."
else ifeq ($(origin MODEL),command line)
	rm -rf "reports/$(MODEL)"
	$(CLI) site
else
	@echo "지울 대상을 지정하세요: make clean-reports MODEL=이름  또는  make clean-reports ALL=1"; exit 1
endif
