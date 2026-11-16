# 실행 환경별 가이드

같은 코드와 설정을 모든 환경에서 쓴다. 바뀌는 것은 런타임(백엔드 · 장치)뿐이다. `make` 가 플랫폼을 자동 감지한다(`scripts/platform.sh`). `PLATFORM=cuda|cpu|mac` 으로 강제할 수 있다.

| 환경 | 감지값 | 실행 방식 | 추론 백엔드 | 학습 |
|---|---|---|---|---|
| Windows WSL2 / Linux + NVIDIA | `cuda` | Docker `piiguard-laya:cuda` (torch 2.14 + cu130) | torch CUDA, bf16 AMP | CUDA |
| Apple Silicon 맥 (네이티브) | `mac` | `.venv` (`scripts/mac_native.sh`) | **laya-mlx (MLX/Metal)** | torch MPS |
| Apple Silicon 맥 (Docker) | `PLATFORM=cpu` | Docker `piiguard-laya:cpu` linux/arm64 | torch CPU | CPU |
| Linux amd64/arm64 (GPU 없음), Intel 맥 | `cpu` | Docker `piiguard-laya:cpu` | torch CPU | CPU (느림) |
| NVIDIA arm64 (DGX Spark / GH200) | `cuda` | Docker `piiguard-laya:cuda` (cu130 aarch64 wheel) | torch CUDA | CUDA |

어느 환경이든 폴더 구조와 명령은 같다. Docker 컨테이너에는 `configs/`, `datasets/`, `training/`, `models/`, `reports/`, `feedback/` 가 연결된다. 컨테이너는 현재 사용자 id 로 실행되어 만든 파일의 소유자가 호스트 사용자와 같다.

## 모델 파일

모델은 프로젝트 안의 `models/` 에 있다. `models/base/<이름>/` 은 Base 모델(Hugging Face 에서 받은 원본), `models/trained/<이름>/` 은 학습된 모델이다. 모델은 `MODEL=base/<이름>` 또는 `MODEL=trained/<이름>` 으로 지정한다. Base 모델(`base/laya-multilingual`, `hf:org/repo` 등)은 `models/base/` 에 없을 때만 처음 한 번 Hugging Face 에서 받고, 받은 저장소 · 커밋을 그 폴더의 `MODEL_CARD.json` 에 남긴다. 네이티브 실행과 Docker 가 같은 폴더를 쓴다.

```bash
HF_TOKEN=hf_xxx make bench MODEL=hf:org/private-repo    # 비공개 저장소 → models/base/org--private-repo/ (이후 base/org--private-repo)
```

## 1. Windows WSL2 / Linux + NVIDIA

필요: Docker Desktop(WSL2 통합) 또는 WSL 안의 Docker Engine + NVIDIA Container Toolkit, Windows NVIDIA 드라이버.

```bash
make platform          # → cuda
make build             # piiguard-laya:cuda (최초 1회, ~10GB)
make test
make prepare
make bench LIMIT=200   # 데이터셋마다 200건으로 빠른 동작 확인 (1분 안팎)
make bench             # 전체 평가
```

- RTX 50xx(Blackwell, sm_120)는 CUDA 12.8 이상 빌드가 필요하다. 이미지는 cu130 을 쓴다.
- GPU 를 여러 장 쓰면 `GPU_ID=1 make bench` 로 고른다.
- GPU 배치 크기는 `BATCH=` (기본 `configs/bench.yaml` 의 32)로 바꾼다.
- 정밀도에 따라 확률이 미세하게 달라진다. `LAYA_CUDA_AMP=fp32 make bench MODEL=... FORCE=1 RECALIBRATE=1` 로 비교할 수 있다. 같은 모델의 결과를 덮어쓰므로 필요하면 `reports/<모델 id>` 를 먼저 복사해 둔다.

## 2. Apple Silicon 맥 — 네이티브 (권장)

Docker Desktop for Mac 의 컨테이너는 Linux VM 안에서 돌기 때문에 **Metal GPU 를 쓸 수 없다**. 그래서 맥에서는 네이티브 실행이 최적이다.

```bash
make mac-setup         # .venv: torch(MPS) + laya + laya-mlx 0.6.0 (Python 3.11+, macOS 14+)
make env               # laya_mlx: true, mps: true 확인
make prepare
make bench             # 추론은 MLX: 처음 평가할 때 models/base/laya-multilingual/mlx/ 로 자동 변환
make train NAME=r1 DATA=seed MICRO_BATCH=1 GRAD_ACCUM=32    # 학습은 MPS
make bench MODEL=trained/r1    # 학습된 모델도 처음 평가할 때 models/trained/r1/mlx/ 로 자동 변환
```

- **추론은 MLX**: `laya-mlx` 는 upstream 가중치를 MLX 로 옮긴 것이다. 공개 측정은 M3 Max 기준 multilingual 짧은 질문 P50 7.4 ms이며, upstream 과 63/63 검증 문항에서 같은 답을 냈다. 실제 수치는 결과 페이지의 지연시간 표로 확인한다.
- **학습은 MPS**: RLCD 학습은 upstream PyTorch 코드라 MPS 로 돌린다. 통합 메모리가 16GB 면 `MICRO_BATCH=1 GRAD_ACCUM=32` 정도를 쓴다.
- **MLX 변환**: Base 모델과 학습된 모델 모두 `make bench` 가 MLX 백엔드에서 변환본(`<모델 폴더>/mlx/`)이 없으면 laya-mlx 로 만든다. 직접 만들려면 `make export-mlx MODEL=trained/<이름>`.
- PyTorch MPS 로 추론하려면 `make bench BACKEND=torch DEVICE=mps`. 같은 모델 이름의 결과를 덮어쓴다.
- laya-mlx 0.6.0 은 토큰 예산 인자(`head_max_len`, `max_len`)를 받으므로 torch 와 같은 예산으로 판정한다. 예산 인자를 받지 않는 버전이면 예산 없이 다시 호출하는데, 이때는 체크포인트 기본값을 쓰므로 긴 지침이 잘릴 수 있다.
- 백엔드에 따라 확률이 미세하게 다르므로 **임계값은 실제 서빙할 백엔드로 측정한 결과에서 가져온다.**

## 3. CPU Docker (amd64 / arm64)

```bash
make build-cpu
PLATFORM=cpu make test
PLATFORM=cpu make bench LIMIT=100
OMP_NUM_THREADS=8 PLATFORM=cpu make bench
```

- 같은 Dockerfile 을 `TORCH_INDEX=cpu` 로 빌드한다. arm64 호스트(맥 Docker, Graviton)에서는 arm64 이미지가 자동으로 빌드된다.
- 레지스트리에 멀티아키 이미지를 올리려면 `make build-multiarch IMAGE=registry/piiguard-laya:cpu PUSH=1`.
- CPU 추론은 요청당 수백 ms 수준이다. 전체 평가는 수 시간이 걸릴 수 있으니 먼저 `LIMIT=` 으로 확인한다. 배치 크기는 `PIIGUARD_LAYA_BATCH`(기본 16).

## 4. 오프라인(air-gapped) 환경

인터넷이 되는 환경에서 이미지 · 원본 데이터 · 체크포인트를 준비한 뒤 오프라인 환경으로 옮긴다.

```bash
# (인터넷이 되는 환경)
make build && docker save piiguard-laya:cuda | gzip > piiguard-laya-cuda.tar.gz
make prepare-download                          # datasets/.downloads/ (원본 파일)
make models                                    # 받은 모델 확인. Base 모델은 make bench LIMIT=10 으로 models/base/ 에 받는다
tar czf piiguard-laya-data.tgz datasets/.downloads
tar czf piiguard-laya-models.tgz models/base models/trained

# (오프라인 환경)
docker load < piiguard-laya-cuda.tar.gz
tar xzf piiguard-laya-data.tgz
tar xzf piiguard-laya-models.tgz
HF_HUB_OFFLINE=1 make prepare OFFLINE=1
HF_HUB_OFFLINE=1 make bench
```

- 체크포인트 리비전 고정: Base 모델을 처음 받을 때 `LAYA_REVISION=e333cb7c823cd1787276228260f56d02fbf5180b` (`convaiinnovations/laya`, 2026-11 확인). 받은 커밋은 `models/base/<이름>/MODEL_CARD.json` 에 남는다.
- 데이터 원본: KDPII 는 Zenodo 원본(md5 검증), Gretel 은 Hugging Face `@7b844d1` 에 고정된다. 한국어 번역본은 `WITH_GRETEL_KO=1` 일 때만 받는다.
- 외부 네트워크 요청은 데이터(`make prepare`)와 Base 모델(Hugging Face) 다운로드뿐이다. `HF_HUB_OFFLINE=1` 과 `OFFLINE=1` 을 쓰면 외부 요청이 없다. 오프라인에서 `models/base/` 에 없는 Base 모델을 지정하면 받을 수 없다는 안내와 함께 멈춘다.
