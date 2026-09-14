# Alpamayo Critic PoC

Alpamayo가 만든 여러 주행 궤적을 평가하는 **학습 가능한 Critic 초안**입니다.
CPU에서 데이터 생성 → 학습 → 체크포인트 저장 → 평가 → 후보 재순위화를 실행할 수 있습니다.
실제 Alpamayo 1.5 추론은 별도의 NVIDIA GPU 환경에서 실행합니다.

> 합성 데이터 학습 성공은 파이프라인 검증입니다. 실차 안전 성능이나 Waymo 수준의 Critic을 입증하지 않습니다.
> v1의 충돌·도로 이탈 라벨은 단순 규칙 proxy이며, reactive closed-loop simulation은 아직 연결하지 않았습니다.

## 1. 가장 빠른 실행

Python 3.12 환경에서:

```bash
git clone https://github.com/49ball/critic_model.git
cd critic_model
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
critic-poc demo --out runs/my-first-poc --epochs 5 --scenes 200
```

NVIDIA GPU가 없어도 실행됩니다. Linux에서 CPU wheel만 설치하려면 editable 설치 전에:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

동일한 출력 폴더는 덮어쓰지 않습니다. 다시 실행할 때 새 `--out` 경로를 사용하세요.

생성 결과:

| 파일 | 내용 |
|---|---|
| `data/train.jsonl` | 합성 학습 장면 |
| `data/val.jsonl` | 체크포인트 선택용 검증 장면 |
| `data/calibration.jsonl` | 위험 출력의 temperature 보정 전용 |
| `data/test.jsonl` | 학습·보정에 사용하지 않는 최종 평가 장면 |
| `model/best.pt` | 검증 loss가 가장 낮은 모델 + calibration 정보 |
| `model/last.pt` | 마지막 모델과 optimizer 상태; CLI resume는 미지원 |
| `model/history.json` | epoch별 train/val loss |
| `model/training-summary.json` | 파라미터 수, 데이터 domain, 학습 결과 |
| `evaluation.json` | AUPRC, Brier, 순위 정확도, 보류율 등 |
| `ranking.json` | 후보별 위험·품질·불확실성과 추천 인덱스 |

가중치, 실행 데이터, 로그는 Git에서 제외됩니다.

## 2. 구성

```text
Alpamayo 1.5 (동결)                 회사 모델 / 다른 모델
       │ K개 xyz + rotation                │
       └──────────── 공통 JSONL ───────────┘
                           │
현재 ego + 주변 객체 ───────┼── 직선 도로 corridor (선택)
                           ▼
                  Trajectory Critic
           ┌───────────────┴─────────────────┐
           │ Transformer + cross attention   │
           │ 후보 ↔ 객체 CV 미래의 상대 위치 │
           └───────────────┬─────────────────┘
                           ▼
          risk 3개 + quality 2개 + utility
                           │
             규칙 veto + uncertainty 보류
                           ▼
                 추천 인덱스 또는 null
```

**v1의 실제 구현 범위**

- 위험: CV 예측 충돌 proxy, 직선 corridor 이탈, 운동학 한계 위반.
- 품질: comfort, ego 전방 진행량 proxy. 진행량은 route completion이 아닙니다.
- utility: 안전 조건을 만족하는 후보의 품질 순위 학습.
- 불확실성: MC dropout의 출력 분산. 안전 확률 상한이나 통계적 보증이 아닙니다.
- 현재 주변 객체만 입력하며, agent history·BEV·raw sensor·신호·교차로 지도는 v1에서 미구현입니다.
- 충돌 규칙은 외접 원과 선분 sweep을 사용합니다. 인접 차선을 과도하게 위험으로 판단할 수 있습니다.
- 검증되지 않은 정보는 안전으로 채우지 않습니다. 관측 미완료·지도 누락·학습하지 않은 head가 있으면 최종 추천을 보류합니다.
- 자동 제동이나 안전 fallback trajectory를 생성하지 않습니다. `selected_index=null`은 외부 planner가 처리해야 합니다.

## 3. 학습 명령을 나누어 실행

```bash
critic-poc synthetic --out data/toy --scenes 400 --candidates 8 --steps 64
critic-poc train \
  --train data/toy/train.jsonl \
  --val data/toy/val.jsonl \
  --calibration data/toy/calibration.jsonl \
  --config configs/default.json \
  --out runs/toy-model --device cpu

critic-poc evaluate \
  --checkpoint runs/toy-model/best.pt \
  --data data/toy/test.jsonl --out runs/toy-evaluation.json

critic-poc rank \
  --checkpoint runs/toy-model/best.pt \
  --data data/toy/test.jsonl --scene-index 0 \
  --mc-samples 8 --out runs/toy-ranking.json
```

Critic 자체는 `--device cuda` 또는 `--device cuda:0`로 학습할 수 있습니다.
Alpamayo 가중치는 Critic 학습에 포함하지 않으므로, 후보를 미리 저장하면 GPU 메모리 요구가 크게 줄어듭니다.

Loss:
```text
L = masked BCE(risk)
  + masked SmoothL1(quality)
  + masked SmoothL1(utility)
  + 0.2 × pairwise BCE((utility_A - utility_B) / 0.2)
```

label mask가 0인 값은 loss에서 제외합니다. 자동 preference는 모든 proxy 검사를 통과한 후보끼리만 생성합니다.
사람이 만든 preference는 `[A 인덱스, B 인덱스, A 선호도]`이며 1=A 선호, 0=B 선호, 0.5=동률입니다.
사람 preference에는 위험 후보를 편안함으로 보상하는 쌍을 넣지 마세요.

## 4. 실제 Alpamayo로 넘어가기

자세한 절차는 **[Alpamayo 실행 안내](docs/alpamayo.md)**에 있습니다.

```bash
# 공식 Alpamayo 1.5가 설치된 Python 3.12 + CUDA 환경
critic-poc alpamayo-export \
  --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26 \
  --t0-us 5100000 --samples 8 \
  --world-json examples/world-template.json \
  --out data/alpamayo/clip-001.jsonl
```

이 명령은 실제 모델을 호출합니다. 템플릿의 객체·지도는 **비어 있으므로** 그대로 쓰면 최종 추천을 보류합니다.
실제 장면과 일치하는 관측 데이터를 채워야 합니다. 객체·지도 추정기를 이 저장소에서 자동 설치하거나 실행하지 않습니다.

여러 독립 clip을 공통 JSONL로 모은 다음:

```bash
cat data/alpamayo/clip-*.jsonl > data/alpamayo/all.jsonl
critic-poc label --data data/alpamayo/all.jsonl --out data/alpamayo/labeled.jsonl
critic-poc split --data data/alpamayo/labeled.jsonl --out data/alpamayo/splits

critic-poc train \
  --train data/alpamayo/splits/train.jsonl \
  --val data/alpamayo/splits/val.jsonl \
  --calibration data/alpamayo/splits/calibration.jsonl \
  --out runs/alpamayo-critic --device cuda
```

최소 10개 서로 다른 `group_id`가 있어야 4개 split을 만듭니다. 이는 기능상 최소치이며 연구에 충분한 데이터 규모가 아닙니다.
실제 학습에서는 다양한 상황·후보·위험 사례를 확보해야 합니다.

합성 checkpoint를 `physical_ai_av` domain에 적용하면 기본적으로 오류를 냅니다.
`--allow-domain-shift`는 분석용 점수 출력만 허용하며, domain이 다르면 추천 인덱스는 여전히 null입니다.

## 5. 실데이터 및 시뮬레이터 라벨

[데이터 계약](docs/data-format.md)에 입력 텐서, 좌표, 단위, label mask, 외부 outcome 형식을 정리했습니다.

- Expert trajectory가 무조건 positive인 구조가 아닙니다. 후보마다 개별 outcome을 붙입니다.
- `label`은 저비용 weak label입니다. 같은 규칙으로 평가하면 주로 규칙 모방 성능을 측정하게 됩니다.
- closed-loop simulator의 outcome을 동일한 `labels` 필드에 넣으면 모델 변경 없이 학습할 수 있습니다.
- 로그의 다른 차량 미래를 고정하고 ego 후보만 바꾸는 평가는 비반응 replay입니다. 반사실적 정답으로 취급하지 않습니다.
- “6.4초 궤적 전체를 추종”과 “0.2초 실행 후 재계획”은 다른 라벨입니다. rollout 정책과 horizon을 기록하고, 서로 섞지 마세요.
- 정책 RL 업데이트는 제공하지 않습니다. reranking과 데이터 품질을 검증한 뒤 별도 단계로 확장합니다.

## 6. 주요 파일

| 파일 | 역할 |
|---|---|
| `schema.py` | JSONL·단위·shape·mask 검증 |
| `alpamayo.py` | 실제 Alpamayo 1.5 호출 및 출력 정규화 |
| `geometry.py / rules.py` | 시작 상태 기반 미분, CV 충돌·도로·동역학 proxy |
| `data.py` | variable K/N/T batching, 미래 정답 입력 차단 |
| `model.py / losses.py` | Transformer 및 multi-task / preference 학습 |
| `training.py` | 학습·best/last 저장·독립 calibration |
| `inference.py / metrics.py` | 평가·재순위화·판단 보류 |
| `synthetic.py / cli.py` | 예제 생성 및 명령줄 도구 |

## 7. 검증

```bash
pytest -q
ruff check .
ruff format --check .
python -m build
```

테스트는 데이터 누출, 누락 라벨, padding, 후보 순서 불변성, 구간 중간 충돌,
전부 탈락하는 경우, 실제 gradient 학습, checkpoint 재로딩, CLI 실행을 확인합니다.
로컬 확인 범위와 결과는 [검증 기록](docs/validation.md)에 기록합니다.

## 참고

- [NVIDIA Alpamayo 1.5 공식 구현](https://github.com/NVlabs/alpamayo1.5)
- [공식 추론 예제](https://github.com/NVlabs/alpamayo1.5/blob/main/src/alpamayo1_5/test_inference.py)
- [Physical AI AV 데이터](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [AlpaSim](https://github.com/NVlabs/alpasim)
- [Waymo의 Driver·Simulator·Critic 설명](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/)

본 코드는 제안된 PoC이며 Waymo 내부 구현을 복제한 것이 아닙니다.
NVIDIA 모델·데이터·코드는 각각의 라이선스와 접근 조건을 따릅니다.
