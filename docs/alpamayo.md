# 실제 Alpamayo 1.5 연동

## 실행 환경

- Linux + NVIDIA CUDA GPU, Python 3.12.
- Hugging Face에서 모델과 Physical AI AV 데이터 접근 동의/권한이 필요합니다.
- 공식 README의 측정치는 단일 샘플 약 24GB, 16개 샘플 약 40GB VRAM입니다.
  이 PoC의 batch·samples 및 다른 GPU 조건에서 같은 메모리를 보장하지 않습니다.
- CPU 데모 환경과 Alpamayo 환경을 분리하면 모델 의존성 충돌을 피하기 쉽습니다.

## 설치

공식 저장소의 설치 절차가 기준입니다.

```bash
git clone https://github.com/NVlabs/alpamayo1.5.git
cd alpamayo1.5
uv sync
source .venv/bin/activate
hf auth login

# critic_model이 형제 디렉터리에 clone되어 있다고 가정
python -m pip install --no-deps -e ../critic_model
# 위 환경에 pip가 없는 경우:
# uv pip install --python .venv/bin/python --no-deps -e ../critic_model
critic-poc --help
```

공식 Python package명과 import 경로는 `alpamayo1_5`입니다.
FlashAttention 설치 등 GPU 환경 문제는 공식 README의 troubleshooting을 따르세요.

## 후보 생성

```bash
critic-poc alpamayo-export \
  --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26 \
  --t0-us 5100000 \
  --samples 8 \
  --out ../critic_model/data/alpamayo/clip-001.jsonl
```

- official loader로 카메라와 ego history를 준비하고, 실제 모델의
  `sample_trajectories_from_data_with_vlm_rollout`을 호출합니다.
- 공개 출력 `xyz [1,1,K,T,3]`, `rotation [1,1,K,T,3,3]`에서 평면 x/y/yaw를 만듭니다.
- 첫 점 t0+0.1s, 기본 64점=6.4초. 다른 버전으로 바뀌면 adapter 계약을 먼저 검증하세요.
- 모델 ID, 요청 revision, 확인 가능한 resolved revision, random seed를 provenance에 저장합니다.
- `--revision`에 모델 commit을 지정하고, 공식 저장소도 commit으로 고정하는 것을 권장합니다.
- 공식 loader가 반환하는 ego future 정답과 reasoning text는 Critic 입력에 넣지 않습니다.
- 차 크기 기본값은 예시입니다. 실제 차량에 맞게 `--ego-length`, `--ego-width`를 지정하세요.

## 주변 객체와 도로는 어디서 얻나?

Alpamayo trajectory 출력에는 Critic이 요구하는 객체 목록·map이 자동 포함되지 않습니다.
이 저장소는 **perception stack을 구현하지 않습니다**.

`examples/world-template.json`을 복사한 뒤, 별도 perception 또는 simulator에서 추출한
같은 시점의 관측을 넣고 `--world-json`으로 전달합니다.

```json
{
  "scene_id": "030c760c-ae38-49aa-9ad8-f5650a545d26:5100000",
  "t0_us": 5100000,
  "frame": "ego_t0",
  "observation_complete": false,
  "lane_half_width": null,
  "agents": [],
  "ego": {"length": 4.8, "width": 2.0, "desired_speed": 8.0}
}
```

위 예시는 의도적으로 unknown입니다. 실제 확인 없이 true나 임의의 lane 폭을 넣지 마세요.
`scene_id`, `t0_us`, `frame`이 다르면 adapter가 오류를 냅니다.

Physical AI AV 데이터의 공개 card에는 map이 포함되지 않는다고 명시되어 있습니다.
offline machine label을 가져오더라도 미래 smoothing 정보가 현재 입력에 섞이지 않는지 확인해야 합니다.
지도 없는 장면에서는 ego comfort/progress 라벨만으로 일부 학습을 진행할 수 있지만,
v1의 엄격한 추천 모드는 corridor 검사가 없으면 추천을 보류합니다.

실제 주변 객체 예:
```json
{"x":25.0,"y":1.0,"yaw":0.0,"vx":3.0,"vy":0.0,"length":4.5,"width":1.9}
```

좌표가 global/clip-origin이면 ego t0의 역변환을 적용해야 합니다.
vehicle center, yaw convention, 속도 좌표도 일치해야 합니다.

## 이것이 테스트된 범위

현재 Mac CPU에서 출력 shape/rotation 변환과 데이터 계약을 자동 테스트했습니다.
**Alpamayo 가중치 다운로드, CUDA 실행, Physical AI AV 실제 clip 로딩은 이 호스트에서 실행하지 않았습니다.**
GPU 호스트에서는 우선 위 명령으로 1개 clip을 내보내고, 좌표를 시각적으로 점검한 뒤 데이터 규모를 늘리세요.

공식 출처:
- [추론 코드](https://github.com/NVlabs/alpamayo1.5/blob/main/src/alpamayo1_5/test_inference.py)
- [입출력 loader](https://github.com/NVlabs/alpamayo1.5/blob/main/src/alpamayo1_5/load_physical_aiavdataset.py)
- [설치 및 메모리](https://github.com/NVlabs/alpamayo1.5)
- [공개 데이터 범위](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
