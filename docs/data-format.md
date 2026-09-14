# 데이터 계약 v1

한 줄이 한 장면인 JSONL입니다. 동일 장면의 후보는 같은 길이 T를 갖습니다.
장면 간 K/N/T 차이는 collate가 padding하고, 모든 pooling/loss는 mask를 사용합니다.

## 필수 입력

```json
{
  "schema_version": 1,
  "scene_id": "clip-a:5100000",
  "group_id": "clip-a",
  "domain": "physical_ai_av",
  "frame": "ego_t0",
  "dt": 0.1,
  "t0_us": 5100000,
  "ego": {"speed": 5.0, "length": 4.8, "width": 2.0, "desired_speed": 5.0},
  "observation_complete": true,
  "lane_half_width": 4.0,
  "agents": [
    {"x": 25.0, "y": 0.0, "yaw": 0.0, "vx": 3.0, "vy": 0.0, "length": 4.5, "width": 1.9}
  ],
  "candidates": [
    [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
    [[0.48, 0.0, 0.0], [0.94, 0.0, 0.0]]
  ]
}
```

이 예시는 shape 설명용 0.2초 장면입니다. 실제 Alpamayo는 64점입니다.

- 좌표: t0 ego 위치/방향 기준. x 전방, y 좌측, yaw 반시계 방향.
- 단위: m, s, rad. 후보 첫 점은 t0+dt, 마지막 점은 t0+T×dt.
- 현재 ego 위치와 yaw는 (0,0,0). 이 시작 상태를 미분에 포함합니다.
- ego 초기 가속도를 받지 않으므로 첫 jerk는 0으로 두며, 이후 jerk만 계산합니다.
- world 좌표나 clip 시작 좌표를 넣으면 안 됩니다. 위치·속도·heading을 함께 변환하세요.
- 2D 평면 기준입니다. 경사/roll/pitch를 표현하는 일반적인 도로 모델이 아닙니다.
- `desired_speed`는 진행량 정규화용 목표 속도이며 법적 제한 속도가 아닙니다.
- `lane_half_width`는 y=0 중심의 직선 통행 corridor 절반 너비입니다.
  곡선·교차로·여러 차선 topology를 이 숫자로 표현하면 안 됩니다. 모르면 null.
- `observation_complete=false`면 collision label mask=0이며 추천을 보류합니다.
  `agents=[]`와 true는 관측 영역에 객체가 없음을 확인한 경우에만 사용하세요.
- 객체 미래 정답·사후 smoothing 특징은 online 입력에 사용하지 않습니다.
- `group_id`는 원본 log/drive/clip 단위입니다. 인접 프레임에 새 group을 만들지 마세요.
- 최대 K=128, N=128, T=256은 데이터 계약 제한이며 동시 최대 크기의 메모리를 보장하지 않습니다.

## 모델 텐서

| 이름 | shape | 내용 |
|---|---|---|
| agents | [B,N,8] | x,y,sin(yaw),cos(yaw),vx,vy,length,width |
| ego | [B,7] | speed,length,width,desired_speed,lane_width,lane_known,observation_complete |
| trajectory | [B,K,T,8] | x,y,sin(yaw),cos(yaw),speed,acceleration,jerk,time |
| agent_mask | [B,N] | 유효 객체 |
| time_mask | [B,K,T] | 유효 미래점 |
| candidate_mask | [B,K] | 유효 후보 |
| risk_logits | [B,K,3] | collision_cv / offroad_corridor / dynamics |
| quality | [B,K,2] | comfort / progress_proxy, 0~1 |
| utility | [B,K] | 품질 선호 순위, 0~1 |

시간 self-attention은 제안 궤적 전체를 봅니다. 이것은 관측 미래 누출이 아닙니다.
주변 객체 future는 현재 위치·속도를 선형 외삽합니다. reactive multi-agent prediction은 없습니다.

## 학습 라벨

입력에 `labels`를 추가합니다. 모델은 해당 필드를 입력 encoder에 전달하지 않습니다.

```json
{
  "source": "rules_cv",
  "version": "cv-disc-corridor-v1",
  "risk": [[0,0,0],[0,0,0]],
  "risk_mask": [[1,1,1],[1,1,1]],
  "quality": [[0.9,0.8],[0.95,0.7]],
  "quality_mask": [[1,1],[1,1]],
  "utility": [0.84,0.8],
  "utility_mask": [1,1],
  "pairs": [[0,1,1.0]]
}
```

- 모든 target은 finite 0~1. 위험은 높을수록 나쁨, 품질은 높을수록 좋음.
- mask는 정확히 0 또는 1. 알 수 없는 값은 target=0, mask=0.
- 위험 head 3개의 순서는 고정입니다. 신호 위반 라벨을 collision 자리에 넣으면 안 됩니다.
- quality 중 불확실한 항목은 개별 mask를 0으로 둡니다.
- 안전이 확인되지 않은 후보의 utility target은 mask=0으로 두는 것을 권장합니다.
- pair = [A,B,P(A>B)]. 동률은 0.5. 같은 후보끼리 비교 금지.
- 자동 label은 feasible 후보끼리 utility 차이 0.03 이상일 때 pair를 만듭니다.
- weak label이 틀릴 수 있으므로 source와 version을 보존하고 사람 검토를 별도로 수행하세요.

## 외부 시뮬레이터 연결

시뮬레이터가 출력한 결과를 위 `labels` 배열로 변환하면 `train`이 바로 읽습니다.
`source="simulation:engine-version"` 및 추가 metadata에 다음을 기록하세요.

1. candidate 전체 실행인지, prefix 후 어떤 정책으로 재계획했는지.
2. label horizon, 차량 controller, agent 반응 모델, seed.
3. 충돌·도로 이탈·동역학 위반의 정확한 정의.
4. 반복 rollout의 사건 비율인지 1회 이진 결과인지.
5. simulation validity 및 실패한 rollout의 mask.

같은 의미의 head만 같은 checkpoint에서 학습하세요. v1은 여러 label semantics를 자동 정렬하지 않습니다.
`risk_proxy` JSON 필드명은 의도적으로 유지합니다. 실차 충돌 확률로 오해하지 않도록 합니다.

## 분할 / 확률 보정

`split`은 group 단위로 train 65%, val 15%, calibration 10%, test 10%를 배정합니다.
후보끼리 무작위 분할하지 않습니다. 유사한 인접 clip도 같은 drive group으로 묶으세요.

Temperature fitting은 별도 calibration split에만 적용합니다.
head별 표본이 10개 미만이거나 양/음성 중 하나가 없으면 해당 head는 보정하지 않습니다.
보정된 값도 원래 label 분포 안에서의 proxy 확률입니다. OOD 안전 상한이 아닙니다.

평가는 train/val/calibration에 나타난 group을 기본적으로 거부합니다.
`--allow-seen-groups`는 디버깅용이며 결과에 seen_groups=true로 남습니다.
