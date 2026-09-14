# 로컬 검증 기록

검증일: 2026-09-15. macOS ARM64, Python 3.12.14, PyTorch 2.8.0, CPU.

## 실행 확인

- `pytest -q`: 21개 통과.
- `ruff check .`, `ruff format --check .`: 통과.
- editable 설치 및 `critic-poc` CLI 실행: 통과.
- `python -m build --no-isolation`: sdist와 wheel 생성 성공.
- CUDA / 실제 Alpamayo 모델 추론: 실행하지 않음. GPU 호스트에서 별도 검증 필요.

## 실제 실행한 학습 명령

```bash
critic-poc demo --out runs/verified-cpu-demo --epochs 8 --scenes 240
```

- 합성 240 장면, 장면당 후보 8개, 미래 32점 × 0.1초.
- train 156 / val 36 / calibration 24 / test 24 장면, 원본 group 분리.
- 모델 파라미터 98,502개.
- epoch 1 train loss 0.6301, val loss 0.4864.
- epoch 8 train loss 0.1622, best val loss 0.1756 (epoch 7).
- 최종 test의 weak-label pairwise accuracy 약 0.9563 (183쌍).
- learned collision-CV proxy recall 약 0.9661, offroad proxy recall 약 0.7222.
- 규칙 veto를 함께 적용한 결정의 selection rate 약 0.9167.

**해석:** 위 수치는 단순 합성 장면에서 규칙 라벨을 학습한 결과입니다.
같은 규칙으로 학습·평가하므로 실제 안전 향상, 사람 선호, Alpamayo 성능 개선의 근거가 아닙니다.
규칙 veto가 막은 위반을 Learned Critic의 성과로 계산하면 안 됩니다.
최종 평가의 regret은 선택된 장면 조건부 지표이며 coverage와 함께 읽어야 합니다.

## 회귀 검증

- 주변 객체/지도 누락을 안전 label로 처리하지 않음.
- 미래 outcome 필드를 바꿔도 모델 입력은 동일.
- 후보 순서를 바꾸면 출력 순서만 바뀜.
- 다른 길이·후보 수·객체 수와 batching해도 padding이 결과를 바꾸지 않음.
- 두 waypoint 사이에 발생한 CV 충돌도 선분 sweep으로 감지.
- 모든 후보 탈락 또는 관측 누락 시 selected_index=null.
- 원본 group이 train/val/calibration과 test에 겹치면 평가 거부.
- domain shift를 명시적으로 허용해도 추천은 보류하고 eligible_indices를 비움.
- 선택된 후보에 outcome 라벨이 없으면 위반율은 null, label coverage는 0.

실험 데이터·가중치는 `.gitignore`로 제외되어 저장소에 포함되지 않습니다.
동일 명령으로 생성할 수 있으며, 실제 실험에는 새 출력 경로를 사용하세요.
