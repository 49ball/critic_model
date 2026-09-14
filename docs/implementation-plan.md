# Alpamayo Critic PoC 구현 계획

목표: Alpamayo 1.5의 복수 궤적을 공통 JSONL로 내보내고, PyTorch Critic의
학습·평가·순위화를 CPU/GPU에서 재현한다. 사용자 요청으로 구현 및 origin/main push 승인됨.

## 설계 / 첫 버전 범위
- Python 3.12, PyTorch 2.8, NumPy. NVIDIA 모델 의존성은 별도 환경.
- 공통 좌표계: t0 ego, x 전방, y 좌측, yaw 반시계, m/s/rad, 첫 미래점 dt.
- 현재 주변 객체 + ego + 후보 궤적. 미래 ground truth는 모델 입력에서 제외.
- 소형 Transformer + 상대 위치 기반 interaction + risk/quality/utility heads.
- 규칙은 CV 객체 예측·보수적 원형 envelope·직선 corridor·kinematics proxy.
  실제 안전 검증층이나 reactive closed-loop simulator로 부르지 않는다.
- unknown은 safe가 아니다. 누락 라벨 mask, 모든 후보 탈락 시 selected=null.
- train/val/test는 group_id(원본 log/clip) 분리. domain shift는 명시적 허용.
- 합성 샘플은 plumbing 확인용. Alpamayo 가중치·사용자 데이터는 커밋하지 않는다.

## 순서와 검증
1. schema.py, geometry.py, rules.py / tests/test_contract.py
   - [x] 좌표·시간·shape·NaN 검증, rotation→yaw, 시작점 미분.
   - [x] 위험 후보 reject / all-rejected / 누락 관측 abstain 테스트.
2. model.py, data.py, losses.py / tests/test_learning.py
   - [x] variable K/N/T padding·mask, empty agents, candidate permutation.
   - [x] masked labels와 preference 방향, optimizer loss 감소 테스트.
3. synthetic.py, training.py, inference.py, cli.py / tests/test_pipeline.py
   - [x] 데이터 생성→2 epoch 학습→checkpoint reload→평가→순위화.
   - [x] log 단위 split leakage·domain shift 방지.
4. alpamayo.py / adapter tests
   - [x] 공식 1.5 API의 [B,group,K,T,3], rotation 텐서 정규화.
   - [x] GPU export 경로 및 world sidecar 계약. NVIDIA GPU 실행은 이 호스트 미검증.
5. README.md, docs/data-format.md, configs/default.json, CI
   - [x] 한국어 quickstart / 실데이터 학습 / 향후 closed-loop 확장 안내.
   - [x] pytest, ruff, CPU demo, wheel 빌드. docs/validation.md에 결과 기록.
