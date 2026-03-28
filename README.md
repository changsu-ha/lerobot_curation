# LeRobot Dataset Curation Tools

LeRobot 2.0 format 데이터셋(RB-Y1 로봇)을 대상으로 하는 데이터 큐레이션 도구 모음입니다.
대량의 에피소드를 샘플링하여 읽고, FK를 통해 Cartesian pose로 변환한 뒤,
작업 구간(phase)을 자동으로 분할합니다.

## 기능

### 1. Joint → Cartesian 변환 (`joint_to_cartesian.py`)
- **LeRobot 2.0 데이터셋 로딩** — HuggingFace Hub 또는 로컬 경로에서 chunk 단위 parquet 파일 읽기
- **Forward Kinematics** — [Pinocchio](https://github.com/stack-of-tasks/pinocchio) + [RB-Y1 URDF](https://github.com/RainbowRobotics/rby1-sdk/tree/main/models)를 이용한 joint → Cartesian 변환
- **양손 end-effector 추적** — `ee_right`, `ee_left` 프레임의 6-DOF pose (position + SO(3) + RPY) 계산
- **observation.state & action 동시 변환** — 현재 상태와 명령을 모두 Cartesian으로 변환하여 비교
- **결과 저장** — `.npz` 파일로 position, rotation matrix (SO(3)), RPY 저장
- **시각화** — 시간축 기반 x, y, z, roll, pitch, yaw 플롯 (state=파랑, action=빨강)

### 2. 에피소드 구간 자동 분할 (`segment_episodes.py`)
- **균일 샘플링** — 대량의 데이터셋에서 N개 에피소드를 균일하게 샘플링
- **자동 구간 분할** — gripper 상태 변화 + EE 속도 기반으로 6개 작업 phase 자동 분류
  - `approach` → `grasp` → `move` → `insertion` → `place` → `move_to_ready`
- **YAML 설정** — threshold, dwell time 등 세분화 파라미터를 YAML로 관리
- **세그먼트 시각화** — phase별 색상 배경이 적용된 Cartesian pose + speed + gripper 플롯
- **통계 요약** — 전체 샘플 에피소드의 phase별 평균/표준편차 duration 리포트

## 데이터셋 Joint 구성

`observation.state`와 `action`은 44차원 벡터로 구성됩니다:

| Index | Joint | 개수 |
|-------|-------|------|
| 0–5 | `torso_0` ~ `torso_5` | 6 |
| 6–12 | `right_arm_0` ~ `right_arm_6` | 7 |
| 13–19 | `left_arm_0` ~ `left_arm_6` | 7 |
| 20–31 | right gripper | 12 |
| 32–43 | left gripper | 12 |

FK 계산에는 torso (6) + right arm (7) + left arm (7) = 20개 joint이 사용되며, gripper 값 중 처음 2개는 URDF의 prismatic gripper joint에 매핑됩니다.

## 설치

### Conda (권장)

```bash
conda env create -f environment.yml
conda activate data-curation
```

### pip

```bash
pip install -r requirements.txt
```

### 의존성

- `pin` (Pinocchio) — rigid body dynamics / FK 계산
- `numpy`, `pandas`, `pyarrow` — 데이터 처리
- `matplotlib` — 시각화
- `huggingface_hub` — HuggingFace 데이터셋 다운로드
- `scipy` — 신호 처리 (median filter, Gaussian smoothing)
- `pyyaml` — 설정 파일 로딩

### URDF 준비

RB-Y1 URDF 파일은 [rby1-sdk](https://github.com/RainbowRobotics/rby1-sdk) 레포지토리에서 받을 수 있습니다:

```bash
git clone https://github.com/RainbowRobotics/rby1-sdk.git
# URDF 경로: rby1-sdk/models/rby1a/urdf/model.urdf
```

## 사용법

### 1. Joint → Cartesian 변환

#### HuggingFace Hub 데이터셋 사용

```bash
python joint_to_cartesian.py \
    --dataset tony346/rby1_HF_Test \
    --urdf /path/to/rby1-sdk/models/rby1a/urdf/model.urdf \
    --episodes 0 1 2 \
    --save-dir ./output
```

#### 로컬 데이터셋 사용

```bash
python joint_to_cartesian.py \
    --dataset /path/to/local/dataset \
    --urdf /path/to/model.urdf \
    --local \
    --save-dir ./output
```

#### 전체 에피소드 처리

```bash
python joint_to_cartesian.py \
    --dataset tony346/rby1_HF_Test \
    --urdf /path/to/model.urdf
```

#### CLI 옵션 (`joint_to_cartesian.py`)

| 옵션 | 필수 | 설명 |
|------|------|------|
| `--dataset` | O | HuggingFace repo ID 또는 로컬 경로 |
| `--urdf` | O | RB-Y1 URDF 파일 경로 |
| `--episodes` | X | 처리할 에피소드 ID 목록 (기본: 전체) |
| `--save-dir` | X | 출력 디렉토리 (기본: `./output`) |
| `--local` | X | `--dataset`을 로컬 경로로 취급 |

### 2. 에피소드 구간 분할

```bash
# 5개 에피소드를 균일 샘플링하여 구간 분할
python segment_episodes.py \
    --dataset tony346/rby1_HF_Test \
    --urdf /path/to/model.urdf \
    --n-samples 5 \
    --save-dir ./output

# 커스텀 설정 파일 사용
python segment_episodes.py \
    --dataset /path/to/local/dataset \
    --urdf /path/to/model.urdf \
    --n-samples 10 \
    --config segment_config.yaml \
    --local

# 왼손 end-effector 기준으로 분할
python segment_episodes.py \
    --dataset tony346/rby1_HF_Test \
    --urdf /path/to/model.urdf \
    --n-samples 3 \
    --ee-frame ee_left
```

#### CLI 옵션 (`segment_episodes.py`)

| 옵션 | 필수 | 설명 |
|------|------|------|
| `--dataset` | O | HuggingFace repo ID 또는 로컬 경로 |
| `--urdf` | O | RB-Y1 URDF 파일 경로 |
| `--n-samples` | O | 균일 샘플링할 에피소드 수 |
| `--config` | X | 세그멘테이션 설정 YAML 경로 (기본: 내장 기본값) |
| `--save-dir` | X | 출력 디렉토리 (기본: `./output`) |
| `--local` | X | `--dataset`을 로컬 경로로 취급 |
| `--ee-frame` | X | 분할 기준 end-effector (기본: `ee_right`) |

#### 세그멘테이션 설정 (`segment_config.yaml`)

YAML 파일로 분할 파라미터를 조정할 수 있습니다:

```yaml
gripper:
  thresh_closed: 0.01    # 그리퍼 닫힘 판정 (m)
  thresh_open: 0.04      # 그리퍼 열림 판정 (m)
  dwell_frames: 6        # 상태 변화 확인 프레임 수

velocity:
  thresh_moving: 0.005   # 이동 중 판정 (m/s)
  thresh_stationary: 0.002  # 정지 판정 (m/s)
  dwell_frames: 10

preprocessing:
  gripper_median_kernel: 5  # 그리퍼 신호 median filter
  pos_smooth_sigma: 3.0     # 위치 Gaussian smoothing
  vel_smooth_sigma: 5.0     # 속도 Gaussian smoothing
```

#### 분할 알고리즘

1. **Gripper anchor 검출** — 히스테리시스 state machine으로 gripper close(grasp) / open(place) 이벤트 검출
2. **속도 기반 세분화** — 각 anchor 구간 내에서 EE 속도로 이동/정지 판별
3. **Phase 할당**:
   - `[0, T_grasp)` → approach (이동 중) / grasp (정지)
   - `[T_grasp, T_place)` → move (이동 중) / insertion (정지 + 목표 위치)
   - `[T_place, N)` → place (정지) / move_to_ready (이동 중)
4. **Fallback** — gripper anchor 미검출 시 속도 기반 heuristic으로 대체

## 출력 구조

```
output/
├── fk_results/
│   ├── episode_000000_ee_right_fk.npz
│   ├── episode_000000_ee_left_fk.npz
│   └── ...
├── plots/                                          # joint_to_cartesian.py
│   ├── episode_000000_ee_right.png
│   └── ...
├── segment_plots/                                  # segment_episodes.py
│   ├── episode_000000_ee_right_segmented.png
│   ├── episode_000000_ee_left_segmented.png
│   └── phase_duration_summary.png
└── segmentation_results.json                       # 전체 분할 결과
```

### .npz 파일 내용

각 `.npz` 파일에는 다음 배열이 포함됩니다:

| Key | Shape | 설명 |
|-----|-------|------|
| `timestamps` | (N,) | 타임스탬프 (초) |
| `state_pos` | (N, 3) | observation.state FK — position [x, y, z] |
| `state_rotation_matrix` | (N, 3, 3) | observation.state FK — SO(3) rotation matrix |
| `state_rpy` | (N, 3) | observation.state FK — [roll, pitch, yaw] |
| `action_pos` | (N, 3) | action FK — position [x, y, z] |
| `action_rotation_matrix` | (N, 3, 3) | action FK — SO(3) rotation matrix |
| `action_rpy` | (N, 3) | action FK — [roll, pitch, yaw] |

### .npz 파일 로딩 예시

```python
import numpy as np

data = np.load("output/fk_results/episode_000000_ee_right_fk.npz")
timestamps = data["timestamps"]           # (N,)
state_pos = data["state_pos"]             # (N, 3)
state_rot = data["state_rotation_matrix"] # (N, 3, 3)
state_rpy = data["state_rpy"]             # (N, 3)
```

## 커스터마이징

### Joint 매핑
데이터셋의 joint 구성이 다른 경우 `joint_to_cartesian.py` 상단의 상수를 수정하세요:
- `DATASET_JOINT_NAMES` — FK에 사용할 joint 이름과 순서 (dataset index 순)
- `GRIPPER_MAPPING` — gripper joint 이름 → dataset index 매핑

### 세그멘테이션 파라미터
`segment_config.yaml`을 수정하여 분할 threshold를 조정하세요.
`--visualize` 플롯의 speed/gripper 서브플롯을 참고하여 적절한 값을 찾을 수 있습니다.
