# LeRobot RB-Y1 Joint-to-Cartesian Conversion

LeRobot 2.0 format 데이터셋의 joint state/command를 [Pinocchio](https://github.com/stack-of-tasks/pinocchio) forward kinematics와 [RB-Y1 URDF](https://github.com/RainbowRobotics/rby1-sdk/tree/main/models)를 이용하여 Cartesian space 6-DOF pose로 변환하고, matplotlib으로 시각화하는 도구입니다.

## 기능

- **LeRobot 2.0 데이터셋 로딩** — HuggingFace Hub 또는 로컬 경로에서 chunk 단위 parquet 파일 읽기
- **Forward Kinematics** — Pinocchio + RB-Y1 URDF를 이용한 joint → Cartesian 변환
- **양손 end-effector 추적** — `ee_right`, `ee_left` 프레임의 6-DOF pose (position + SO(3) + RPY) 계산
- **observation.state & action 동시 변환** — 현재 상태와 명령을 모두 Cartesian으로 변환하여 비교
- **결과 저장** — `.npz` 파일로 position, rotation matrix (SO(3)), RPY 저장
- **시각화** — 시간축 기반 x, y, z, roll, pitch, yaw 플롯 (state=파랑, action=빨강)

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

### URDF 준비

RB-Y1 URDF 파일은 [rby1-sdk](https://github.com/RainbowRobotics/rby1-sdk) 레포지토리에서 받을 수 있습니다:

```bash
git clone https://github.com/RainbowRobotics/rby1-sdk.git
# URDF 경로: rby1-sdk/models/rby1a/urdf/model.urdf
```

## 사용법

### HuggingFace Hub 데이터셋 사용

```bash
python joint_to_cartesian.py \
    --dataset tony346/rby1_HF_Test \
    --urdf /path/to/rby1-sdk/models/rby1a/urdf/model.urdf \
    --episodes 0 1 2 \
    --save-dir ./output
```

### 로컬 데이터셋 사용

```bash
python joint_to_cartesian.py \
    --dataset /path/to/local/dataset \
    --urdf /path/to/model.urdf \
    --local \
    --save-dir ./output
```

### 전체 에피소드 처리

```bash
python joint_to_cartesian.py \
    --dataset tony346/rby1_HF_Test \
    --urdf /path/to/model.urdf
```

### CLI 옵션

| 옵션 | 필수 | 설명 |
|------|------|------|
| `--dataset` | O | HuggingFace repo ID 또는 로컬 경로 |
| `--urdf` | O | RB-Y1 URDF 파일 경로 |
| `--episodes` | X | 처리할 에피소드 ID 목록 (기본: 전체) |
| `--save-dir` | X | 출력 디렉토리 (기본: `./output`) |
| `--local` | X | `--dataset`을 로컬 경로로 취급 |

## 출력 구조

```
output/
├── fk_results/
│   ├── episode_000000_ee_right_fk.npz
│   ├── episode_000000_ee_left_fk.npz
│   ├── episode_000001_ee_right_fk.npz
│   └── ...
└── plots/
    ├── episode_000000_ee_right.png
    ├── episode_000000_ee_left.png
    └── ...
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

데이터셋의 joint 구성이 다른 경우 `joint_to_cartesian.py` 상단의 상수를 수정하세요:

- `DATASET_JOINT_NAMES` — FK에 사용할 joint 이름과 순서 (dataset index 순)
- `GRIPPER_MAPPING` — gripper joint 이름 → dataset index 매핑
