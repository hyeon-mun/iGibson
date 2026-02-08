# iGibson VLA Data Collection

iGibson 시뮬레이터를 활용한 VLA(Vision-Language-Action) 데이터 수집 파이프라인입니다.
[LeRobot](https://github.com/huggingface/lerobot) 포맷을 지원하여 로봇 학습 데이터셋을 생성할 수 있습니다.

> 이 프로젝트는 [StanfordVL/iGibson](https://github.com/StanfordVL/iGibson)을 기반으로 합니다.

## 📋 목차

- [설치](#설치)
- [데이터 다운로드](#데이터-다운로드)
- [데이터 수집](#데이터-수집)
- [데이터 로드](#데이터-로드)
- [프로젝트 구조](#프로젝트-구조)

---

## 설치

### 방법 1: Docker (권장)

```bash
# 레포 클론
git clone https://github.com/hyeon-mun/iGibson.git
cd iGibson

# Docker 이미지 빌드
docker build -t igibson-vla -f .devcontainer/Dockerfile .

# 컨테이너 실행 (GPU 필요)
docker run -it --gpus all \
    -v $(pwd):/workspace/iGibson \
    -v /path/to/data:/workspace/iGibson/data \
    igibson-vla
```

### 방법 2: 수동 설치

```bash
# 1. iGibson 설치
git clone https://github.com/hyeon-mun/iGibson.git
cd iGibson
pip install -e .

# 2. 추가 의존성 설치
pip install h5py pandas pyarrow pillow

# 3. ffmpeg 설치 (비디오 인코딩용)
sudo apt-get install ffmpeg

# 4. LeRobot 클론 (LeRobot 포맷 사용 시)
git clone https://github.com/huggingface/lerobot.git third_party/lerobot
```

---

## 데이터 다운로드

### iGibson Assets 다운로드

```bash
# 기본 에셋 다운로드
python -m igibson.utils.assets_utils --download_assets

# 데모 데이터 다운로드
python -m igibson.utils.assets_utils --download_demo_data
```

### Gibson 데이터셋 다운로드

Gibson 데이터셋은 라이센스 동의 후 다운로드할 수 있습니다:
1. [Stanford Gibson 데이터셋](http://gibsonenv.stanford.edu/database/) 페이지 방문
2. 라이센스 동의 후 다운로드 링크 획득
3. 다운로드 후 `data/` 폴더에 배치

```bash
# 데이터 폴더 구조
data/
├── assets/
├── g_dataset/          # Gibson 데이터셋
│   ├── Rs/
│   ├── Beechwood/
│   └── ...
└── ig_dataset/         # iGibson 데이터셋
```

---

## 데이터 수집

### LeRobot 포맷 (권장)

LeRobot v3.0 포맷으로 데이터를 수집합니다. HuggingFace 생태계와 호환됩니다.

```bash
# 기본 수집 (비디오 모드)
python scripts/lerobot_data_collection.py \
    --scene Rs \
    --num_episodes 100 \
    --output_dir ./lerobot_dataset \
    --repo_id igibson_nav

# 이미지 모드 (비디오 인코딩 없이)
python scripts/lerobot_data_collection.py \
    --scene Rs \
    --num_episodes 100 \
    --output_dir ./lerobot_dataset \
    --repo_id igibson_nav \
    --no_video

# Depth 없이 수집
python scripts/lerobot_data_collection.py \
    --scene Rs \
    --num_episodes 100 \
    --no_depth
```

**생성되는 데이터 구조:**
```
lerobot_dataset/igibson_nav/
├── data/
│   └── chunk-000/
│       └── file-000.parquet      # 프레임 데이터
├── meta/
│   ├── info.json                 # 데이터셋 메타데이터
│   ├── stats.json                # 정규화 통계
│   ├── tasks.parquet             # 태스크 목록
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet  # 에피소드 메타데이터
└── videos/                       # 비디오 파일 (--no_video 미사용 시)
    └── observation_images_rgb/
        └── chunk-000/
            └── episode-000000.mp4
```

**수집되는 Features:**

| Feature | Shape | 설명 |
|---------|-------|------|
| `observation.images.rgb` | (480, 640, 3) | RGB 카메라 이미지 |
| `observation.images.depth` | (480, 640, 3) | Depth 이미지 (RGB 변환) |
| `observation.state` | (13,) | 로봇 상태 벡터 |
| `action` | (2,) | [linear_vel, angular_vel] |
| `observation.goal_distance` | (1,) | 목표까지 거리 |
| `task` | string | 자연어 명령 |

**observation.state 구성:**
- `pos_x, pos_y, pos_z`: 로봇 위치
- `quat_x, quat_y, quat_z, quat_w`: 로봇 방향 (quaternion)
- `lin_vel_x, lin_vel_y, lin_vel_z`: 선속도
- `ang_vel_x, ang_vel_y, ang_vel_z`: 각속도

### HDF5 포맷

기존 방식의 HDF5 포맷으로 데이터를 수집합니다.

```bash
# Shell 스크립트 사용
./scripts/run_vla_collection.sh -s Rs -n 100

# Python 직접 실행
python scripts/vla_data_collection.py \
    --scene Rs \
    --num_episodes 100 \
    --output_dir ./vla_dataset
```

### HDF5 → LeRobot 변환

기존 HDF5 데이터를 LeRobot 포맷으로 변환합니다.

```bash
python scripts/convert_hdf5_to_lerobot.py \
    --input_dir ./vla_dataset \
    --output_dir ./lerobot_dataset \
    --repo_id igibson_nav_converted
```

---

## 데이터 로드

### LeRobot 포맷 로드

```python
import pandas as pd
import json

# 메타데이터 로드
with open("lerobot_dataset/igibson_nav/meta/info.json") as f:
    info = json.load(f)
print(f"Episodes: {info['total_episodes']}, Frames: {info['total_frames']}")

# 프레임 데이터 로드
df = pd.read_parquet("lerobot_dataset/igibson_nav/data/chunk-000/file-000.parquet")
print(df.head())

# 통계 로드 (정규화용)
with open("lerobot_dataset/igibson_nav/meta/stats.json") as f:
    stats = json.load(f)
```

### HDF5 포맷 로드

```python
from scripts.vla_data_loader import VLADataset, VLATorchDataset

# 기본 로드
dataset = VLADataset("./vla_dataset")
print(dataset.get_statistics())

# PyTorch DataLoader와 함께 사용
torch_dataset = VLATorchDataset("./vla_dataset")
loader = torch.utils.data.DataLoader(torch_dataset, batch_size=32)

for batch in loader:
    rgb = batch["rgb"]        # (B, C, H, W)
    action = batch["action"]  # (B, 2)
    # ...
```

---

## 프로젝트 구조

```
iGibson/
├── scripts/
│   ├── lerobot_data_collection.py   # LeRobot 포맷 수집
│   ├── convert_hdf5_to_lerobot.py   # HDF5 → LeRobot 변환
│   ├── vla_data_collection.py       # HDF5 포맷 수집
│   ├── vla_data_loader.py           # HDF5 데이터 로더
│   ├── run_lerobot_collection.sh    # LeRobot 수집 스크립트
│   ├── run_vla_collection.sh        # HDF5 수집 스크립트
│   └── configs/
│       └── vla_collection_config.yaml
├── .devcontainer/
│   ├── Dockerfile
│   └── devcontainer.json
├── data/                            # 데이터 폴더 (gitignore)
├── third_party/
│   └── lerobot/                     # LeRobot (별도 클론 필요)
└── igibson/                         # iGibson 코어
```

---

## 설정 파일

`scripts/configs/vla_collection_config.yaml`에서 수집 설정을 변경할 수 있습니다:

```yaml
# Scene 설정
scene: gibson
scene_id: Rs

# 이미지 설정
image_width: 640
image_height: 480

# LiDAR 설정 (Velodyne VLP-16 스타일)
n_horizontal_rays: 360
n_vertical_beams: 16
laser_linear_range: 100.0

# Task 설정
task: point_nav_random
target_dist_min: 3.0
target_dist_max: 10.0
```

---

## 지원 Scene

| Scene ID | 설명 |
|----------|------|
| Rs | 작은 아파트 |
| Beechwood | 큰 주택 |
| Ihlen | 중간 크기 주택 |
| Merom | 사무실 |
| ... | [전체 목록](http://gibsonenv.stanford.edu/database/) |

---

## 문제 해결

### EGL 에러
```bash
# headless 렌더링을 위한 환경변수 설정
unset DISPLAY
```

### ffmpeg 미설치
```bash
sudo apt-get install ffmpeg
```

### GPU 메모리 부족
```bash
# 이미지 해상도 줄이기
python scripts/lerobot_data_collection.py \
    --image_height 240 --image_width 320 ...
```

---

## 라이센스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.
iGibson은 [Stanford의 라이센스](https://github.com/StanfordVL/iGibson)를 따릅니다.

---

## 참고 자료

- [iGibson 공식 문서](http://svl.stanford.edu/igibson/)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [Gibson 데이터셋](http://gibsonenv.stanford.edu/database/)
