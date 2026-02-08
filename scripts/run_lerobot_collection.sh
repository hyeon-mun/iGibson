#!/bin/bash
# LeRobot 포맷 데이터 수집 실행 스크립트

# 기본 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../lerobot_dataset"
REPO_ID="igibson_nav"
NUM_EPISODES=100
MAX_STEPS=500
SCENE="Rs"
FPS=30

# 사용법 출력
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Collect data from iGibson in LeRobot format"
    echo ""
    echo "Options:"
    echo "  -s, --scene SCENE_ID      Scene to use (default: Rs)"
    echo "  -n, --num-episodes N      Number of episodes (default: 100)"
    echo "  -m, --max-steps N         Max steps per episode (default: 500)"
    echo "  -o, --output-dir DIR      Output directory (default: ../lerobot_dataset)"
    echo "  -r, --repo-id ID          LeRobot dataset ID (default: igibson_nav)"
    echo "  -f, --fps N               Frames per second (default: 30)"
    echo "  --no-depth                Disable depth image collection"
    echo "  --no-video                Save as images instead of video"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 -s Rs -n 50"
    echo "  $0 --scene Beechwood --num-episodes 200 --repo-id beechwood_nav"
    echo ""
    echo "Output format:"
    echo "  LeRobot v3.0 format with:"
    echo "  - observation.images.rgb: RGB camera images (video)"
    echo "  - observation.images.depth: Depth images (video)"
    echo "  - observation.state: Robot state [pos, quat, vel]"
    echo "  - action: [linear_vel, angular_vel]"
    echo "  - task: Natural language command"
}

# 인자 파싱
EXTRA_FLAGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--scene)
            SCENE="$2"
            shift 2
            ;;
        -n|--num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        -m|--max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        -f|--fps)
            FPS="$2"
            shift 2
            ;;
        --no-depth)
            EXTRA_FLAGS="$EXTRA_FLAGS --no_depth"
            shift
            ;;
        --no-video)
            EXTRA_FLAGS="$EXTRA_FLAGS --no_video"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "LeRobot Data Collection"
echo "========================================"
echo "Scene: $SCENE"
echo "Episodes: $NUM_EPISODES"
echo "Max steps: $MAX_STEPS"
echo "FPS: $FPS"
echo "Repo ID: $REPO_ID"
echo "Output: $OUTPUT_DIR/$REPO_ID"
echo "========================================"

# 실행
python3 "${SCRIPT_DIR}/lerobot_data_collection.py" \
    --scene "$SCENE" \
    --num_episodes "$NUM_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --repo_id "$REPO_ID" \
    --fps "$FPS" \
    --config "${SCRIPT_DIR}/configs/vla_collection_config.yaml" \
    $EXTRA_FLAGS

echo ""
echo "========================================"
echo "Data collection complete!"
echo "Dataset saved at: $OUTPUT_DIR/$REPO_ID"
echo ""
echo "To load the dataset in Python:"
echo "  from lerobot.datasets.lerobot_dataset import LeRobotDataset"
echo "  dataset = LeRobotDataset('$REPO_ID', root='$OUTPUT_DIR')"
echo "========================================"
