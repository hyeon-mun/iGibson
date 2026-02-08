#!/bin/bash
# VLA 데이터 수집 실행 스크립트

# 기본 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../vla_dataset"
NUM_EPISODES=100
MAX_STEPS=500
SCENE="Rs"

# 사용법 출력
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --scene SCENE_ID      Scene to use (default: Rs)"
    echo "  -n, --num-episodes N      Number of episodes (default: 100)"
    echo "  -m, --max-steps N         Max steps per episode (default: 500)"
    echo "  -o, --output-dir DIR      Output directory (default: ../vla_dataset)"
    echo "  --no-velodyne             Disable Velodyne LiDAR"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 -s Rs -n 50"
    echo "  $0 --scene Beechwood --num-episodes 200 --output-dir ./data"
}

# 인자 파싱
VELODYNE_FLAG=""
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
        --no-velodyne)
            VELODYNE_FLAG="--no_velodyne"
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
echo "VLA Data Collection"
echo "========================================"
echo "Scene: $SCENE"
echo "Episodes: $NUM_EPISODES"
echo "Max steps: $MAX_STEPS"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# 실행
python3 "${SCRIPT_DIR}/vla_data_collection.py" \
    --scene "$SCENE" \
    --num_episodes "$NUM_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --config "${SCRIPT_DIR}/configs/vla_collection_config.yaml" \
    $VELODYNE_FLAG

echo "Done!"
