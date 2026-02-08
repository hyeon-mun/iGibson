#!/usr/bin/env python3
"""
기존 HDF5 VLA 데이터를 LeRobot 포맷으로 변환

사용법:
    python convert_hdf5_to_lerobot.py --input_dir ./vla_dataset --output_dir ./lerobot_dataset
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import PIL.Image

# lerobot 모듈 경로 추가
LEROBOT_PATH = Path(__file__).parent.parent / "third_party" / "lerobot" / "src"
sys.path.insert(0, str(LEROBOT_PATH))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def convert_hdf5_to_lerobot(
    input_dir: str,
    output_dir: str,
    repo_id: str = "igibson_nav_converted",
    fps: int = 30,
    use_videos: bool = True,
    filter_success: bool = False,
):
    """
    HDF5 VLA 데이터셋을 LeRobot 포맷으로 변환

    Args:
        input_dir: HDF5 파일이 있는 디렉토리
        output_dir: LeRobot 데이터셋 저장 디렉토리
        repo_id: LeRobot 데이터셋 ID
        fps: 프레임 레이트
        use_videos: 비디오로 저장할지 (False면 PNG)
        filter_success: True면 성공한 에피소드만 변환
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # HDF5 파일 찾기
    hdf5_files = sorted(glob.glob(str(input_path / "episode_*.hdf5")))
    if not hdf5_files:
        log.error(f"No HDF5 files found in {input_dir}")
        return

    log.info(f"Found {len(hdf5_files)} HDF5 files")

    # 첫 번째 파일에서 데이터 형태 확인
    with h5py.File(hdf5_files[0], "r") as f:
        has_rgb = "rgb" in f
        has_depth = "depth" in f
        num_frames = f.attrs["num_frames"]

        if has_rgb:
            rgb_shape = f["rgb"].shape[1:]  # (H, W, C)
            log.info(f"RGB shape: {rgb_shape}")
        if has_depth:
            depth_shape = f["depth"].shape[1:]
            log.info(f"Depth shape: {depth_shape}")

    # Features 정의
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (13,),
            "names": [
                "pos_x", "pos_y", "pos_z",
                "quat_x", "quat_y", "quat_z", "quat_w",
                "lin_vel_x", "lin_vel_y", "lin_vel_z",
                "ang_vel_x", "ang_vel_y", "ang_vel_z",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["linear_velocity", "angular_velocity"],
        },
        "observation.goal_distance": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["distance"],
        },
    }

    if has_rgb:
        features["observation.images.rgb"] = {
            "dtype": "video" if use_videos else "image",
            "shape": tuple(rgb_shape),
            "names": ["height", "width", "channels"],
        }

    if has_depth:
        features["observation.images.depth"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (rgb_shape[0], rgb_shape[1], 3),  # RGB로 변환
            "names": ["height", "width", "channels"],
        }

    # 기존 디렉토리 삭제
    dataset_path = output_path / repo_id
    if dataset_path.exists():
        import shutil
        log.warning(f"Removing existing dataset at {dataset_path}")
        shutil.rmtree(dataset_path)

    # LeRobot 데이터셋 생성
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        fps=fps,
        robot_type="turtlebot",
        features=features,
        use_videos=use_videos,
        image_writer_threads=4 if use_videos else 0,
        vcodec="h264",
    )

    # 변환 통계
    converted_episodes = 0
    total_frames = 0
    skipped_episodes = 0

    # 각 HDF5 파일 변환
    for ep_idx, hdf5_file in enumerate(hdf5_files):
        try:
            with h5py.File(hdf5_file, "r") as f:
                # 성공 필터링
                success = f.attrs.get("success", True)
                if filter_success and not success:
                    log.info(f"Skipping failed episode: {hdf5_file}")
                    skipped_episodes += 1
                    continue

                command = f.attrs["command"]
                num_frames_ep = f.attrs["num_frames"]

                if num_frames_ep == 0:
                    log.warning(f"Skipping empty episode: {hdf5_file}")
                    skipped_episodes += 1
                    continue

                log.info(f"Converting episode {converted_episodes}: {hdf5_file} ({num_frames_ep} frames)")

                # 데이터 로드
                positions = f["position"][:]
                orientations = f["orientation"][:]
                linear_velocities = f["linear_velocity"][:]
                angular_velocities = f["angular_velocity"][:]
                actions = f["action"][:]
                distances = f["distance_to_goal"][:]

                rgb_data = f["rgb"][:] if has_rgb else None
                depth_data = f["depth"][:] if has_depth else None

                # 에피소드 버퍼 생성
                dataset.episode_buffer = dataset.create_episode_buffer(episode_index=converted_episodes)

                # 프레임 변환
                for frame_idx in range(num_frames_ep):
                    # 상태 벡터 구성
                    state = np.concatenate([
                        positions[frame_idx],           # 3
                        orientations[frame_idx],        # 4 (quaternion)
                        linear_velocities[frame_idx],   # 3
                        angular_velocities[frame_idx],  # 3
                    ]).astype(np.float32)

                    frame = {
                        "observation.state": state,
                        "action": actions[frame_idx].astype(np.float32),
                        "observation.goal_distance": np.array([distances[frame_idx]], dtype=np.float32),
                        "task": command if isinstance(command, str) else command.decode("utf-8"),
                    }

                    # RGB 이미지
                    if rgb_data is not None:
                        rgb_img = rgb_data[frame_idx]
                        if rgb_img.max() <= 1.0:
                            rgb_img = (rgb_img * 255).astype(np.uint8)
                        frame["observation.images.rgb"] = PIL.Image.fromarray(rgb_img)

                    # Depth 이미지 (RGB로 변환)
                    if depth_data is not None:
                        depth = depth_data[frame_idx]
                        # Depth를 0-255로 정규화
                        depth_normalized = np.clip(depth, 0, 10) / 10.0
                        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                        # Grayscale을 RGB로 변환
                        if len(depth_uint8.shape) == 2:
                            depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
                        elif depth_uint8.shape[-1] == 1:
                            depth_rgb = np.concatenate([depth_uint8] * 3, axis=-1)
                        else:
                            depth_rgb = depth_uint8[:, :, :3]
                        frame["observation.images.depth"] = PIL.Image.fromarray(depth_rgb)

                    dataset.add_frame(frame)

                # 에피소드 저장
                dataset.save_episode()

                converted_episodes += 1
                total_frames += num_frames_ep

        except Exception as e:
            log.error(f"Failed to convert {hdf5_file}: {e}")
            import traceback
            traceback.print_exc()
            skipped_episodes += 1
            continue

    # 데이터셋 마무리
    dataset.finalize()

    log.info(f"\n=== Conversion Complete ===")
    log.info(f"Converted episodes: {converted_episodes}")
    log.info(f"Skipped episodes: {skipped_episodes}")
    log.info(f"Total frames: {total_frames}")
    log.info(f"Output path: {output_path / repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 VLA dataset to LeRobot format")
    parser.add_argument("--input_dir", type=str, required=True, help="Input HDF5 dataset directory")
    parser.add_argument("--output_dir", type=str, default="./lerobot_dataset", help="Output directory")
    parser.add_argument("--repo_id", type=str, default="igibson_nav_converted", help="LeRobot dataset ID")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--no_video", action="store_true", help="Save as images instead of video")
    parser.add_argument("--filter_success", action="store_true", help="Only convert successful episodes")
    args = parser.parse_args()

    convert_hdf5_to_lerobot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        use_videos=not args.no_video,
        filter_success=args.filter_success,
    )


if __name__ == "__main__":
    main()
