#!/usr/bin/env python3
"""
iGibson -> LeRobot 데이터 수집 파이프라인

LeRobot 데이터 포맷(Parquet + MP4/PNG)으로 직접 저장합니다.
Python 3.8 호환을 위해 LeRobot 라이브러리 대신 직접 포맷 생성.

수집 데이터:
- observation.images.rgb: RGB 카메라 이미지
- observation.images.depth: Depth 이미지
- observation.state: 로봇 상태 (position, orientation, velocity)
- action: 로봇 액션 (linear_vel, angular_vel)
- task: 언어 명령
"""

from __future__ import annotations

import os
# EGL headless 모드를 위해 DISPLAY 환경변수 제거
os.environ.pop("DISPLAY", None)

import argparse
import datetime
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.utils import parse_config, rotate_vector_3d

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# LeRobot v3.0 상수
CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_MB = 100
DEFAULT_VIDEO_FILE_SIZE_MB = 200


class PurePursuitController:
    """Pure Pursuit 경로 추종 컨트롤러"""

    def __init__(self, lookahead_dist=0.5, max_linear_vel=0.5, max_angular_vel=1.0):
        self.lookahead_dist = lookahead_dist
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel

    def reset(self):
        pass

    def get_action(self, robot_pos, robot_yaw, waypoints, goal_pos):
        """Pure Pursuit 알고리즘으로 액션 계산"""
        robot_pos = np.array(robot_pos[:2])
        goal_pos = np.array(goal_pos[:2])

        dist_to_goal = np.linalg.norm(robot_pos - goal_pos)
        if dist_to_goal < 0.3:
            return np.array([0.0, 0.0]), True

        target_point = self._find_lookahead_point(robot_pos, waypoints)
        if target_point is None:
            target_point = goal_pos

        dx = target_point[0] - robot_pos[0]
        dy = target_point[1] - robot_pos[1]
        goal_angle = np.arctan2(dy, dx)

        angle_error = goal_angle - robot_yaw
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        angular_vel = -0.5 * angle_error
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)

        angle_factor = max(0, 1.0 - abs(angle_error) / (np.pi / 2))
        linear_vel = self.max_linear_vel * (0.2 + 0.8 * angle_factor)

        action = np.array([
            linear_vel / self.max_linear_vel,
            angular_vel / self.max_angular_vel
        ])

        return np.clip(action, -1, 1), False

    def _find_lookahead_point(self, robot_pos, waypoints):
        if waypoints is None or len(waypoints) == 0:
            return None
        distances = np.linalg.norm(waypoints - robot_pos, axis=1)
        closest_idx = np.argmin(distances)
        for i in range(closest_idx, len(waypoints)):
            if np.linalg.norm(waypoints[i] - robot_pos) >= self.lookahead_dist:
                return waypoints[i]
        return waypoints[-1]


class LeRobotDatasetWriter:
    """LeRobot 포맷 데이터셋 Writer (Python 3.8 호환)"""

    def __init__(
        self,
        root: Path,
        repo_id: str,
        fps: int,
        features: Dict[str, Dict],
        robot_type: str = "turtlebot",
        use_videos: bool = True,
    ):
        self.root = root / repo_id
        self.repo_id = repo_id
        self.fps = fps
        self.features = features
        self.robot_type = robot_type
        self.use_videos = use_videos

        # 디렉토리 생성
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

        # 이미지/비디오 키 분리
        self.image_keys = [k for k, v in features.items() if v.get("dtype") in ["image", "video"]]
        self.state_keys = [k for k, v in features.items() if v.get("dtype") not in ["image", "video"]]

        for key in self.image_keys:
            (self.root / "images" / key.replace(".", "_")).mkdir(parents=True, exist_ok=True)

        # 상태 변수
        self.total_episodes = 0
        self.total_frames = 0
        self.tasks = {}  # task_name -> task_index
        self.episodes_metadata = []
        self.all_stats = {}

        # 에피소드 버퍼
        self.episode_buffer = None

        # info.json 초기화
        self._write_info()

    def _write_info(self):
        """info.json 작성"""
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "fps": self.fps,
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_MB,
            "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_MB,
            "splits": {"train": f"0:{self.total_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": self.features,
        }
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def create_episode_buffer(self, episode_index: int) -> Dict:
        """에피소드 버퍼 생성"""
        return {
            "episode_index": episode_index,
            "frames": [],
            "tasks": [],
        }

    def add_frame(self, frame: Dict):
        """프레임 추가"""
        if self.episode_buffer is None:
            raise RuntimeError("Episode buffer not created. Call create_episode_buffer first.")

        frame_idx = len(self.episode_buffer["frames"])
        ep_idx = self.episode_buffer["episode_index"]

        processed_frame = {
            "frame_index": frame_idx,
            "timestamp": frame_idx / self.fps,
            "episode_index": ep_idx,
        }

        # Task 처리
        task = frame.get("task", "")
        self.episode_buffer["tasks"].append(task)

        # 이미지 저장
        for key in self.image_keys:
            if key in frame and frame[key] is not None:
                img = frame[key]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)

                img_dir = self.root / "images" / key.replace(".", "_") / f"episode-{ep_idx:06d}"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"frame-{frame_idx:06d}.png"
                img.save(img_path)

        # 상태/액션 데이터
        for key in self.state_keys:
            if key in frame:
                processed_frame[key] = frame[key]

        self.episode_buffer["frames"].append(processed_frame)

    def save_episode(self):
        """에피소드 저장"""
        if self.episode_buffer is None or len(self.episode_buffer["frames"]) == 0:
            return

        ep_idx = self.episode_buffer["episode_index"]
        frames = self.episode_buffer["frames"]
        tasks = list(set(self.episode_buffer["tasks"]))
        ep_length = len(frames)

        # Task 인덱스 업데이트
        for task in tasks:
            if task not in self.tasks:
                self.tasks[task] = len(self.tasks)

        # Parquet 데이터 준비
        data = {
            "index": [],
            "frame_index": [],
            "timestamp": [],
            "episode_index": [],
            "task_index": [],
        }

        for key in self.state_keys:
            data[key] = []

        for i, frame in enumerate(frames):
            data["index"].append(self.total_frames + i)
            data["frame_index"].append(frame["frame_index"])
            data["timestamp"].append(frame["timestamp"])
            data["episode_index"].append(frame["episode_index"])

            task = self.episode_buffer["tasks"][i]
            data["task_index"].append(self.tasks.get(task, 0))

            for key in self.state_keys:
                if key in frame:
                    data[key].append(frame[key])

        # Parquet 저장
        chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE
        file_idx = ep_idx % DEFAULT_CHUNK_SIZE

        data_dir = self.root / "data" / f"chunk-{chunk_idx:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)

        # 기존 파일에 append 또는 새 파일 생성
        parquet_path = data_dir / f"file-{file_idx:03d}.parquet"

        # DataFrame 생성
        df = pd.DataFrame(data)

        # 기존 파일이 있으면 합치기
        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_parquet(parquet_path, compression="snappy")

        # 에피소드 메타데이터
        ep_metadata = {
            "episode_index": ep_idx,
            "length": ep_length,
            "tasks": tasks,
            "dataset_from_index": self.total_frames,
            "dataset_to_index": self.total_frames + ep_length,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
        }
        self.episodes_metadata.append(ep_metadata)

        # 통계 업데이트
        self._update_stats(frames)

        # 카운터 업데이트
        self.total_episodes += 1
        self.total_frames += ep_length

        # 메타데이터 파일 업데이트
        self._write_info()
        self._write_tasks()
        self._write_episodes()
        self._write_stats()

        # 비디오 인코딩 (use_videos=True인 경우)
        if self.use_videos:
            self._encode_videos(ep_idx)

        # 버퍼 초기화
        self.episode_buffer = None

    def _update_stats(self, frames: List[Dict]):
        """통계 업데이트"""
        for key in self.state_keys:
            if key not in frames[0]:
                continue

            values = np.array([f[key] for f in frames])

            if key not in self.all_stats:
                self.all_stats[key] = {
                    "min": values.min(axis=0).tolist(),
                    "max": values.max(axis=0).tolist(),
                    "mean": values.mean(axis=0).tolist(),
                    "std": values.std(axis=0).tolist(),
                }
            else:
                # 점진적 업데이트 (간단화)
                old = self.all_stats[key]
                self.all_stats[key] = {
                    "min": np.minimum(old["min"], values.min(axis=0)).tolist(),
                    "max": np.maximum(old["max"], values.max(axis=0)).tolist(),
                    "mean": ((np.array(old["mean"]) + values.mean(axis=0)) / 2).tolist(),
                    "std": ((np.array(old["std"]) + values.std(axis=0)) / 2).tolist(),
                }

    def _write_tasks(self):
        """tasks.parquet 작성"""
        if not self.tasks:
            return
        tasks_df = pd.DataFrame({
            "task_index": list(self.tasks.values()),
        }, index=list(self.tasks.keys()))
        tasks_df.to_parquet(self.root / "meta" / "tasks.parquet")

    def _write_episodes(self):
        """episodes parquet 작성"""
        if not self.episodes_metadata:
            return

        ep_dir = self.root / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # 에피소드 메타데이터를 DataFrame으로 변환
        df = pd.DataFrame(self.episodes_metadata)
        df.to_parquet(ep_dir / "file-000.parquet")

    def _write_stats(self):
        """stats.json 작성"""
        with open(self.root / "meta" / "stats.json", "w") as f:
            json.dump(self.all_stats, f, indent=2)

    def _encode_videos(self, episode_index: int):
        """이미지를 비디오로 인코딩"""
        for key in self.image_keys:
            img_dir = self.root / "images" / key.replace(".", "_") / f"episode-{episode_index:06d}"
            if not img_dir.exists():
                continue

            video_key = key.replace(".", "_")
            video_dir = self.root / "videos" / video_key / "chunk-000"
            video_dir.mkdir(parents=True, exist_ok=True)

            video_path = video_dir / f"episode-{episode_index:06d}.mp4"

            # ffmpeg로 인코딩
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.fps),
                "-i", str(img_dir / "frame-%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                str(video_path)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
                # 이미지 삭제
                shutil.rmtree(img_dir)
                log.info(f"Video encoded: {video_path}")
            except subprocess.CalledProcessError as e:
                log.warning(f"Video encoding failed for {key}: {e}. Keeping images.")
            except FileNotFoundError:
                log.warning(f"ffmpeg not found. Keeping images instead of video for {key}.")

    def finalize(self):
        """데이터셋 마무리"""
        self._write_info()
        self._write_tasks()
        self._write_episodes()
        self._write_stats()
        log.info(f"Dataset finalized: {self.total_episodes} episodes, {self.total_frames} frames")


class LeRobotDataCollector:
    """LeRobot 포맷 데이터 수집 클래스"""

    COMMAND_TEMPLATES = [
        "Navigate to the {target}",
        "Go to the {target}",
        "Move to the {target}",
        "Head towards the {target}",
        "Find your way to the {target}",
    ]

    TARGET_NAMES = [
        "goal point",
        "destination",
        "target location",
        "marked position",
        "waypoint",
    ]

    def __init__(
        self,
        config_path: str,
        output_dir: str,
        repo_id: str = "igibson_nav",
        fps: int = 30,
        use_depth: bool = True,
        use_videos: bool = True,
        image_size: Tuple[int, int] = (480, 640),
    ):
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.fps = fps
        self.use_depth = use_depth
        self.use_videos = use_videos
        self.image_size = image_size

        # Config 로드
        self.config = self._load_config(config_path)

        # 환경 생성
        log.info("Creating iGibson environment...")
        self.env = iGibsonEnv(
            config_file=self.config,
            mode="headless",
            action_timestep=1/self.fps,
            physics_timestep=1/120.0,
        )

        self.controller = PurePursuitController()

        # Features 정의
        features = self._define_features()

        # 기존 디렉토리 삭제
        dataset_path = self.output_dir / self.repo_id
        if dataset_path.exists():
            log.warning(f"Removing existing dataset at {dataset_path}")
            shutil.rmtree(dataset_path)

        # Dataset Writer 생성
        self.writer = LeRobotDatasetWriter(
            root=self.output_dir,
            repo_id=self.repo_id,
            fps=self.fps,
            features=features,
            robot_type="turtlebot",
            use_videos=self.use_videos,
        )

        log.info(f"Environment created. Scene: {self.config.get('scene_id', 'unknown')}")

    def _load_config(self, config_path: str) -> Dict:
        if config_path and os.path.exists(config_path):
            config = parse_config(config_path)
        else:
            config = parse_config(os.path.join(igibson.configs_path, "turtlebot_static_nav.yaml"))

        config["output"] = ["task_obs", "rgb", "depth", "scan"]
        config["image_width"] = self.image_size[1]
        config["image_height"] = self.image_size[0]
        config["task"] = "point_nav_random"
        config["target_dist_min"] = 3.0
        config["target_dist_max"] = 10.0

        return config

    def _define_features(self) -> Dict:
        features = {
            "observation.images.rgb": {
                "dtype": "video" if self.use_videos else "image",
                "shape": [self.image_size[0], self.image_size[1], 3],
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [13],
                "names": [
                    "pos_x", "pos_y", "pos_z",
                    "quat_x", "quat_y", "quat_z", "quat_w",
                    "lin_vel_x", "lin_vel_y", "lin_vel_z",
                    "ang_vel_x", "ang_vel_y", "ang_vel_z",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["linear_velocity", "angular_velocity"],
            },
            "observation.goal_distance": {
                "dtype": "float32",
                "shape": [1],
                "names": ["distance"],
            },
        }

        if self.use_depth:
            features["observation.images.depth"] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": [self.image_size[0], self.image_size[1], 3],
                "names": ["height", "width", "channels"],
            }

        return features

    def _generate_command(self) -> str:
        template = np.random.choice(self.COMMAND_TEMPLATES)
        target = np.random.choice(self.TARGET_NAMES)
        return template.format(target=target)

    def _get_robot_state(self, robot) -> np.ndarray:
        pos = robot.get_position()
        orn = robot.get_orientation()
        rpy = robot.get_rpy()
        lin_vel = robot.get_linear_velocity()
        ang_vel = robot.get_angular_velocity()

        lin_vel_local = rotate_vector_3d(lin_vel, *rpy)
        ang_vel_local = rotate_vector_3d(ang_vel, *rpy)

        state = np.concatenate([
            np.array(pos),
            np.array(orn),
            np.array(lin_vel_local),
            np.array(ang_vel_local),
        ]).astype(np.float32)

        return state

    def _process_rgb(self, obs) -> Optional[Image.Image]:
        if "rgb" in obs:
            rgb = (obs["rgb"] * 255).astype(np.uint8)
            return Image.fromarray(rgb)
        return None

    def _process_depth(self, obs) -> Optional[Image.Image]:
        if "depth" not in obs:
            return None

        depth = obs["depth"]
        depth_normalized = np.clip(depth, 0, 10) / 10.0
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        if len(depth_uint8.shape) == 2:
            depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
        elif depth_uint8.shape[-1] == 1:
            depth_rgb = np.concatenate([depth_uint8] * 3, axis=-1)
        else:
            depth_rgb = depth_uint8

        return Image.fromarray(depth_rgb)

    def collect_episode(self, episode_idx: int, max_steps: int = 500) -> Tuple[int, bool]:
        obs = self.env.reset()
        robot = self.env.robots[0]
        self.controller.reset()

        target_pos = self.env.task.target_pos
        initial_pos = self.env.task.initial_pos

        try:
            waypoints, geodesic_dist = self.env.scene.get_shortest_path(
                floor=self.env.task.floor_num,
                source_world=initial_pos[:2],
                target_world=target_pos[:2],
                entire_path=True
            )
        except Exception as e:
            log.warning(f"Failed to compute path: {e}")
            waypoints = None
            geodesic_dist = np.linalg.norm(target_pos[:2] - initial_pos[:2])

        command = self._generate_command()
        log.info(f"Episode {episode_idx}: '{command}' | Distance: {geodesic_dist:.2f}m")

        self.writer.episode_buffer = self.writer.create_episode_buffer(episode_idx)

        done = False
        success = False
        num_frames = 0

        for step in range(max_steps):
            robot_state = self._get_robot_state(robot)
            robot_pos = robot.get_position()
            robot_yaw = robot.get_rpy()[2]

            action, reached = self.controller.get_action(
                robot_pos[:2], robot_yaw, waypoints, target_pos[:2]
            )

            if reached:
                success = True
                done = True

            dist_to_goal = np.linalg.norm(robot_pos[:2] - target_pos[:2])

            frame = {
                "observation.images.rgb": self._process_rgb(obs),
                "observation.state": robot_state,
                "action": action.astype(np.float32),
                "observation.goal_distance": np.array([dist_to_goal], dtype=np.float32),
                "task": command,
            }

            if self.use_depth:
                depth_img = self._process_depth(obs)
                if depth_img is not None:
                    frame["observation.images.depth"] = depth_img

            self.writer.add_frame(frame)
            num_frames += 1

            obs, reward, env_done, info = self.env.step(action)

            if env_done or done:
                success = info.get("success", success)
                break

        self.writer.save_episode()
        log.info(f"Episode {episode_idx} finished: {num_frames} frames, Success: {success}")

        return num_frames, success

    def collect_dataset(self, num_episodes: int, max_steps_per_episode: int = 500) -> Dict:
        log.info(f"Starting data collection: {num_episodes} episodes")

        stats = {
            "total_episodes": num_episodes,
            "successful_episodes": 0,
            "total_frames": 0,
            "scene_id": self.config.get("scene_id", "unknown"),
            "start_time": datetime.datetime.now().isoformat(),
        }

        for episode_idx in range(num_episodes):
            try:
                num_frames, success = self.collect_episode(episode_idx, max_steps_per_episode)
                stats["total_frames"] += num_frames
                if success:
                    stats["successful_episodes"] += 1
            except Exception as e:
                log.error(f"Episode {episode_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        stats["end_time"] = datetime.datetime.now().isoformat()
        stats["success_rate"] = stats["successful_episodes"] / num_episodes if num_episodes > 0 else 0

        self.writer.finalize()

        log.info(f"Data collection complete!")
        log.info(f"Total episodes: {num_episodes}")
        log.info(f"Successful episodes: {stats['successful_episodes']} ({stats['success_rate']*100:.1f}%)")
        log.info(f"Total frames: {stats['total_frames']}")
        log.info(f"Dataset saved at: {self.output_dir / self.repo_id}")

        return stats

    def close(self):
        if hasattr(self, 'writer'):
            self.writer.finalize()
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="iGibson -> LeRobot Data Collection")
    parser.add_argument("--config", type=str, default=None, help="iGibson config file path")
    parser.add_argument("--scene", type=str, default="Rs", help="Scene ID")
    parser.add_argument("--output_dir", type=str, default="./lerobot_dataset", help="Output directory")
    parser.add_argument("--repo_id", type=str, default="igibson_nav", help="LeRobot dataset ID")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--no_depth", action="store_true", help="Disable depth image collection")
    parser.add_argument("--no_video", action="store_true", help="Save as images instead of video")
    parser.add_argument("--image_height", type=int, default=480, help="Image height")
    parser.add_argument("--image_width", type=int, default=640, help="Image width")
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(igibson.configs_path, "turtlebot_static_nav.yaml")

    collector = LeRobotDataCollector(
        config_path=config_path,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        use_depth=not args.no_depth,
        use_videos=not args.no_video,
        image_size=(args.image_height, args.image_width),
    )

    if args.scene:
        collector.config["scene_id"] = args.scene

    try:
        stats = collector.collect_dataset(
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
        )
    finally:
        collector.close()


if __name__ == "__main__":
    main()
