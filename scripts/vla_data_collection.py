#!/usr/bin/env python3
"""
VLA (Vision-Language-Action) 데이터 수집 파이프라인

수집 데이터:
- Command: 언어 명령 (예: "Go to the kitchen")
- RGB 이미지
- Velodyne 16ch LiDAR
- 로봇 Pose (position, orientation, velocity)
- Action (linear, angular velocity)
- Planned path (waypoints)

Non-Interactive DB (Gibson)에서도 동작합니다.
"""

import os
# EGL headless 모드를 위해 DISPLAY 환경변수 제거 (필수!)
os.environ.pop("DISPLAY", None)

import argparse
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.utils import parse_config, rotate_vector_3d

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class PurePursuitController:
    """Pure Pursuit 경로 추종 컨트롤러"""

    def __init__(self, lookahead_dist=0.5, max_linear_vel=0.5, max_angular_vel=1.0):
        self.lookahead_dist = lookahead_dist
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.waypoint_idx = 0

    def reset(self):
        self.waypoint_idx = 0

    def get_action(self, robot_pos, robot_yaw, waypoints, goal_pos):
        """
        Pure Pursuit 알고리즘으로 액션 계산

        Args:
            robot_pos: 로봇 위치 [x, y]
            robot_yaw: 로봇 방향 (rad)
            waypoints: 경로 웨이포인트 (N, 2)
            goal_pos: 최종 목표 위치 [x, y]

        Returns:
            action: [linear_vel, angular_vel] (normalized to [-1, 1])
            done: 목표 도달 여부
        """
        robot_pos = np.array(robot_pos[:2])
        goal_pos = np.array(goal_pos[:2])

        # 목표 도달 체크
        dist_to_goal = np.linalg.norm(robot_pos - goal_pos)
        if dist_to_goal < 0.3:
            return np.array([0.0, 0.0]), True

        # Lookahead point 찾기
        target_point = self._find_lookahead_point(robot_pos, waypoints)
        if target_point is None:
            target_point = goal_pos

        # 목표 방향 계산
        dx = target_point[0] - robot_pos[0]
        dy = target_point[1] - robot_pos[1]
        goal_angle = np.arctan2(dy, dx)

        # 각도 오차 계산 및 정규화
        angle_error = goal_angle - robot_yaw
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # 더 안정적인 정규화

        # 부드러운 제어 (낮은 gain)
        # iGibson에서 양수 angular = 반시계 방향이므로 부호 반전
        angular_vel = -0.5 * angle_error  # gain = 0.5, 부호 반전
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)

        # 각도 오차에 따른 전진 속도 (부드러운 전환)
        # 오차가 작을수록 더 빠르게 전진
        angle_factor = max(0, 1.0 - abs(angle_error) / (np.pi / 2))  # 90도에서 0, 0도에서 1
        linear_vel = self.max_linear_vel * (0.2 + 0.8 * angle_factor)  # 최소 20% 속도 유지

        # Normalize to [-1, 1]
        action = np.array([
            linear_vel / self.max_linear_vel,
            angular_vel / self.max_angular_vel
        ])

        return np.clip(action, -1, 1), False

    def _find_lookahead_point(self, robot_pos, waypoints):
        """Lookahead 거리에 있는 웨이포인트 찾기"""
        if waypoints is None or len(waypoints) == 0:
            return None

        # 현재 위치에서 가장 가까운 웨이포인트부터 시작
        distances = np.linalg.norm(waypoints - robot_pos, axis=1)
        closest_idx = np.argmin(distances)

        # Lookahead 거리 이상 떨어진 첫 번째 웨이포인트 찾기
        for i in range(closest_idx, len(waypoints)):
            if np.linalg.norm(waypoints[i] - robot_pos) >= self.lookahead_dist:
                return waypoints[i]

        # 없으면 마지막 웨이포인트 반환
        return waypoints[-1]


class VLADataCollector:
    """VLA 데이터 수집 클래스"""

    # 언어 명령 템플릿
    COMMAND_TEMPLATES = [
        "Navigate to the {target}",
        "Go to the {target}",
        "Move to the {target}",
        "Head towards the {target}",
        "Find your way to the {target}",
    ]

    # 목표 위치 이름 (실제로는 랜덤 포인트이지만 설명용)
    TARGET_NAMES = [
        "goal point",
        "destination",
        "target location",
        "marked position",
        "waypoint",
    ]

    def __init__(self, config_path, output_dir, use_velodyne=True):
        """
        Args:
            config_path: iGibson config 파일 경로
            output_dir: 데이터 저장 디렉토리
            use_velodyne: Velodyne 16ch LiDAR 사용 여부
        """
        self.output_dir = output_dir
        self.use_velodyne = use_velodyne
        os.makedirs(output_dir, exist_ok=True)

        # Config 로드 및 수정
        self.config = self._load_config(config_path)

        # 환경 생성
        log.info("Creating iGibson environment...")
        self.env = iGibsonEnv(
            config_file=self.config,
            mode="headless",
            action_timestep=1/30.0,
            physics_timestep=1/120.0,
        )

        # 컨트롤러 생성
        self.controller = PurePursuitController(
            lookahead_dist=0.5,
            max_linear_vel=0.5,
            max_angular_vel=1.0
        )

        log.info(f"Environment created. Scene: {self.config.get('scene_id', 'unknown')}")

    def _load_config(self, config_path):
        """Config 파일 로드 및 VLA 수집용으로 수정"""
        if config_path and os.path.exists(config_path):
            config = parse_config(config_path)
        else:
            # 기본 config
            config = parse_config(os.path.join(igibson.configs_path, "turtlebot_static_nav.yaml"))

        # 센서 출력 설정
        config["output"] = ["task_obs", "rgb", "depth", "scan"]

        # 이미지 해상도
        config["image_width"] = config.get("image_width", 640)
        config["image_height"] = config.get("image_height", 480)

        # Velodyne 16ch 설정
        if self.use_velodyne:
            config["n_horizontal_rays"] = 360
            config["n_vertical_beams"] = 16
            config["laser_linear_range"] = 100.0
            config["laser_angular_range"] = 360.0
            config["min_laser_dist"] = 0.1

        # 랜덤 네비게이션 task
        config["task"] = "point_nav_random"
        config["target_dist_min"] = 3.0
        config["target_dist_max"] = 10.0

        return config

    def _generate_command(self):
        """랜덤 언어 명령 생성"""
        template = np.random.choice(self.COMMAND_TEMPLATES)
        target = np.random.choice(self.TARGET_NAMES)
        return template.format(target=target)

    def _get_velodyne_lidar(self):
        """Velodyne 16ch LiDAR 데이터 취득"""
        try:
            # 렌더러에서 LiDAR 데이터 취득
            lidar_points = self.env.simulator.renderer.get_lidar_all()
            return lidar_points
        except Exception as e:
            log.warning(f"Failed to get Velodyne LiDAR: {e}")
            return np.zeros((0, 3))

    def _get_scan_as_pointcloud(self, obs):
        """2D scan을 3D point cloud로 변환 (Velodyne 미지원 시 대체)"""
        if "scan" not in obs:
            return np.zeros((0, 3))

        scan = obs["scan"]
        n_rays = len(scan)

        # Config에서 LiDAR 파라미터 가져오기
        laser_linear_range = self.config.get("laser_linear_range", 5.6)
        laser_angular_range = self.config.get("laser_angular_range", 240.0)
        min_laser_dist = self.config.get("min_laser_dist", 0.05)

        # 각도 계산
        half_range = np.radians(laser_angular_range / 2.0)
        angles = np.linspace(-half_range, half_range, n_rays)

        # 거리 계산 (normalized -> actual)
        distances = scan * (laser_linear_range - min_laser_dist) + min_laser_dist

        # 3D 포인트 계산
        x = distances * np.cos(angles)
        y = distances * np.sin(angles)
        z = np.zeros_like(x)

        return np.stack([x, y, z], axis=1)

    def _get_robot_state(self, robot):
        """로봇 상태 취득"""
        pos = robot.get_position()
        orn = robot.get_orientation()
        rpy = robot.get_rpy()
        lin_vel = robot.get_linear_velocity()
        ang_vel = robot.get_angular_velocity()

        # 로봇 프레임에서의 속도
        lin_vel_local = rotate_vector_3d(lin_vel, *rpy)
        ang_vel_local = rotate_vector_3d(ang_vel, *rpy)

        return {
            "position": np.array(pos),
            "orientation": np.array(orn),
            "rpy": np.array(rpy),
            "linear_velocity": np.array(lin_vel),
            "angular_velocity": np.array(ang_vel),
            "linear_velocity_local": np.array(lin_vel_local),
            "angular_velocity_local": np.array(ang_vel_local),
        }

    def collect_episode(self, episode_idx, max_steps=500):
        """
        단일 에피소드 데이터 수집

        Args:
            episode_idx: 에피소드 인덱스
            max_steps: 최대 스텝 수

        Returns:
            episode_data: 수집된 데이터
            success: 성공 여부
        """
        # 환경 리셋 (랜덤 시작점, 목표점)
        obs = self.env.reset()
        robot = self.env.robots[0]
        self.controller.reset()

        # 목표 정보
        target_pos = self.env.task.target_pos
        initial_pos = self.env.task.initial_pos

        # 최단경로 계획
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

        # 언어 명령 생성
        command = self._generate_command()

        # 데이터 저장용 리스트
        episode_data = {
            "command": command,
            "target_pos": target_pos,
            "initial_pos": initial_pos,
            "waypoints": waypoints,
            "geodesic_dist": geodesic_dist,
            "frames": []
        }

        log.info(f"Episode {episode_idx}: '{command}' | Distance: {geodesic_dist:.2f}m")

        # 에피소드 루프
        done = False
        success = False

        for step in range(max_steps):
            # 현재 로봇 상태
            robot_state = self._get_robot_state(robot)
            robot_pos = robot_state["position"]
            robot_yaw = robot_state["rpy"][2]

            # 액션 계산 (경로 추종)
            action, reached = self.controller.get_action(
                robot_pos[:2], robot_yaw, waypoints, target_pos[:2]
            )

            if reached:
                success = True
                done = True

            # 프레임 데이터 수집
            frame_data = {
                "step": step,
                "timestamp": time.time(),
                # 센서 데이터
                "rgb": (obs["rgb"] * 255).astype(np.uint8) if "rgb" in obs else None,
                "depth": obs.get("depth", None),
                "scan": obs.get("scan", None),
                # LiDAR (Velodyne 또는 2D scan 변환)
                "lidar": self._get_velodyne_lidar() if self.use_velodyne else self._get_scan_as_pointcloud(obs),
                # 로봇 상태
                "position": robot_state["position"],
                "orientation": robot_state["orientation"],
                "rpy": robot_state["rpy"],
                "linear_velocity": robot_state["linear_velocity_local"],
                "angular_velocity": robot_state["angular_velocity_local"],
                # 액션
                "action": action,
                # Task 정보
                "distance_to_goal": np.linalg.norm(robot_pos[:2] - target_pos[:2]),
            }
            episode_data["frames"].append(frame_data)

            # 환경 스텝
            obs, reward, env_done, info = self.env.step(action)

            if env_done or done:
                success = info.get("success", success)
                break

        episode_data["num_steps"] = len(episode_data["frames"])
        episode_data["success"] = success

        log.info(f"Episode {episode_idx} finished: {episode_data['num_steps']} steps, Success: {success}")

        return episode_data, success

    def save_episode_hdf5(self, episode_data, episode_idx):
        """에피소드 데이터를 HDF5로 저장"""
        filename = os.path.join(self.output_dir, f"episode_{episode_idx:06d}.hdf5")
        frames = episode_data["frames"]
        num_frames = len(frames)

        with h5py.File(filename, "w") as f:
            # 메타데이터
            f.attrs["command"] = episode_data["command"]
            f.attrs["scene_id"] = self.config.get("scene_id", "unknown")
            f.attrs["num_frames"] = num_frames
            f.attrs["success"] = episode_data["success"]
            f.attrs["geodesic_dist"] = episode_data["geodesic_dist"]
            f.attrs["timestamp"] = datetime.datetime.now().isoformat()

            # 목표/시작 위치
            f.create_dataset("target_pos", data=episode_data["target_pos"])
            f.create_dataset("initial_pos", data=episode_data["initial_pos"])

            # 경로 웨이포인트
            if episode_data["waypoints"] is not None:
                f.create_dataset("waypoints", data=episode_data["waypoints"])

            # 프레임 데이터
            if num_frames > 0:
                # RGB 이미지
                if frames[0]["rgb"] is not None:
                    rgb_stack = np.stack([fr["rgb"] for fr in frames])
                    f.create_dataset("rgb", data=rgb_stack, compression="gzip", compression_opts=4)

                # Depth
                if frames[0]["depth"] is not None:
                    depth_stack = np.stack([fr["depth"] for fr in frames])
                    f.create_dataset("depth", data=depth_stack, compression="gzip", compression_opts=4)

                # Scan (2D)
                if frames[0]["scan"] is not None:
                    scan_stack = np.stack([fr["scan"] for fr in frames])
                    f.create_dataset("scan", data=scan_stack)

                # LiDAR (3D point cloud) - 가변 길이 처리
                lidar_group = f.create_group("lidar")
                for i, fr in enumerate(frames):
                    if fr["lidar"] is not None and len(fr["lidar"]) > 0:
                        lidar_group.create_dataset(f"frame_{i:06d}", data=fr["lidar"])

                # 로봇 상태
                f.create_dataset("position", data=np.stack([fr["position"] for fr in frames]))
                f.create_dataset("orientation", data=np.stack([fr["orientation"] for fr in frames]))
                f.create_dataset("rpy", data=np.stack([fr["rpy"] for fr in frames]))
                f.create_dataset("linear_velocity", data=np.stack([fr["linear_velocity"] for fr in frames]))
                f.create_dataset("angular_velocity", data=np.stack([fr["angular_velocity"] for fr in frames]))

                # 액션
                f.create_dataset("action", data=np.stack([fr["action"] for fr in frames]))

                # 거리
                f.create_dataset("distance_to_goal", data=np.array([fr["distance_to_goal"] for fr in frames]))

                # 타임스탬프
                f.create_dataset("timestamp", data=np.array([fr["timestamp"] for fr in frames]))

        log.info(f"Saved: {filename}")
        return filename

    def collect_dataset(self, num_episodes, max_steps_per_episode=500):
        """
        전체 데이터셋 수집

        Args:
            num_episodes: 수집할 에피소드 수
            max_steps_per_episode: 에피소드당 최대 스텝 수
        """
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
                episode_data, success = self.collect_episode(episode_idx, max_steps_per_episode)
                self.save_episode_hdf5(episode_data, episode_idx)

                stats["total_frames"] += episode_data["num_steps"]
                if success:
                    stats["successful_episodes"] += 1

            except Exception as e:
                log.error(f"Episode {episode_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        stats["end_time"] = datetime.datetime.now().isoformat()
        stats["success_rate"] = stats["successful_episodes"] / num_episodes if num_episodes > 0 else 0

        # 통계 저장
        stats_file = os.path.join(self.output_dir, "dataset_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        log.info(f"Data collection complete!")
        log.info(f"Total episodes: {num_episodes}")
        log.info(f"Successful episodes: {stats['successful_episodes']} ({stats['success_rate']*100:.1f}%)")
        log.info(f"Total frames: {stats['total_frames']}")

        return stats

    def close(self):
        """환경 종료"""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="VLA Data Collection Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--scene", type=str, default="Rs", help="Scene ID")
    parser.add_argument("--output_dir", type=str, default="./vla_dataset", help="Output directory")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--no_velodyne", action="store_true", help="Disable Velodyne LiDAR")
    args = parser.parse_args()

    # Config 수정
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(igibson.configs_path, "turtlebot_static_nav.yaml")

    # 수집기 생성
    collector = VLADataCollector(
        config_path=config_path,
        output_dir=args.output_dir,
        use_velodyne=not args.no_velodyne,
    )

    # Scene ID 오버라이드
    if args.scene:
        collector.config["scene_id"] = args.scene

    try:
        # 데이터 수집
        stats = collector.collect_dataset(
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
        )
    finally:
        collector.close()


if __name__ == "__main__":
    main()
