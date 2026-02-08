#!/usr/bin/env python3
"""
VLA 데이터셋 로더

수집된 HDF5 데이터를 로드하여 학습에 사용할 수 있는 형태로 변환합니다.
PyTorch Dataset 형식으로 제공합니다.
"""

import glob
import json
import os

import h5py
import numpy as np


class VLAEpisode:
    """단일 에피소드 데이터 클래스"""

    def __init__(self, hdf5_path):
        self.path = hdf5_path
        self._load_metadata()

    def _load_metadata(self):
        """메타데이터만 로드 (lazy loading)"""
        with h5py.File(self.path, "r") as f:
            self.command = f.attrs["command"]
            self.scene_id = f.attrs["scene_id"]
            self.num_frames = f.attrs["num_frames"]
            self.success = f.attrs["success"]
            self.geodesic_dist = f.attrs["geodesic_dist"]

    def load_full(self):
        """전체 에피소드 데이터 로드"""
        data = {}
        with h5py.File(self.path, "r") as f:
            # 메타데이터
            data["command"] = f.attrs["command"]
            data["scene_id"] = f.attrs["scene_id"]
            data["num_frames"] = f.attrs["num_frames"]
            data["success"] = f.attrs["success"]

            # 목표/시작 위치
            data["target_pos"] = f["target_pos"][:]
            data["initial_pos"] = f["initial_pos"][:]

            # 웨이포인트
            if "waypoints" in f:
                data["waypoints"] = f["waypoints"][:]

            # 프레임 데이터
            if "rgb" in f:
                data["rgb"] = f["rgb"][:]
            if "depth" in f:
                data["depth"] = f["depth"][:]
            if "scan" in f:
                data["scan"] = f["scan"][:]

            # 로봇 상태
            data["position"] = f["position"][:]
            data["orientation"] = f["orientation"][:]
            data["rpy"] = f["rpy"][:]
            data["linear_velocity"] = f["linear_velocity"][:]
            data["angular_velocity"] = f["angular_velocity"][:]

            # 액션
            data["action"] = f["action"][:]

            # 거리
            data["distance_to_goal"] = f["distance_to_goal"][:]

            # LiDAR (가변 길이)
            if "lidar" in f:
                lidar_group = f["lidar"]
                data["lidar"] = []
                for i in range(self.num_frames):
                    key = f"frame_{i:06d}"
                    if key in lidar_group:
                        data["lidar"].append(lidar_group[key][:])
                    else:
                        data["lidar"].append(np.zeros((0, 3)))

        return data

    def load_frame(self, frame_idx):
        """특정 프레임만 로드"""
        with h5py.File(self.path, "r") as f:
            frame = {
                "command": f.attrs["command"],
                "rgb": f["rgb"][frame_idx] if "rgb" in f else None,
                "depth": f["depth"][frame_idx] if "depth" in f else None,
                "scan": f["scan"][frame_idx] if "scan" in f else None,
                "position": f["position"][frame_idx],
                "orientation": f["orientation"][frame_idx],
                "rpy": f["rpy"][frame_idx],
                "linear_velocity": f["linear_velocity"][frame_idx],
                "angular_velocity": f["angular_velocity"][frame_idx],
                "action": f["action"][frame_idx],
                "distance_to_goal": f["distance_to_goal"][frame_idx],
            }

            # LiDAR
            if "lidar" in f:
                key = f"frame_{frame_idx:06d}"
                if key in f["lidar"]:
                    frame["lidar"] = f["lidar"][key][:]
                else:
                    frame["lidar"] = np.zeros((0, 3))

        return frame

    def __len__(self):
        return self.num_frames

    def __repr__(self):
        return f"VLAEpisode(path={self.path}, frames={self.num_frames}, success={self.success})"


class VLADataset:
    """VLA 데이터셋 클래스"""

    def __init__(self, data_dir, filter_success=False):
        """
        Args:
            data_dir: HDF5 파일이 있는 디렉토리
            filter_success: True면 성공한 에피소드만 로드
        """
        self.data_dir = data_dir
        self.episodes = []

        # HDF5 파일 찾기
        hdf5_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
        print(f"Found {len(hdf5_files)} episode files")

        for path in hdf5_files:
            ep = VLAEpisode(path)
            if filter_success and not ep.success:
                continue
            self.episodes.append(ep)

        print(f"Loaded {len(self.episodes)} episodes")

        # 통계 로드
        stats_file = os.path.join(data_dir, "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                self.stats = json.load(f)
        else:
            self.stats = {}

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]

    def get_all_frames(self):
        """모든 프레임을 순회하는 제너레이터"""
        for ep_idx, episode in enumerate(self.episodes):
            data = episode.load_full()
            for frame_idx in range(episode.num_frames):
                yield {
                    "episode_idx": ep_idx,
                    "frame_idx": frame_idx,
                    "command": data["command"],
                    "rgb": data["rgb"][frame_idx] if "rgb" in data else None,
                    "depth": data["depth"][frame_idx] if "depth" in data else None,
                    "position": data["position"][frame_idx],
                    "rpy": data["rpy"][frame_idx],
                    "action": data["action"][frame_idx],
                }

    def get_statistics(self):
        """데이터셋 통계"""
        total_frames = sum(ep.num_frames for ep in self.episodes)
        success_episodes = sum(1 for ep in self.episodes if ep.success)

        return {
            "num_episodes": len(self.episodes),
            "num_frames": total_frames,
            "success_rate": success_episodes / len(self.episodes) if self.episodes else 0,
            "avg_frames_per_episode": total_frames / len(self.episodes) if self.episodes else 0,
        }


# PyTorch Dataset (선택적)
try:
    import torch
    from torch.utils.data import Dataset

    class VLATorchDataset(Dataset):
        """PyTorch Dataset 형식"""

        def __init__(self, data_dir, filter_success=False, transform=None):
            self.dataset = VLADataset(data_dir, filter_success)
            self.transform = transform

            # 프레임 인덱스 매핑 생성
            self.frame_map = []  # (episode_idx, frame_idx)
            for ep_idx, episode in enumerate(self.dataset.episodes):
                for frame_idx in range(episode.num_frames):
                    self.frame_map.append((ep_idx, frame_idx))

        def __len__(self):
            return len(self.frame_map)

        def __getitem__(self, idx):
            ep_idx, frame_idx = self.frame_map[idx]
            episode = self.dataset.episodes[ep_idx]
            frame = episode.load_frame(frame_idx)

            # 텐서 변환
            sample = {
                "command": frame["command"],
                "rgb": torch.from_numpy(frame["rgb"]).permute(2, 0, 1).float() / 255.0 if frame["rgb"] is not None else None,
                "depth": torch.from_numpy(frame["depth"]).float() if frame["depth"] is not None else None,
                "position": torch.from_numpy(frame["position"]).float(),
                "rpy": torch.from_numpy(frame["rpy"]).float(),
                "action": torch.from_numpy(frame["action"]).float(),
            }

            if self.transform:
                sample = self.transform(sample)

            return sample

except ImportError:
    VLATorchDataset = None
    print("PyTorch not installed. VLATorchDataset not available.")


def visualize_episode(episode_path, output_dir=None):
    """에피소드 시각화"""
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Skipping visualization.")
        return

    episode = VLAEpisode(episode_path)
    data = episode.load_full()

    print(f"Episode: {episode_path}")
    print(f"Command: {data['command']}")
    print(f"Frames: {data['num_frames']}")
    print(f"Success: {data['success']}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i in range(min(10, data["num_frames"])):  # 처음 10프레임만
        if data.get("rgb") is not None:
            rgb = data["rgb"][i]

            # 정보 오버레이
            info_text = f"Frame {i} | Action: [{data['action'][i][0]:.2f}, {data['action'][i][1]:.2f}]"
            cv2.putText(rgb, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved visualization to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLA Dataset Loader")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--visualize", type=str, default=None, help="Episode to visualize")
    parser.add_argument("--output_dir", type=str, default="./vis", help="Visualization output")
    args = parser.parse_args()

    if args.visualize:
        visualize_episode(args.visualize, args.output_dir)
    else:
        # 데이터셋 통계 출력
        dataset = VLADataset(args.data_dir)
        stats = dataset.get_statistics()

        print("\n=== Dataset Statistics ===")
        print(f"Episodes: {stats['num_episodes']}")
        print(f"Total Frames: {stats['num_frames']}")
        print(f"Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"Avg Frames/Episode: {stats['avg_frames_per_episode']:.1f}")
