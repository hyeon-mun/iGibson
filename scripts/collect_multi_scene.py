#!/usr/bin/env python3
"""
여러 Scene에서 VLA 데이터 수집

Non-Interactive (Gibson) Scene 리스트에서 순차적으로 데이터를 수집합니다.
"""

import argparse
import os
import subprocess
import sys

# Gibson (Non-Interactive) Scene 예시 리스트
GIBSON_SCENES = [
    "Rs",          # 데모용 작은 씬
    "Beechwood",
    "Benevolence",
    "Ihlen",
    "Merom",
    "Pomaria",
    "Wainscott",
]


def main():
    parser = argparse.ArgumentParser(description="Multi-scene VLA Data Collection")
    parser.add_argument("--scenes", nargs="+", default=GIBSON_SCENES[:3],
                        help="Scenes to collect from")
    parser.add_argument("--episodes_per_scene", type=int, default=50,
                        help="Episodes per scene")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--output_base", type=str, default="./vla_dataset",
                        help="Base output directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    collection_script = os.path.join(script_dir, "vla_data_collection.py")
    config_file = os.path.join(script_dir, "configs", "vla_collection_config.yaml")

    print("=" * 60)
    print("Multi-Scene VLA Data Collection")
    print("=" * 60)
    print(f"Scenes: {args.scenes}")
    print(f"Episodes per scene: {args.episodes_per_scene}")
    print(f"Output base: {args.output_base}")
    print("=" * 60)

    for scene in args.scenes:
        output_dir = os.path.join(args.output_base, scene)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n>>> Collecting from scene: {scene}")

        cmd = [
            sys.executable, collection_script,
            "--scene", scene,
            "--num_episodes", str(args.episodes_per_scene),
            "--max_steps", str(args.max_steps),
            "--output_dir", output_dir,
        ]

        if os.path.exists(config_file):
            cmd.extend(["--config", config_file])

        try:
            subprocess.run(cmd, check=True)
            print(f">>> Completed: {scene}")
        except subprocess.CalledProcessError as e:
            print(f">>> Failed: {scene} - {e}")
            continue
        except KeyboardInterrupt:
            print("\n>>> Interrupted by user")
            break

    print("\n" + "=" * 60)
    print("Collection Complete!")
    print("=" * 60)

    # 전체 통계 출력
    total_episodes = 0
    for scene in args.scenes:
        scene_dir = os.path.join(args.output_base, scene)
        if os.path.exists(scene_dir):
            n_files = len([f for f in os.listdir(scene_dir) if f.endswith(".hdf5")])
            print(f"  {scene}: {n_files} episodes")
            total_episodes += n_files

    print(f"\nTotal: {total_episodes} episodes")


if __name__ == "__main__":
    main()
