#!/usr/bin/env python3
"""Replay HDF5 actions with Isaac/PhysX physics and MuJoCo rendering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

DEFAULT_HYDRA_OVERRIDES = [
    "env.scene.insertive_object=cube",
    "env.scene.receptive_object=cube",
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", "--dataset_file", dest="dataset_file", required=True)
parser.add_argument("--task", type=str, default="OmniReset-Arx5-OSC-State-Deploy-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=3.0)
parser.add_argument("--ee_body_name", type=str, default="link6")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--metrics_path", type=str, default=None)
parser.add_argument("--mujoco_xml", type=str, default="mujoco_arx5/models/arx5_robosuite_tabletop_dynamic.xml")
parser.add_argument("--mujoco_camera", type=str, default="external_camera")
parser.add_argument("--mujoco_video_path", type=str, default="videos/mujoco_isaac_replays/cube_000000.mp4")
parser.add_argument("--mujoco_video_width", type=int, default=640)
parser.add_argument("--mujoco_video_height", type=int, default=480)
parser.add_argument("--mujoco_insertive_body", type=str, default="insertive_cube")
parser.add_argument("--mujoco_receptive_body", type=str, default="receptive_cube")
parser.add_argument("--randomize_light_angles", action="store_true", default=False)
parser.add_argument("--mujoco_light_yaw_range_deg", type=float, nargs=2, default=(0.0, 360.0))
parser.add_argument("--mujoco_light_elevation_range_deg", type=float, nargs=2, default=(35.0, 75.0))
parser.add_argument("--mujoco_light_distance_range", type=float, nargs=2, default=(2.0, 3.0))
parser.add_argument("--mujoco_light_seed", type=int, default=None)
parser.add_argument(
    "--reset_state_dataset_path",
    type=str,
    default="Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube/resets_ObjectAnywhereEEAnywhere.pt",
)
parser.add_argument("--reset_state_match_tolerance_m", type=float, default=1.0e-4)
parser.add_argument("--no-video", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Deploy-Play contains Isaac camera sensors; the replay video below is still MuJoCo-rendered.
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + DEFAULT_HYDRA_OVERRIDES + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from mujoco_arx5.isaac_render_bridge.hdf5_replay import (  # noqa: E402
    Hdf5ActionReplayConfig,
    Hdf5ActionReplayRunner,
    configure_cube_replay_reset,
)
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing
    if args_cli.num_envs != 1:
        raise ValueError("This replay script expects --num_envs 1 for a single HDF5 rollout.")
    configure_cube_replay_reset(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    config = Hdf5ActionReplayConfig(
        dataset=args_cli.dataset_file,
        mujoco_xml=args_cli.mujoco_xml,
        mujoco_camera=args_cli.mujoco_camera,
        mujoco_video_path=args_cli.mujoco_video_path,
        mujoco_video_width=args_cli.mujoco_video_width,
        mujoco_video_height=args_cli.mujoco_video_height,
        mujoco_insertive_body=args_cli.mujoco_insertive_body,
        mujoco_receptive_body=args_cli.mujoco_receptive_body,
        randomize_light_angles=args_cli.randomize_light_angles,
        mujoco_light_yaw_range_deg=tuple(args_cli.mujoco_light_yaw_range_deg),
        mujoco_light_elevation_range_deg=tuple(args_cli.mujoco_light_elevation_range_deg),
        mujoco_light_distance_range=tuple(args_cli.mujoco_light_distance_range),
        mujoco_light_seed=args_cli.mujoco_light_seed,
        reset_state_dataset_path=args_cli.reset_state_dataset_path,
        reset_state_match_tolerance_m=args_cli.reset_state_match_tolerance_m,
        metrics_path=args_cli.metrics_path,
        ee_body_name=args_cli.ee_body_name,
        env_spacing=args_cli.env_spacing,
        record_video=not args_cli.no_video,
        real_time=args_cli.real_time,
    )
    runner = Hdf5ActionReplayRunner(env, config, simulation_app=simulation_app)
    try:
        runner.run_all()
    finally:
        runner.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
