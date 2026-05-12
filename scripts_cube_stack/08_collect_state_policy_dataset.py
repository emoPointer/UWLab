#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect cube-stack state-policy rollouts into an HDF5 dataset."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts" / "reinforcement_learning" / "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402


DEFAULT_HYDRA_OVERRIDES = [
    "env.scene.insertive_object=cube",
    "env.scene.receptive_object=cube",
]


parser = argparse.ArgumentParser(description="Collect cube-stack state-policy data with an RSL-RL policy.")
parser.add_argument("--task", type=str, default="OmniReset-Arx5-OSC-State-Deploy-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--env_spacing", type=float, default=3.0)
parser.add_argument("--num_demos", type=int, default=50)
parser.add_argument("--output_file", type=str, default="./datasets/cube_stack_state_policy.hdf5")
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--max_steps_per_demo", type=int, default=160)
parser.add_argument("--save_failed", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--ee_body_name", type=str, default="link6")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + DEFAULT_HYDRA_OVERRIDES + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from play_checkpoint_utils import load_runner_checkpoint_for_play  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@dataclass
class EpisodeBuffer:
    actions: list[np.ndarray] = field(default_factory=list)
    table_cam: list[np.ndarray] = field(default_factory=list)
    wrist_cam: list[np.ndarray] = field(default_factory=list)
    eef_pos: list[np.ndarray] = field(default_factory=list)
    eef_rot_6d: list[np.ndarray] = field(default_factory=list)
    insertive_cube_pos: list[np.ndarray] = field(default_factory=list)
    insertive_cube_quat: list[np.ndarray] = field(default_factory=list)
    receptive_cube_pos: list[np.ndarray] = field(default_factory=list)
    receptive_cube_quat: list[np.ndarray] = field(default_factory=list)

    def append(self, sample: dict[str, np.ndarray]) -> None:
        self.actions.append(sample["actions"])
        self.table_cam.append(sample["table_cam"])
        self.wrist_cam.append(sample["wrist_cam"])
        self.eef_pos.append(sample["eef_pos"])
        self.eef_rot_6d.append(sample["eef_rot_6d"])
        self.insertive_cube_pos.append(sample["insertive_cube_pos"])
        self.insertive_cube_quat.append(sample["insertive_cube_quat"])
        self.receptive_cube_pos.append(sample["receptive_cube_pos"])
        self.receptive_cube_quat.append(sample["receptive_cube_quat"])

    def __len__(self) -> int:
        return len(self.actions)

    def reset(self) -> None:
        self.actions.clear()
        self.table_cam.clear()
        self.wrist_cam.clear()
        self.eef_pos.clear()
        self.eef_rot_6d.clear()
        self.insertive_cube_pos.clear()
        self.insertive_cube_quat.clear()
        self.receptive_cube_pos.clear()
        self.receptive_cube_quat.clear()


def _stack(values: list[np.ndarray]) -> np.ndarray:
    return np.stack(values, axis=0)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _find_single_body_id(robot, body_name: str) -> int:
    body_ids, body_names = robot.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one body matching {body_name!r}, found {body_names}.")
    return int(body_ids[0])


def _camera_rgb(scene, camera_name: str) -> np.ndarray:
    camera = scene.sensors.get(camera_name)
    if camera is None:
        raise RuntimeError(f"Missing camera sensor: {camera_name}")
    rgb = camera.data.output["rgb"][..., :3]
    return _to_numpy(rgb).astype(np.uint8, copy=False)


def _rotation_6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
    rot_matrix = matrix_from_quat(quat)
    return rot_matrix[:, :, :2].transpose(1, 2).reshape(quat.shape[0], 6)


def _collect_step_samples(unwrapped_env, actions: torch.Tensor, ee_body_id: int) -> list[dict[str, np.ndarray]]:
    scene = unwrapped_env.scene
    robot = scene["robot"]
    camera_name_to_hdf5_key = {"external_camera": "table_cam", "wrist_camera": "wrist_cam"}
    camera_frames = {
        hdf5_key: _camera_rgb(scene, camera_name)
        for camera_name, hdf5_key in camera_name_to_hdf5_key.items()
    }

    eef_pos = _to_numpy(robot.data.body_link_pos_w[:, ee_body_id])
    eef_rot_6d = _to_numpy(_rotation_6d_from_quat(robot.data.body_link_quat_w[:, ee_body_id]))
    insertive_cube_pos = _to_numpy(scene["insertive_object"].data.root_pos_w)
    insertive_cube_quat = _to_numpy(scene["insertive_object"].data.root_quat_w)
    receptive_cube_pos = _to_numpy(scene["receptive_object"].data.root_pos_w)
    receptive_cube_quat = _to_numpy(scene["receptive_object"].data.root_quat_w)
    action_np = _to_numpy(actions)

    samples = []
    for env_id in range(unwrapped_env.num_envs):
        samples.append(
            {
                "actions": action_np[env_id],
                "table_cam": camera_frames["table_cam"][env_id],
                "wrist_cam": camera_frames["wrist_cam"][env_id],
                "eef_pos": eef_pos[env_id],
                "eef_rot_6d": eef_rot_6d[env_id],
                "insertive_cube_pos": insertive_cube_pos[env_id],
                "insertive_cube_quat": insertive_cube_quat[env_id],
                "receptive_cube_pos": receptive_cube_pos[env_id],
                "receptive_cube_quat": receptive_cube_quat[env_id],
            }
        )
    return samples


def _demo_output_path(output_path: Path, demo_index: int) -> Path:
    if output_path.suffix.lower() in {".hdf5", ".h5"}:
        return output_path.with_name(f"{output_path.stem}_demo_{demo_index:06d}{output_path.suffix}")
    return output_path / f"demo_{demo_index:06d}.hdf5"


def _demo_video_path(demo_output_path: Path) -> Path:
    return demo_output_path.with_suffix(".mp4")


def _write_demo_video(video_path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        print(f"[WARN] No table camera frames captured; skipping video: {video_path}", flush=True)
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_demo(
    output_file: Path,
    episode: EpisodeBuffer,
    *,
    env_name: str,
    checkpoint_path: str,
    demo_index: int,
    source_env_id: int,
    source_env_origin: np.ndarray,
    env_spacing: float,
    num_envs: int,
    success: bool,
    control_frequency_hz: float,
    table_cam_video_path: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(output_file, "w")
    h5_file.attrs["env_name"] = env_name
    h5_file.attrs["checkpoint"] = checkpoint_path
    h5_file.attrs["control_frequency_hz"] = control_frequency_hz
    h5_file.attrs["schema"] = "uwlab_cube_stack_state_policy_v2"
    h5_file.attrs["demo_index"] = demo_index
    h5_file.attrs["source_env_id"] = source_env_id
    h5_file.attrs["source_env_origin"] = source_env_origin.astype(np.float32, copy=False)
    h5_file.attrs["env_spacing"] = float(env_spacing)
    h5_file.attrs["source_num_envs"] = int(num_envs)
    h5_file.attrs["num_demos"] = 1
    h5_file.attrs["total_finished_rollouts"] = 1
    h5_file.attrs["table_cam_video"] = str(table_cam_video_path)

    obs_group = h5_file.create_group("obs")
    h5_file.create_dataset("actions", data=_stack(episode.actions), compression="gzip", shuffle=True)
    obs_group.create_dataset("table_cam", data=_stack(episode.table_cam), compression="gzip", shuffle=True)
    obs_group.create_dataset("wrist_cam", data=_stack(episode.wrist_cam), compression="gzip", shuffle=True)
    obs_group.create_dataset("eef_pos", data=_stack(episode.eef_pos), compression="gzip", shuffle=True)
    obs_group.create_dataset("eef_rot_6d", data=_stack(episode.eef_rot_6d), compression="gzip", shuffle=True)
    obs_group.create_dataset(
        "insertive_cube_pos", data=_stack(episode.insertive_cube_pos), compression="gzip", shuffle=True
    )
    obs_group.create_dataset(
        "insertive_cube_quat", data=_stack(episode.insertive_cube_quat), compression="gzip", shuffle=True
    )
    obs_group.create_dataset(
        "receptive_cube_pos", data=_stack(episode.receptive_cube_pos), compression="gzip", shuffle=True
    )
    obs_group.create_dataset(
        "receptive_cube_quat", data=_stack(episode.receptive_cube_quat), compression="gzip", shuffle=True
    )
    h5_file.attrs["num_samples"] = len(episode)
    h5_file.attrs["success"] = bool(success)
    h5_file.attrs["control_frequency_hz"] = control_frequency_hz
    h5_file.close()


def _make_runner(env: RslRlVecEnvWrapper, agent_cfg: RslRlBaseRunnerCfg, log_dir: str | None):
    if agent_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if agent_cfg.class_name == "DistillationRunner":
        return DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")


def _success_mask(unwrapped_env, dones: torch.Tensor) -> torch.Tensor:
    success = torch.zeros_like(dones, dtype=torch.bool)
    term_manager = unwrapped_env.termination_manager
    if "success" in term_manager.active_terms:
        success |= term_manager.get_term("success")
    return success


def _set_fixed_robot_workspace_reset(env_cfg: ManagerBasedRLEnvCfg) -> None:
    env_cfg.events.reset_from_reset_states = EventTerm(
        func=task_mdp.FixedRobotWorkspaceTaskPairReset,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "table_cfg": SceneEntityCfg("table"),
            "robot_pose": (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0),
            "table_pose": (0.0, 0.0, 0.799375, 1.0, 0.0, 0.0, 0.0),
            "insertive_object_pose": (-0.30, -0.20, 0.87, 1.0, 0.0, 0.0, 0.0),
            "receptive_object_pose": (-0.30, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0),
            "robot_xy_jitter_m": 0.0,
            "workspace_x_range": (-0.4, -0.2),
            "workspace_y_range": (-0.3, -0.1),
            "insertive_workspace_x_range": (-0.4, -0.2),
            "insertive_workspace_y_range": (-0.3, -0.1),
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "log_every_reset": False,
            "sync_visuals": True,
        },
    )
    if hasattr(env_cfg.events, "align_deploy_scene_to_robosuite_table"):
        align_params = env_cfg.events.align_deploy_scene_to_robosuite_table.params
        align_params["task_object_color_range"] = None
        align_params["insertive_object_color"] = (0.0, 1.0, 0.0)
        align_params["receptive_object_color"] = (1.0, 0.0, 0.0)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        _set_fixed_robot_workspace_reset(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = _make_runner(env, agent_cfg, log_dir=None)
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    load_runner_checkpoint_for_play(runner, checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic

    unwrapped_env = env.unwrapped
    ee_body_id = _find_single_body_id(unwrapped_env.scene["robot"], args_cli.ee_body_name)
    control_frequency_hz = 1.0 / float(unwrapped_env.step_dt)
    output_path = Path(args_cli.output_file).expanduser()
    if output_path.suffix.lower() in {".hdf5", ".h5"}:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    print(
        "[INFO] Collecting cube-stack policy dataset: "
        f"output_prefix={output_path} num_envs={unwrapped_env.num_envs} num_demos={args_cli.num_demos} "
        f"env_spacing={args_cli.env_spacing:.3f} "
        f"control_frequency_hz={control_frequency_hz:.3f}"
    )

    episode_buffers = [EpisodeBuffer() for _ in range(unwrapped_env.num_envs)]
    env_origins_np = _to_numpy(unwrapped_env.scene.env_origins)
    saved_demos = 0
    total_finished = 0
    obs = env.get_observations()

    while simulation_app.is_running() and saved_demos < args_cli.num_demos:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            step_samples = _collect_step_samples(unwrapped_env, actions, ee_body_id)
            for env_id, sample in enumerate(step_samples):
                episode_buffers[env_id].append(sample)

            obs, _, dones, _ = env.step(actions)
            dones = dones.to(dtype=torch.bool)
            success_mask = _success_mask(unwrapped_env, dones)
            forced_done_ids = [
                env_id
                for env_id, episode in enumerate(episode_buffers)
                if len(episode) >= args_cli.max_steps_per_demo
            ]
            done_env_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1).detach().cpu().tolist()
            forced_only_env_ids = sorted(set(forced_done_ids) - set(done_env_ids))
            done_env_ids = sorted(set(done_env_ids + forced_done_ids))
            if forced_only_env_ids:
                forced_env_ids = torch.tensor(forced_only_env_ids, dtype=torch.long, device=unwrapped_env.device)
                unwrapped_env._reset_idx(forced_env_ids)
                obs = env.get_observations()

            policy_reset_mask = dones.clone()
            if forced_done_ids:
                policy_reset_mask[torch.tensor(forced_done_ids, dtype=torch.long, device=policy_reset_mask.device)] = True
            policy_nn.reset(policy_reset_mask)

            for env_id in done_env_ids:
                success = bool(success_mask[env_id].item()) if env_id < len(success_mask) else False
                if success or args_cli.save_failed:
                    demo_output_path = _demo_output_path(output_path, saved_demos)
                    demo_video_path = _demo_video_path(demo_output_path)
                    _write_demo(
                        demo_output_path,
                        episode_buffers[env_id],
                        env_name=args_cli.task,
                        checkpoint_path=checkpoint_path,
                        demo_index=saved_demos,
                        source_env_id=env_id,
                        source_env_origin=env_origins_np[env_id],
                        env_spacing=args_cli.env_spacing,
                        num_envs=unwrapped_env.num_envs,
                        success=success,
                        control_frequency_hz=control_frequency_hz,
                        table_cam_video_path=demo_video_path,
                    )
                    _write_demo_video(demo_video_path, episode_buffers[env_id].table_cam, fps=control_frequency_hz)
                    saved_demos += 1
                    print(
                        f"[INFO] saved {demo_output_path}: env={env_id} "
                        f"steps={len(episode_buffers[env_id])} success={success} video={demo_video_path}"
                    )
                episode_buffers[env_id].reset()
                total_finished += 1
                if saved_demos >= args_cli.num_demos:
                    break

        if args_cli.real_time:
            sleep_time = float(unwrapped_env.step_dt) - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    simulation_app.close()
    print(f"[INFO] Saved {saved_demos} demo files with prefix {output_path}; finished rollouts={total_finished}")


if __name__ == "__main__":
    main()
