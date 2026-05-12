#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay collected cube-stack state-policy HDF5 rollouts in Isaac Lab."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_HYDRA_OVERRIDES = [
    "env.scene.insertive_object=cube",
    "env.scene.receptive_object=cube",
]


parser = argparse.ArgumentParser(description="Replay collected cube-stack state-policy rollouts.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="/home/emopointer/UWLab/datasets/cube_stack_state_policy_demo_*.hdf5",
)
parser.add_argument("--task", type=str, default="OmniReset-Arx5-OSC-State-Deploy-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=3.0)
parser.add_argument("--video_path", type=str, default="./videos/cube_stack_replays")
parser.add_argument("--camera_name", type=str, default="external_camera")
parser.add_argument("--ee_body_name", type=str, default="link6")
parser.add_argument("--metrics_path", type=str, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--loop", action="store_true", default=False)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402


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
        align_params["robot_xy_jitter_m"] = 0.0
        align_params["task_object_color_range"] = None
        align_params["insertive_object_color"] = (0.0, 1.0, 0.0)
        align_params["receptive_object_color"] = (1.0, 0.0, 0.0)


def _read_rollout(dataset_file: Path) -> dict[str, np.ndarray]:
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")

    with h5py.File(dataset_file, "r") as h5_file:
        required_keys = [
            "actions",
            "obs/insertive_cube_pos",
            "obs/insertive_cube_quat",
            "obs/receptive_cube_pos",
            "obs/receptive_cube_quat",
        ]
        missing = [key for key in required_keys if key not in h5_file]
        if missing:
            raise KeyError(f"Missing required dataset keys in {dataset_file}: {missing}")
        rollout = {
            "actions": h5_file["actions"][:].astype(np.float32, copy=False),
            "insertive_cube_pos": h5_file["obs/insertive_cube_pos"][:].astype(np.float32, copy=False),
            "insertive_cube_quat": h5_file["obs/insertive_cube_quat"][:].astype(np.float32, copy=False),
            "receptive_cube_pos": h5_file["obs/receptive_cube_pos"][:].astype(np.float32, copy=False),
            "receptive_cube_quat": h5_file["obs/receptive_cube_quat"][:].astype(np.float32, copy=False),
        }
        if "obs/eef_pos" in h5_file:
            rollout["eef_pos"] = h5_file["obs/eef_pos"][:].astype(np.float32, copy=False)
        if "source_env_origin" in h5_file.attrs:
            rollout["source_env_origin"] = np.asarray(h5_file.attrs["source_env_origin"], dtype=np.float32)
        if "source_env_id" in h5_file.attrs:
            rollout["source_env_id"] = np.asarray([h5_file.attrs["source_env_id"]], dtype=np.int64)
        return rollout


def _resolve_dataset_files(dataset_arg: str) -> list[Path]:
    dataset_path = Path(dataset_arg).expanduser()
    if any(char in dataset_arg for char in "*?[]"):
        paths = [Path(path) for path in glob.glob(str(dataset_path))]
    elif dataset_path.is_dir():
        paths = sorted(dataset_path.glob("*.hdf5")) + sorted(dataset_path.glob("*.h5"))
    else:
        paths = [dataset_path]

    files = sorted(path for path in paths if path.is_file())
    if not files:
        raise FileNotFoundError(f"No HDF5 files found from dataset_file={dataset_arg!r}")
    return files


def _video_output_path(video_path_arg: str, dataset_file: Path, multiple_files: bool) -> Path:
    video_path = Path(video_path_arg).expanduser()
    if multiple_files or video_path.suffix.lower() != ".mp4":
        return video_path / f"{dataset_file.stem}_external_replay.mp4"
    return video_path


def _infer_source_env_origin(rollout: dict[str, np.ndarray], env_spacing: float) -> np.ndarray:
    if "source_env_origin" in rollout:
        return rollout["source_env_origin"]

    receptive_xy = rollout["receptive_cube_pos"][0, :2]
    workspace_x_center = -0.3
    workspace_y_center = -0.2
    origin_x = round(float((receptive_xy[0] - workspace_x_center) / env_spacing)) * env_spacing
    origin_y = round(float((receptive_xy[1] - workspace_y_center) / env_spacing)) * env_spacing
    local_x = receptive_xy[0] - origin_x
    local_y = receptive_xy[1] - origin_y
    if not (-0.45 <= local_x <= -0.15 and -0.35 <= local_y <= -0.05):
        origin_x = round(float((receptive_xy[0] - workspace_x_center) / 0.5)) * 0.5
        origin_y = round(float((receptive_xy[1] - workspace_y_center) / 0.5)) * 0.5
    return np.asarray([origin_x, origin_y, 0.0], dtype=np.float32)


def _recorded_eef_local(rollout: dict[str, np.ndarray], source_env_origin: np.ndarray) -> np.ndarray | None:
    if "eef_pos" not in rollout:
        return None
    return rollout["eef_pos"] - source_env_origin.reshape(1, 3)


def _find_single_body_id(robot, body_name: str) -> int:
    body_ids, body_names = robot.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one body matching {body_name!r}, found {body_names}.")
    return int(body_ids[0])


def _eef_local(unwrapped_env, ee_body_id: int) -> np.ndarray:
    robot = unwrapped_env.scene["robot"]
    eef_world = robot.data.body_link_pos_w[0, ee_body_id]
    eef_local = eef_world - unwrapped_env.scene.env_origins[0]
    return eef_local.detach().cpu().numpy().astype(np.float64, copy=False)


def _write_object_pose(
    unwrapped_env,
    asset_name: str,
    pos: np.ndarray,
    quat: np.ndarray,
    source_env_origin: np.ndarray,
) -> None:
    asset = unwrapped_env.scene[asset_name]
    device = unwrapped_env.device
    env_ids = torch.tensor([0], dtype=torch.long, device=device)
    target_pos = pos - source_env_origin + unwrapped_env.scene.env_origins[0].detach().cpu().numpy()
    pose = torch.tensor(np.concatenate((target_pos, quat), axis=0), dtype=torch.float32, device=device).unsqueeze(0)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device), env_ids=env_ids)


def _reset_to_recorded_initial_state(env, rollout: dict[str, np.ndarray]) -> None:
    unwrapped_env = env.unwrapped
    env.reset()
    source_env_origin = _infer_source_env_origin(rollout, env_spacing=float(unwrapped_env.cfg.scene.env_spacing))
    print(f"[INFO] Replay source env origin: {[round(float(v), 4) for v in source_env_origin]}", flush=True)
    _write_object_pose(
        unwrapped_env,
        "insertive_object",
        rollout["insertive_cube_pos"][0],
        rollout["insertive_cube_quat"][0],
        source_env_origin,
    )
    _write_object_pose(
        unwrapped_env,
        "receptive_object",
        rollout["receptive_cube_pos"][0],
        rollout["receptive_cube_quat"][0],
        source_env_origin,
    )
    unwrapped_env.scene.write_data_to_sim()
    unwrapped_env.sim.forward()


def _summarize_eef_error(
    replay_eef: list[np.ndarray],
    recorded_eef: np.ndarray | None,
    *,
    label: str,
    metrics_path: Path | None,
) -> dict[str, object] | None:
    if recorded_eef is None:
        return None
    replay = np.asarray(replay_eef, dtype=np.float64)
    reference = np.asarray(recorded_eef, dtype=np.float64)
    count = min(len(replay), len(reference))
    if count == 0:
        return None
    replay = replay[:count]
    reference = reference[:count]
    error = np.linalg.norm(replay - reference, axis=1)
    summary: dict[str, object] = {
        "label": label,
        "count": int(count),
        "mean_m": float(np.mean(error)),
        "max_m": float(np.max(error)),
        "final_m": float(error[-1]),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "first_replay_eef": replay[0].tolist(),
        "first_hdf5_eef": reference[0].tolist(),
        "final_replay_eef": replay[-1].tolist(),
        "final_hdf5_eef": reference[-1].tolist(),
    }
    print(
        f"[INFO] {label} eef position error vs hdf5: "
        f"count={count} mean={summary['mean_m']:.6f}m "
        f"max={summary['max_m']:.6f}m final={summary['final_m']:.6f}m "
        f"rmse={summary['rmse_m']:.6f}m",
        flush=True,
    )
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[INFO] wrote eef metrics: {metrics_path}", flush=True)
    return summary


def _camera_rgb(unwrapped_env, camera_name: str) -> np.ndarray:
    camera = unwrapped_env.scene.sensors.get(camera_name)
    if camera is None:
        raise RuntimeError(f"Missing camera sensor: {camera_name}")
    return camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8, copy=False)


def _write_video(frames: list[np.ndarray], video_path: Path, fps: float) -> None:
    if not frames:
        print("[WARN] No frames captured; skipping replay video.")
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"[INFO] Saved replay camera video: {video_path}")


def _replay_once(
    env,
    rollout: dict[str, np.ndarray],
    *,
    camera_name: str,
    ee_body_id: int,
    real_time: bool,
    record_video: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    _reset_to_recorded_initial_state(env, rollout)
    actions = torch.tensor(rollout["actions"], dtype=torch.float32, device=env.unwrapped.device)
    step_dt = float(env.unwrapped.step_dt)
    print(f"[INFO] Replaying {actions.shape[0]} steps at {1.0 / step_dt:.3f} Hz.")
    frames = [_camera_rgb(env.unwrapped, camera_name)] if record_video else []
    eef_trace = [_eef_local(env.unwrapped, ee_body_id)]

    for step_id, action in enumerate(actions):
        if not simulation_app.is_running():
            break
        start_time = time.time()
        env.step(action.unsqueeze(0))
        if record_video:
            frames.append(_camera_rgb(env.unwrapped, camera_name))
        eef_trace.append(_eef_local(env.unwrapped, ee_body_id))
        if step_id % 10 == 0:
            print(f"[INFO] replay step {step_id + 1}/{actions.shape[0]}", flush=True)
        if real_time:
            sleep_time = step_dt - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    return frames, eef_trace


def _metrics_output_path(metrics_path_arg: str | None, dataset_file: Path, multiple_files: bool) -> Path | None:
    if metrics_path_arg is None:
        return None
    metrics_path = Path(metrics_path_arg).expanduser()
    if multiple_files or metrics_path.suffix.lower() != ".json":
        return metrics_path / f"{dataset_file.stem}_isaac_eef_metrics.json"
    return metrics_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing
    if args_cli.num_envs != 1:
        raise ValueError("This replay script expects --num_envs 1 for a single HDF5 rollout.")
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        _set_fixed_robot_workspace_reset(env_cfg)

    dataset_files = _resolve_dataset_files(args_cli.dataset_file)
    print(f"[INFO] Found {len(dataset_files)} rollout file(s) to replay.")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    ee_body_id = _find_single_body_id(env.unwrapped.scene["robot"], args_cli.ee_body_name)

    try:
        for file_index, dataset_file in enumerate(dataset_files):
            if not simulation_app.is_running():
                break
            rollout = _read_rollout(dataset_file)
            print(
                f"[INFO] Loaded rollout {file_index + 1}/{len(dataset_files)}: "
                f"{dataset_file} steps={len(rollout['actions'])}"
            )
            video_frames: list[np.ndarray] = []
            eef_trace: list[np.ndarray] = []
            while simulation_app.is_running():
                frames, replay_eef = _replay_once(
                    env,
                    rollout,
                    camera_name=args_cli.camera_name,
                    ee_body_id=ee_body_id,
                    real_time=args_cli.real_time,
                    record_video=bool(args_cli.video_path),
                )
                video_frames.extend(frames)
                eef_trace.extend(replay_eef)
                if not args_cli.loop:
                    break
            source_env_origin = _infer_source_env_origin(
                rollout, env_spacing=float(env.unwrapped.cfg.scene.env_spacing)
            )
            _summarize_eef_error(
                eef_trace,
                _recorded_eef_local(rollout, source_env_origin),
                label="isaac",
                metrics_path=_metrics_output_path(
                    args_cli.metrics_path, dataset_file, multiple_files=len(dataset_files) > 1
                ),
            )
            if args_cli.video_path:
                _write_video(
                    video_frames,
                    _video_output_path(args_cli.video_path, dataset_file, multiple_files=len(dataset_files) > 1),
                    fps=1.0 / float(env.unwrapped.step_dt),
                )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
