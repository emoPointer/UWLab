#!/usr/bin/env python
"""Deploy a pi0.5 UWLab cube-stack policy through the openpi websocket server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_HYDRA_OVERRIDES = [
    "env.scene.insertive_object=cube",
    "env.scene.receptive_object=cube",
]
DEFAULT_PROMPT = "Place the green block on top of the red block."


parser = argparse.ArgumentParser(description="Run pi0.5 LoRA cube-stack policy in Isaac Lab.")
parser.add_argument("--task", type=str, default="OmniReset-Arx5-OSC-State-Deploy-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=3.0)
parser.add_argument("--policy-host", type=str, default="localhost")
parser.add_argument("--policy-port", type=int, default=8000)
parser.add_argument("--api-key", type=str, default=None)
parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
parser.add_argument("--max-steps", type=int, default=160)
parser.add_argument("--execute-horizon", type=int, default=10)
parser.add_argument("--append-zero-state-dims", type=int, default=1)
parser.add_argument("--table-camera-name", type=str, default="external_camera")
parser.add_argument("--wrist-camera-name", type=str, default="wrist_camera")
parser.add_argument("--ee-body-name", type=str, default="link6")
parser.add_argument("--action-scale", type=float, default=1.0)
parser.add_argument("--video-path", type=str, default="./videos/pi05_lora_deploy/cube_stack_pi05_lora_deploy.mp4")
parser.add_argument("--metrics-path", type=str, default="./videos/pi05_lora_deploy/cube_stack_pi05_lora_metrics.json")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + DEFAULT_HYDRA_OVERRIDES + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402
from PIL import Image  # noqa: E402

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
            "insertive_object_color": (0.0, 1.0, 0.0),
            "receptive_object_color": (1.0, 0.0, 0.0),
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


def _find_single_body_id(robot, body_name: str) -> int:
    body_ids, body_names = robot.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one body matching {body_name!r}, found {body_names}.")
    return int(body_ids[0])


def _camera_rgb(unwrapped_env, camera_name: str) -> np.ndarray:
    camera = unwrapped_env.scene.sensors.get(camera_name)
    if camera is None:
        raise RuntimeError(f"Missing camera sensor: {camera_name}")
    return camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8, copy=False)


def _resize_like_training(image_array: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    image_array = np.asarray(image_array)
    if np.issubdtype(image_array.dtype, np.floating):
        image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    if image_array.ndim != 3 or image_array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image with 3 channels, got shape {image_array.shape}")
    if image_array.shape[:2] == target_size:
        return image_array.astype(np.uint8, copy=False)
    image = Image.fromarray(image_array.astype(np.uint8, copy=False))
    return np.asarray(image.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS))


def _rotation_6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
    rot_matrix = matrix_from_quat(quat)
    return rot_matrix[:, :, :2].transpose(1, 2).reshape(quat.shape[0], 6)


def _build_state(unwrapped_env, ee_body_id: int, append_zero_state_dims: int) -> np.ndarray:
    robot = unwrapped_env.scene["robot"]
    eef_pos = robot.data.body_link_pos_w[0, ee_body_id].detach().cpu().numpy().astype(np.float32, copy=False)
    eef_rot_6d = (
        _rotation_6d_from_quat(robot.data.body_link_quat_w[0:1, ee_body_id])[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    parts = [eef_pos.reshape(-1), eef_rot_6d.reshape(-1)]
    if append_zero_state_dims:
        parts.append(np.zeros((append_zero_state_dims,), dtype=np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _build_policy_observation(unwrapped_env, ee_body_id: int) -> dict:
    return {
        "observation/image": _resize_like_training(_camera_rgb(unwrapped_env, args_cli.table_camera_name)),
        "observation/wrist_image": _resize_like_training(_camera_rgb(unwrapped_env, args_cli.wrist_camera_name)),
        "observation/state": _build_state(unwrapped_env, ee_body_id, args_cli.append_zero_state_dims),
        "prompt": args_cli.prompt,
    }


def _success_mask(unwrapped_env, dones: torch.Tensor) -> torch.Tensor:
    success = torch.zeros_like(dones, dtype=torch.bool)
    term_manager = unwrapped_env.termination_manager
    if "success" in term_manager.active_terms:
        success |= term_manager.get_term("success")
    return success


def _write_video(frames: list[np.ndarray], video_path: Path, fps: float) -> None:
    if not frames:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"[INFO] Saved deployment video: {video_path}", flush=True)


def _write_metrics(metrics: dict, metrics_path: Path | None) -> None:
    if metrics_path is None:
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[INFO] Saved deployment metrics: {metrics_path}", flush=True)


def _action_tensor(action: np.ndarray, device: torch.device | str) -> torch.Tensor:
    action = np.asarray(action, dtype=np.float32) * float(args_cli.action_scale)
    return torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)


def _as_bool(tensor: torch.Tensor, index: int = 0) -> bool:
    return bool(tensor[index].detach().cpu().item())


def _deploy_episode(env, policy_client, ee_body_id: int) -> dict:
    env.reset()
    step_dt = float(env.unwrapped.step_dt)
    fps = 1.0 / step_dt
    frames = [_camera_rgb(env.unwrapped, args_cli.table_camera_name)] if args_cli.video_path else []
    infer_ms: list[float] = []
    chunk_count = 0
    total_steps = 0
    success = False
    done = False

    print(
        f"[INFO] Running pi0.5 policy: max_steps={args_cli.max_steps} "
        f"execute_horizon={args_cli.execute_horizon} control_hz={fps:.3f}",
        flush=True,
    )

    while simulation_app.is_running() and total_steps < args_cli.max_steps and not done:
        observation = _build_policy_observation(env.unwrapped, ee_body_id)
        infer_start = time.time()
        result = policy_client.infer(observation)
        chunk_count += 1
        if "policy_timing" in result and "infer_ms" in result["policy_timing"]:
            infer_ms.append(float(result["policy_timing"]["infer_ms"]))
        else:
            infer_ms.append(1000.0 * (time.time() - infer_start))

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected action chunk [T, action_dim], got shape {actions.shape}")
        if actions.shape[1] != env.num_actions:
            raise ValueError(
                f"Policy action dim {actions.shape[1]} does not match env action dim {env.num_actions}."
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("Policy returned NaN or Inf actions.")
        execute_steps = min(args_cli.execute_horizon, actions.shape[0], args_cli.max_steps - total_steps)
        if execute_steps <= 0:
            break
        print(
            f"[INFO] chunk={chunk_count} shape={tuple(actions.shape)} "
            f"execute={execute_steps} infer_ms={infer_ms[-1]:.1f} "
            f"first_action={np.round(actions[0], 4).tolist()}",
            flush=True,
        )

        for chunk_step in range(execute_steps):
            if not simulation_app.is_running():
                break
            start_time = time.time()
            _, _, dones, _ = env.step(_action_tensor(actions[chunk_step], env.unwrapped.device))
            total_steps += 1
            dones = dones.to(dtype=torch.bool)
            success_mask = _success_mask(env.unwrapped, dones)
            success = success or _as_bool(success_mask)
            done = _as_bool(dones) or success
            if args_cli.video_path:
                frames.append(_camera_rgb(env.unwrapped, args_cli.table_camera_name))
            if args_cli.real_time:
                sleep_time = step_dt - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
            if done:
                break

    if args_cli.video_path:
        _write_video(frames, Path(args_cli.video_path).expanduser(), fps=fps)

    metrics = {
        "success": success,
        "done": done,
        "steps": total_steps,
        "chunks": chunk_count,
        "max_steps": args_cli.max_steps,
        "execute_horizon": args_cli.execute_horizon,
        "control_hz": fps,
        "prompt": args_cli.prompt,
        "append_zero_state_dims": args_cli.append_zero_state_dims,
        "mean_infer_ms": float(np.mean(infer_ms)) if infer_ms else None,
        "max_infer_ms": float(np.max(infer_ms)) if infer_ms else None,
    }
    _write_metrics(metrics, Path(args_cli.metrics_path).expanduser() if args_cli.metrics_path else None)
    return metrics


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg) -> None:
    if args_cli.num_envs != 1:
        raise ValueError("This deployment script currently expects --num_envs 1.")
    if args_cli.execute_horizon <= 0:
        raise ValueError("--execute-horizon must be positive.")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        _set_fixed_robot_workspace_reset(env_cfg)

    print(
        f"[INFO] Connecting to openpi policy server at {args_cli.policy_host}:{args_cli.policy_port}",
        flush=True,
    )
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        api_key=args_cli.api_key,
    )
    print(f"[INFO] Server metadata: {policy_client.get_server_metadata()}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    ee_body_id = _find_single_body_id(env.unwrapped.scene["robot"], args_cli.ee_body_name)

    try:
        metrics = _deploy_episode(env, policy_client, ee_body_id)
        print(
            f"[INFO] Deployment finished: success={metrics['success']} "
            f"steps={metrics['steps']} chunks={metrics['chunks']}",
            flush=True,
        )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
