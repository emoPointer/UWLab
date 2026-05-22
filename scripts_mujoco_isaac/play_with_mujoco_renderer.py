#!/usr/bin/env python3
"""Run an Isaac policy with PhysX physics and mirrored MuJoCo rendering."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RSL_RL_DIR = SCRIPT_DIR.parent / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_DIR))

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint_path", nargs="?", default=None, help="RSL-RL checkpoint path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac envs. MuJoCo mirrors env 0.")
parser.add_argument("--task", type=str, default=None, help="IsaacLab task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many policy steps; 0 means run until app closes.")
parser.add_argument("--stop_on_done", action="store_true", default=False, help="Stop after env 0 first resets/dones.")
parser.add_argument(
    "--mujoco_xml",
    type=str,
    default="mujoco_arx5/models/arx5_robosuite_tabletop_dynamic.xml",
    help="MuJoCo render-only scene XML.",
)
parser.add_argument("--mujoco_camera", type=str, default="external_camera", help="MuJoCo camera to render.")
parser.add_argument("--mujoco_video_path", type=str, default="videos/mujoco_isaac_bridge/play.mp4")
parser.add_argument("--mujoco_video_width", type=int, default=640)
parser.add_argument("--mujoco_video_height", type=int, default=480)
parser.add_argument("--mujoco_insertive_body", type=str, default="insertive_cube")
parser.add_argument("--mujoco_receptive_body", type=str, default="receptive_cube")
parser.add_argument("--record_mujoco_video", action="store_true", default=False)
parser.add_argument("--randomize_mujoco_light_angles", action="store_true", default=False)
parser.add_argument("--mujoco_light_yaw_range", type=float, nargs=2, default=(0.0, 360.0))
parser.add_argument("--mujoco_light_elevation_range", type=float, nargs=2, default=(35.0, 75.0))
parser.add_argument("--mujoco_light_distance_range", type=float, nargs=2, default=(2.0, 3.0))
parser.add_argument("--mujoco_light_target", type=float, nargs=3, default=(-0.3, -0.2, 0.84))
parser.add_argument(
    "--mujoco_policy_images",
    action="store_true",
    default=False,
    help="Use MuJoCo-rendered external/wrist camera frames as structured vision-policy image observations.",
)
parser.add_argument("--mujoco_external_camera", type=str, default="external_camera")
parser.add_argument("--mujoco_wrist_camera", type=str, default="wrist_camera")
parser.add_argument("--print_actor_output", action="store_true", default=False)
parser.add_argument("--print_actor_output_interval", type=int, default=20)
parser.add_argument("--print_mirror_debug", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Vision tasks keep Isaac camera sensors alive so their authored world poses can
# drive the MuJoCo camera pose, even when policy RGB comes from MuJoCo.
args_cli.enable_cameras = bool(args_cli.task is not None and "Vision" in args_cli.task)
if args_cli.checkpoint is None:
    args_cli.checkpoint = args_cli.checkpoint_path
if args_cli.checkpoint is None:
    parser.error("checkpoint is required")
delattr(args_cli, "checkpoint_path")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn.functional as F
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from mujoco_arx5.isaac_render_bridge import IsaacMuJoCoRenderConfig, IsaacMuJoCoRenderSession
from play_checkpoint_utils import load_runner_checkpoint_for_play
from uwlab_rl.rsl_rl.vision_distill_runner import VisionDistillOnPolicyRunner
from uwlab_tasks.utils.hydra import hydra_task_config


def _zero_policy_image_observation(env, output_size: tuple[int, int] = (128, 128), channels: int = 3) -> torch.Tensor:
    return torch.zeros(env.num_envs, channels, output_size[0], output_size[1], device=env.device)


def _configure_mujoco_policy_image_env(env_cfg) -> None:
    """Keep structured vision obs, but avoid reading Isaac camera RGB tensors."""
    policy_obs = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if policy_obs is None:
        raise ValueError("--mujoco_policy_images requires an env with structured vision policy observations.")
    for term_name in ("external_rgb", "wrist_rgb"):
        term_cfg = getattr(policy_obs, term_name, None)
        if term_cfg is None:
            raise ValueError(f"--mujoco_policy_images requires policy observation term {term_name!r}.")
        term_cfg.func = _zero_policy_image_observation
        term_cfg.params = {"output_size": (128, 128), "channels": 3}

    for sensor_name in ("external_camera", "wrist_camera"):
        sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
        if sensor_cfg is None:
            raise ValueError(f"--mujoco_policy_images requires scene sensor {sensor_name!r} for camera pose sync.")
        sensor_cfg.update_latest_camera_pose = True


def _preprocess_mujoco_policy_frame(
    frame,
    target: torch.Tensor,
    *,
    crop_size: int | tuple[int, int] | None,
    crop_top: int = 0,
    crop_left: int | None = None,
    crop_right: int = 0,
    output_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Match ``process_image_crop_resize`` for one MuJoCo RGB frame."""
    image = torch.as_tensor(frame).to(device=target.device)
    if image.ndim != 3:
        raise ValueError(f"Expected MuJoCo frame shaped (H, W, C), got {tuple(image.shape)}.")
    if image.shape[-1] < 3:
        raise ValueError(f"Expected MuJoCo RGB frame with at least 3 channels, got {image.shape[-1]}.")
    image = image[..., :3]

    height = image.shape[-3]
    width = image.shape[-2]
    if crop_size is not None:
        if isinstance(crop_size, int):
            crop_height = crop_width = crop_size
        else:
            crop_height, crop_width = crop_size
        if crop_left is None:
            crop_left = width - crop_right - crop_width
        crop_bottom = crop_top + crop_height
        crop_end = crop_left + crop_width
        if crop_top < 0 or crop_left < 0 or crop_bottom > height or crop_end > width:
            raise ValueError(
                "MuJoCo image crop is outside camera bounds: "
                f"image=({height}, {width}), crop_top={crop_top}, crop_left={crop_left}, "
                f"crop_size=({crop_height}, {crop_width}), crop_right={crop_right}."
            )
        image = image[crop_top:crop_bottom, crop_left:crop_end, :]

    image = image.to(dtype=torch.float32).div_(255.0).clamp_(0.0, 1.0)
    image = image.permute(2, 0, 1).unsqueeze(0)
    if image.shape[-2:] != output_size:
        image = F.interpolate(image, size=output_size, mode="bilinear", antialias=True)
    image = image.to(dtype=target.dtype)
    if target.ndim == 5:
        image = image.unsqueeze(1)
    elif target.ndim != 4:
        raise ValueError(f"Expected target policy image obs with 4 or 5 dims, got {tuple(target.shape)}.")
    return image.expand(target.shape[0], *image.shape[1:]).clone()


def _inject_mujoco_policy_images(obs, frames: dict[str, object], args) -> None:
    policy_obs = obs["policy"]
    external_target = policy_obs["external_rgb"]
    wrist_target = policy_obs["wrist_rgb"]
    policy_obs["external_rgb"] = _preprocess_mujoco_policy_frame(
        frames[args.mujoco_external_camera],
        external_target,
        crop_top=0,
        crop_left=None,
        crop_right=0,
        crop_size=400,
        output_size=(128, 128),
    )
    policy_obs["wrist_rgb"] = _preprocess_mujoco_policy_frame(
        frames[args.mujoco_wrist_camera],
        wrist_target,
        crop_size=None,
        output_size=(128, 128),
    )


def _mujoco_camera_list(args) -> tuple[str, ...]:
    cameras = [args.mujoco_external_camera, args.mujoco_wrist_camera]
    if args.record_mujoco_video and args.mujoco_camera not in cameras:
        cameras.append(args.mujoco_camera)
    return tuple(dict.fromkeys(cameras))


def _derive_mujoco_video_path(base_path: str, camera: str, primary_camera: str) -> str:
    base = Path(base_path)
    suffix = base.suffix or ".mp4"
    stem = base.stem
    primary_marker = f"_{primary_camera}"
    if stem.endswith(primary_marker):
        stem = stem[: -len(primary_marker)]
    elif stem == primary_camera:
        stem = "mujoco"
    return str(base.with_name(f"{stem}_{camera}{suffix}"))


def _make_mujoco_video_recorders(render_session, args, *, fps: float) -> tuple[dict[str, object], dict[str, str]]:
    if not args.record_mujoco_video:
        return {}, {}
    if not args.mujoco_policy_images:
        recorder = render_session.make_video_recorder(fps=fps)
        return ({args.mujoco_camera: recorder} if recorder is not None else {}), {
            args.mujoco_camera: args.mujoco_video_path
        }

    video_paths = {}
    recorders = {}
    for camera in _mujoco_camera_list(args):
        if camera == args.mujoco_camera:
            path = args.mujoco_video_path
        else:
            path = _derive_mujoco_video_path(args.mujoco_video_path, camera, args.mujoco_camera)
        recorder = render_session.make_video_recorder(path=path, fps=fps)
        if recorder is not None:
            recorders[camera] = recorder
            video_paths[camera] = path
    return recorders, video_paths


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    if args_cli.mujoco_policy_images:
        _configure_mujoco_policy_image_env(env_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "VisionDistillOnPolicyRunner":
        runner = VisionDistillOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    load_runner_checkpoint_for_play(runner, resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))

    render_config = IsaacMuJoCoRenderConfig(
        mujoco_xml=args_cli.mujoco_xml,
        mujoco_camera=args_cli.mujoco_camera,
        mujoco_video_path=args_cli.mujoco_video_path,
        mujoco_video_width=args_cli.mujoco_video_width,
        mujoco_video_height=args_cli.mujoco_video_height,
        mujoco_insertive_body=args_cli.mujoco_insertive_body,
        mujoco_receptive_body=args_cli.mujoco_receptive_body,
        randomize_light_angles=args_cli.randomize_mujoco_light_angles,
        mujoco_light_yaw_range_deg=tuple(args_cli.mujoco_light_yaw_range),
        mujoco_light_elevation_range_deg=tuple(args_cli.mujoco_light_elevation_range),
        mujoco_light_distance_range=tuple(args_cli.mujoco_light_distance_range),
        mujoco_light_target=tuple(args_cli.mujoco_light_target),
        mujoco_light_seed=args_cli.seed,
        sync_camera_poses=args_cli.mujoco_policy_images,
        record_video=args_cli.record_mujoco_video,
    )
    render_session = IsaacMuJoCoRenderSession(env, render_config)
    recorder_fps = max(1.0, 1.0 / env.unwrapped.step_dt)
    video_recorders, video_paths = _make_mujoco_video_recorders(render_session, args_cli, fps=recorder_fps)

    obs = env.get_observations()
    if args_cli.mujoco_policy_images and env.num_envs != 1:
        print(
            "[WARN] --mujoco_policy_images mirrors only Isaac env 0; the same MuJoCo images will be used for all envs."
        )
    step_count = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                if args_cli.mujoco_policy_images:
                    frames = render_session.render_frames(_mujoco_camera_list(args_cli), step=step_count)
                    _inject_mujoco_policy_images(obs, frames, args_cli)
                    for camera, recorder in video_recorders.items():
                        recorder.append(frames[camera])
                elif video_recorders:
                    frame = render_session.render_frame(step=step_count)
                    for recorder in video_recorders.values():
                        recorder.append(frame)
                if args_cli.print_mirror_debug and step_count % 20 == 0:
                    print(f"[mujoco-render] step={step_count}")

                actions = policy(obs)
                if args_cli.print_actor_output and step_count % args_cli.print_actor_output_interval == 0:
                    actions_cpu = actions.detach().cpu()
                    env0_action = actions_cpu[0].tolist()
                    print(
                        "[actor output] "
                        f"step={step_count} env0={[round(value, 6) for value in env0_action]} "
                        f"min={actions_cpu.min().item():.6f} max={actions_cpu.max().item():.6f} "
                        f"mean={actions_cpu.mean().item():.6f}",
                        flush=True,
                    )
                obs, _, dones, _ = env.step(actions)
                if policy_nn is not None:
                    policy_nn.reset(dones)
                if args_cli.randomize_mujoco_light_angles and bool(dones[0].item()):
                    render_session.randomize_light_angles()
                step_count += 1

                if args_cli.stop_on_done and bool(dones[0].item()):
                    break
                if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                    break

            sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        for camera, recorder in video_recorders.items():
            recorder.close()
            print(f"[INFO] Saved MuJoCo-rendered video ({camera}): {video_paths[camera]}")
        render_session.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
