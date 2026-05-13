#!/usr/bin/env python3
"""Run an Isaac policy with PhysX physics and render mirrored frames in MuJoCo."""

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
parser.add_argument("--print_mirror_debug", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
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
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from mujoco_arx5.isaac_render_bridge import IsaacMuJoCoRenderConfig, IsaacMuJoCoRenderSession
from play_checkpoint_utils import load_runner_checkpoint_for_play
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

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
        record_video=args_cli.record_mujoco_video,
    )
    render_session = IsaacMuJoCoRenderSession(env, render_config)
    recorder = render_session.make_video_recorder(fps=max(1.0, 1.0 / env.unwrapped.step_dt))

    obs = env.get_observations()
    step_count = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                if recorder is not None:
                    recorder.append(render_session.render_frame(step=step_count))
                if args_cli.print_mirror_debug and step_count % 20 == 0:
                    print(f"[mujoco-render] step={step_count}")

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if policy_nn is not None:
                    policy_nn.reset(dones)
                step_count += 1

                if args_cli.stop_on_done and bool(dones[0].item()):
                    break
                if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                    break

            sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if recorder is not None:
            recorder.close()
            print(f"[INFO] Saved MuJoCo-rendered video: {args_cli.mujoco_video_path}")
        render_session.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
