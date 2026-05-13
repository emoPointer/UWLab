# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import torch.nn as nn
from collections import deque
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner

from .vision_actor_critic import VisionActorCritic
from .vision_distill_ppo import VisionDistillPPO


class _InferenceOnlyTeacher(nn.Module):
    """Placeholder teacher used only to construct a play runner."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return torch.zeros(obs.batch_size[0], self.num_actions, device=obs.device)


class VisionDistillOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that resolves UWLab vision-distillation policy/algorithm classes."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run PPO training and optionally upload training camera clips to wandb."""
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            camera_recordings = self._new_wandb_camera_recordings(it)
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if camera_recordings is not None:
                        self._record_wandb_camera_frames(camera_recordings)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                self._log_wandb_camera_videos(camera_recordings, it)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = self._safe_store_code_state()
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Log training stats and tolerate runners without RND bookkeeping."""
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = f"""{"#" * width}\n{header.center(width, " ")}\n\n"""
        log_string += (
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning """
            f"""{locs["learn_time"]:.3f}s)\n"""
            f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
        )
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
                f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
            )
        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} """
            f"""{time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs["it"] - locs["start_iter"] + 1) * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])))}\n"""
        )
        print(log_string)

    def _new_wandb_camera_recordings(self, iteration: int) -> dict[str, list] | None:
        interval = int(self.cfg.get("wandb_camera_video_interval", 0))
        if interval <= 0 or iteration % interval != 0:
            return None
        if self.log_dir is None or self.disable_logs or self.cfg.get("logger", "tensorboard").lower() != "wandb":
            return None
        camera_names = tuple(self.cfg.get("wandb_camera_video_camera_names", ["external_camera"]))
        return {name: [] for name in camera_names}

    def _record_wandb_camera_frames(self, camera_recordings: dict[str, list]) -> None:
        max_frames = int(self.cfg.get("wandb_camera_video_length", self.num_steps_per_env))
        env_index = int(self.cfg.get("wandb_camera_video_env_index", 0))
        for camera_name, frames in camera_recordings.items():
            if len(frames) >= max_frames:
                continue
            camera = self.env.unwrapped.scene.sensors.get(camera_name)
            if camera is None:
                continue
            rgb = camera.data.output["rgb"][env_index, ..., :3].detach().cpu().numpy()
            frames.append(rgb)

    def _log_wandb_camera_videos(self, camera_recordings: dict[str, list] | None, iteration: int) -> None:
        if not camera_recordings:
            return
        try:
            import numpy as np
            import wandb
        except ModuleNotFoundError:
            print("[WARN] wandb/numpy is not available; skipping camera video upload.")
            return
        if wandb.run is None:
            return
        fps = int(self.cfg.get("wandb_camera_video_fps", max(1.0, 1.0 / self.env.unwrapped.step_dt)))
        log_payload = {}
        for camera_name, frames in camera_recordings.items():
            if not frames:
                continue
            video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
            log_payload[f"train_camera/{camera_name}"] = wandb.Video(video, fps=fps, format="mp4")
        if log_payload:
            wandb.log(log_payload, step=iteration)

    def _safe_store_code_state(self) -> list[str]:
        try:
            from rsl_rl.utils import store_code_state

            return store_code_state(self.log_dir, self.git_status_repos)
        except Exception as exc:
            print(f"[WARN] Failed to store git diff for this run: {exc}")
            return []

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        self.alg_cfg.pop("optimizer", None)
        self.alg_cfg.pop("share_cnn_encoders", None)

        if self.cfg.get("empirical_normalization") is not None:
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        policy_class_name = self.policy_cfg.pop("class_name")
        policy_classes = {
            "ActorCritic": ActorCritic,
            "ActorCriticRecurrent": ActorCriticRecurrent,
            "VisionActorCritic": VisionActorCritic,
        }
        if policy_class_name not in policy_classes:
            raise ValueError(f"Unsupported policy class for VisionDistillOnPolicyRunner: {policy_class_name}")
        actor_critic = policy_classes[policy_class_name](
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class_name = self.alg_cfg.pop("class_name")
        if alg_class_name == "VisionDistillPPO":
            teacher_obs_group = self.alg_cfg.pop("teacher_obs_group", "teacher_policy")
            teacher = self._construct_teacher(obs, teacher_obs_group)
            alg = VisionDistillPPO(
                actor_critic,
                teacher=teacher,
                teacher_obs_group=teacher_obs_group,
                device=self.device,
                **self.alg_cfg,
                multi_gpu_cfg=self.multi_gpu_cfg,
            )
        elif alg_class_name == "PPO":
            alg = PPO(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        else:
            raise ValueError(f"Unsupported algorithm class for VisionDistillOnPolicyRunner: {alg_class_name}")

        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg

    def _construct_teacher(self, obs: TensorDict, teacher_obs_group: str) -> ActorCritic:
        teacher_checkpoint = self.alg_cfg.pop("teacher_checkpoint", "")
        if not teacher_checkpoint:
            if self.log_dir is None:
                print("[INFO] No teacher checkpoint configured; using inference-only placeholder teacher.")
                return _InferenceOnlyTeacher(self.env.num_actions).to(self.device)
            raise ValueError(
                "Vision distillation requires agent.algorithm.teacher_checkpoint=/path/to/state_policy_model.pt"
            )
        teacher_checkpoint = os.path.expanduser(teacher_checkpoint)
        if not os.path.exists(teacher_checkpoint):
            try:
                from isaaclab.utils.assets import retrieve_file_path

                teacher_checkpoint = retrieve_file_path(teacher_checkpoint)
            except Exception as exc:
                raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint}") from exc

        teacher_actor_hidden_dims = self.alg_cfg.pop("teacher_actor_hidden_dims", [512, 256, 128, 64])
        teacher_critic_hidden_dims = self.alg_cfg.pop("teacher_critic_hidden_dims", [512, 256, 128, 64])
        teacher_activation = self.alg_cfg.pop("teacher_activation", "elu")
        teacher_init_noise_std = self.alg_cfg.pop("teacher_init_noise_std", 1.0)
        teacher_noise_std_type = self.alg_cfg.pop("teacher_noise_std_type", "gsde")
        teacher_state_dependent_std = self.alg_cfg.pop("teacher_state_dependent_std", False)
        teacher_actor_obs_normalization = self.alg_cfg.pop("teacher_actor_obs_normalization", True)
        teacher_critic_obs_normalization = self.alg_cfg.pop("teacher_critic_obs_normalization", True)

        teacher_obs_groups = {
            "policy": [teacher_obs_group],
            "critic": self.cfg["obs_groups"].get("critic", ["critic"]),
        }
        teacher = ActorCritic(
            obs,
            teacher_obs_groups,
            self.env.num_actions,
            actor_obs_normalization=teacher_actor_obs_normalization,
            critic_obs_normalization=teacher_critic_obs_normalization,
            actor_hidden_dims=teacher_actor_hidden_dims,
            critic_hidden_dims=teacher_critic_hidden_dims,
            activation=teacher_activation,
            init_noise_std=teacher_init_noise_std,
            noise_std_type=teacher_noise_std_type,
            state_dependent_std=teacher_state_dependent_std,
        ).to(self.device)

        checkpoint = torch.load(teacher_checkpoint, weights_only=False, map_location=self.device)
        teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
        teacher.eval()
        print(f"[INFO] Loaded frozen teacher policy from: {teacher_checkpoint}")
        return teacher
