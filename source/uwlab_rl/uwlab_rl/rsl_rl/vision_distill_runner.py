# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
import torch.nn as nn
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
