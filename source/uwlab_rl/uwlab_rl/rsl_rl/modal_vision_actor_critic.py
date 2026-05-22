# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import GSDENoiseDistribution
from rsl_rl.networks import EmpiricalNormalization, MLP

from .modal_encoders import SSIStyleModalEncoder
from .vision_actor_critic import _make_hidden_mlp


def _as_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(cfg)


class ModalVisionActorCritic(nn.Module):
    """Modal actor with depth/bbox/trajectory encoder and privileged critic."""

    is_recurrent: bool = False
    supports_flat_export: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        modal_encoder: dict[str, Any] | None = None,
        use_joint_pos: bool = False,
        proprio_mlp: dict[str, Any] | None = None,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 128, 64),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "log",
        state_dependent_std: bool = False,
        actor_obs_normalization: bool = False,
        proprio_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        policy_group_name: str = "policy",
        depth_term_name: str = "depth_map",
        bboxes_term_name: str = "bboxes",
        trajectory_term_name: str = "trajectory",
        joint_pos_term_name: str = "joint_pos",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ModalVisionActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        if state_dependent_std:
            raise ValueError("ModalVisionActorCritic does not support state_dependent_std.")

        super().__init__()
        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.policy_group_name = policy_group_name
        self.depth_term_name = depth_term_name
        self.bboxes_term_name = bboxes_term_name
        self.trajectory_term_name = trajectory_term_name
        self.joint_pos_term_name = joint_pos_term_name
        self.use_joint_pos = use_joint_pos
        self.noise_std_type = noise_std_type
        self.actor_obs_normalization = actor_obs_normalization
        self.proprio_obs_normalization = proprio_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = nn.Identity()

        policy_obs = self._get_policy_terms(obs)
        self._prepare_depth(policy_obs[self.depth_term_name])
        self._prepare_bboxes(policy_obs[self.bboxes_term_name])
        self._prepare_trajectory(policy_obs[self.trajectory_term_name])

        self.modal_encoder = SSIStyleModalEncoder(**_as_plain_dict(modal_encoder))
        actor_input_dim = self.modal_encoder.output_dim

        self.proprio_normalizer: nn.Module = nn.Identity()
        self.proprio_encoder: nn.Module | None = None
        if self.use_joint_pos:
            proprio_cfg = _as_plain_dict(proprio_mlp)
            joint_dim = self._prepare_joint_pos(policy_obs[self.joint_pos_term_name]).shape[-1]
            self.proprio_normalizer = EmpiricalNormalization(joint_dim) if proprio_obs_normalization else nn.Identity()
            proprio_hidden_dims = list(proprio_cfg.get("hidden_dims", [64, 64]))
            proprio_feature_dim = int(proprio_cfg.get("feature_dim", 32))
            self.proprio_encoder = MLP(joint_dim, proprio_feature_dim, proprio_hidden_dims, activation)
            actor_input_dim += proprio_feature_dim

        actor_hidden_dims = list(actor_hidden_dims)
        if not actor_hidden_dims:
            raise ValueError("actor_hidden_dims must contain at least one hidden layer.")
        self.actor_features = _make_hidden_mlp(actor_input_dim, actor_hidden_dims, activation)
        self.actor_final = nn.Linear(actor_hidden_dims[-1], num_actions)
        print(f"Modal actor encoder: {self.modal_encoder}")
        print(f"Modal actor trunk: {self.actor_features} -> {self.actor_final}")

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            critic_term = obs[obs_group]
            if len(critic_term.shape) != 2:
                raise ValueError(f"Critic observation group {obs_group!r} must be 2D, got {tuple(critic_term.shape)}.")
            num_critic_obs += critic_term.shape[-1]
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        elif self.noise_std_type == "gsde":
            self.log_std = nn.Parameter(
                torch.ones(actor_hidden_dims[-1], num_actions) * torch.log(torch.tensor(init_noise_std))
            )
        else:
            raise ValueError(f"Unknown noise_std_type={self.noise_std_type!r}.")

        if self.noise_std_type == "gsde":
            self.distribution = GSDENoiseDistribution(action_dim=num_actions)
            self.distribution.sample_weights(self.log_std)
        else:
            self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _get_policy_terms(self, obs: TensorDict) -> TensorDict:
        policy_obs = obs[self.policy_group_name]
        if isinstance(policy_obs, TensorDict):
            return policy_obs
        if isinstance(policy_obs, dict):
            return TensorDict(policy_obs, batch_size=obs.batch_size)
        raise TypeError(f"Modal policy observation group must be a dict/TensorDict, got {type(policy_obs)}.")

    def _strip_single_history(self, tensor: torch.Tensor, expected_ndim: int, term_name: str) -> torch.Tensor:
        if tensor.ndim == expected_ndim + 1:
            if tensor.shape[1] != 1:
                raise ValueError(f"{term_name} expects history length 1, got shape {tuple(tensor.shape)}.")
            tensor = tensor[:, -1]
        if tensor.ndim != expected_ndim:
            raise ValueError(f"{term_name} expected {expected_ndim} dims after history handling, got {tuple(tensor.shape)}.")
        return tensor

    def _prepare_depth(self, depth_map: torch.Tensor) -> torch.Tensor:
        depth_map = self._strip_single_history(depth_map, 5, self.depth_term_name)
        return depth_map.to(dtype=torch.float32)

    def _prepare_bboxes(self, bboxes: torch.Tensor) -> torch.Tensor:
        bboxes = self._strip_single_history(bboxes, 4, self.bboxes_term_name)
        return bboxes.to(dtype=torch.float32)

    def _prepare_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        trajectory = self._strip_single_history(trajectory, 5, self.trajectory_term_name)
        return trajectory.to(dtype=torch.float32)

    def _prepare_joint_pos(self, joint_pos: torch.Tensor) -> torch.Tensor:
        if joint_pos.ndim > 2:
            joint_pos = joint_pos.flatten(start_dim=1)
        return joint_pos.to(dtype=torch.float32)

    def _actor_latent(self, obs: TensorDict) -> torch.Tensor:
        policy_obs = self._get_policy_terms(obs)
        depth_map = self._prepare_depth(policy_obs[self.depth_term_name])
        bboxes = self._prepare_bboxes(policy_obs[self.bboxes_term_name])
        trajectory = self._prepare_trajectory(policy_obs[self.trajectory_term_name])
        features = [self.modal_encoder(depth_map, bboxes, trajectory)]

        if self.use_joint_pos:
            if self.proprio_encoder is None:
                raise RuntimeError("use_joint_pos=True but proprio_encoder is not initialized.")
            joint_pos = self._prepare_joint_pos(policy_obs[self.joint_pos_term_name])
            joint_pos = self.proprio_normalizer(joint_pos)
            features.append(self.proprio_encoder(joint_pos))

        return self.actor_features(torch.cat(features, dim=-1))

    def _update_distribution(self, obs: TensorDict) -> None:
        latent = self._actor_latent(obs)
        mean = self.actor_final(latent)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
            self.distribution = Normal(mean, std)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
            self.distribution = Normal(mean, std)
        elif self.noise_std_type == "gsde":
            self.distribution.proba_distribution(mean, self.log_std, latent)
        else:
            raise ValueError(f"Unknown noise_std_type={self.noise_std_type!r}.")

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.actor_final(self._actor_latent(obs))

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group].flatten(start_dim=1) for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.use_joint_pos and self.proprio_obs_normalization:
            policy_obs = self._get_policy_terms(obs)
            joint_pos = self._prepare_joint_pos(policy_obs[self.joint_pos_term_name])
            self.proprio_normalizer.update(joint_pos)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
