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
from rsl_rl.utils import resolve_nn_activation


def _make_hidden_mlp(input_dim: int, hidden_dims: list[int] | tuple[int, ...], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(resolve_nn_activation(activation))
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class _ResNet18Encoder(nn.Module):
    def __init__(self, feature_dim: int, pretrained: bool = False, imagenet_normalization: bool = False):
        super().__init__()
        from torchvision.models import resnet18

        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except (ImportError, TypeError):
            backbone = resnet18(pretrained=pretrained)

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(nn.Linear(512, feature_dim), nn.ELU())
        self.imagenet_normalization = imagenet_normalization
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        scale_from_uint8 = images.dtype == torch.uint8
        images = images.to(dtype=torch.float32)
        if scale_from_uint8:
            images = images / 255.0
        images = images.clamp(0.0, 1.0)
        if self.imagenet_normalization:
            images = (images - self.image_mean) / self.image_std
        return self.projection(self.backbone(images))


class VisionActorCritic(nn.Module):
    """Vision actor with a privileged MLP critic for ARX5 online distillation."""

    is_recurrent: bool = False
    supports_flat_export: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        vision_encoder: dict[str, Any] | None = None,
        proprio_mlp: dict[str, Any] | None = None,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "log",
        state_dependent_std: bool = False,
        actor_obs_normalization: bool = False,
        proprio_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        policy_group_name: str = "policy",
        joint_pos_term_name: str = "joint_pos",
        external_rgb_term_name: str = "external_rgb",
        wrist_rgb_term_name: str = "wrist_rgb",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "VisionActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        if state_dependent_std:
            raise ValueError("VisionActorCritic does not support state_dependent_std in v1.")

        super().__init__()
        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.policy_group_name = policy_group_name
        self.joint_pos_term_name = joint_pos_term_name
        self.external_rgb_term_name = external_rgb_term_name
        self.wrist_rgb_term_name = wrist_rgb_term_name
        self.noise_std_type = noise_std_type
        self.actor_obs_normalization = actor_obs_normalization
        self.proprio_obs_normalization = proprio_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = nn.Identity()

        vision_encoder = vision_encoder or {}
        proprio_mlp = proprio_mlp or {}
        if vision_encoder.get("name", "resnet18") != "resnet18":
            raise ValueError(f"Unsupported vision encoder: {vision_encoder.get('name')!r}.")

        policy_obs = self._get_policy_terms(obs)
        joint_dim = self._prepare_joint_pos(policy_obs[self.joint_pos_term_name]).shape[-1]
        self._prepare_image(policy_obs[self.external_rgb_term_name])
        self._prepare_image(policy_obs[self.wrist_rgb_term_name])

        vision_feature_dim = int(vision_encoder.get("feature_dim", 128))
        share_camera_encoder = bool(vision_encoder.get("share_camera_encoder", False))
        pretrained = bool(vision_encoder.get("pretrained", False))
        imagenet_normalization = bool(vision_encoder.get("imagenet_normalization", False))
        self.external_encoder = _ResNet18Encoder(vision_feature_dim, pretrained, imagenet_normalization)
        self.wrist_encoder = self.external_encoder if share_camera_encoder else _ResNet18Encoder(
            vision_feature_dim, pretrained, imagenet_normalization
        )

        proprio_hidden_dims = list(proprio_mlp.get("hidden_dims", [128, 128]))
        proprio_feature_dim = int(proprio_mlp.get("feature_dim", 64))
        self.proprio_normalizer = EmpiricalNormalization(joint_dim) if proprio_obs_normalization else nn.Identity()
        self.proprio_encoder = MLP(joint_dim, proprio_feature_dim, proprio_hidden_dims, activation)

        actor_input_dim = 2 * vision_feature_dim + proprio_feature_dim
        actor_hidden_dims = list(actor_hidden_dims)
        if not actor_hidden_dims:
            raise ValueError("actor_hidden_dims must contain at least one hidden layer.")
        self.actor_features = _make_hidden_mlp(actor_input_dim, actor_hidden_dims, activation)
        self.actor_final = nn.Linear(actor_hidden_dims[-1], num_actions)
        print(f"Vision actor encoders: external={self.external_encoder}, wrist={self.wrist_encoder}")
        print(f"Vision actor trunk: {self.actor_features} -> {self.actor_final}")

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            critic_term = obs[obs_group]
            if len(critic_term.shape) != 2:
                raise ValueError(f"Critic observation group {obs_group!r} must be 2D, got {tuple(critic_term.shape)}.")
            num_critic_obs += critic_term.shape[-1]
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")
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
        raise TypeError(f"Vision policy observation group must be a dict/TensorDict, got {type(policy_obs)}.")

    def _prepare_joint_pos(self, joint_pos: torch.Tensor) -> torch.Tensor:
        if joint_pos.ndim > 2:
            joint_pos = joint_pos.flatten(start_dim=1)
        return joint_pos.to(dtype=torch.float32)

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 5:
            if image.shape[1] != 1:
                raise ValueError(f"VisionActorCritic v1 expects image history length 1, got shape {tuple(image.shape)}.")
            image = image[:, -1]
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor shaped (N, C, H, W), got {tuple(image.shape)}.")
        return image.to(dtype=torch.float32)

    def _actor_latent(self, obs: TensorDict) -> torch.Tensor:
        policy_obs = self._get_policy_terms(obs)
        joint_pos = self._prepare_joint_pos(policy_obs[self.joint_pos_term_name])
        joint_pos = self.proprio_normalizer(joint_pos)
        proprio_feature = self.proprio_encoder(joint_pos)

        external_rgb = self._prepare_image(policy_obs[self.external_rgb_term_name])
        wrist_rgb = self._prepare_image(policy_obs[self.wrist_rgb_term_name])
        external_feature = self.external_encoder(external_rgb)
        wrist_feature = self.wrist_encoder(wrist_rgb)

        fused = torch.cat([external_feature, wrist_feature, proprio_feature], dim=-1)
        return self.actor_features(fused)

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
        if self.proprio_obs_normalization:
            policy_obs = self._get_policy_terms(obs)
            joint_pos = self._prepare_joint_pos(policy_obs[self.joint_pos_term_name])
            self.proprio_normalizer.update(joint_pos)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
