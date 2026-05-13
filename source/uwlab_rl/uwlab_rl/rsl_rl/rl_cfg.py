# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # noqa: F401


@configclass
class BehaviorCloningCfg:
    experts_path: list[str] = MISSING  # type: ignore
    """The path to the expert data."""

    experts_loader: callable = "torch.jit.load"
    """The function to construct the expert. Default is None, for which is loaded in the same way student is loaded."""

    experts_env_mapping_func: callable = None
    """The function to map the expert to env_ids. Default is None, for which is mapped to all env_ids"""

    experts_observation_group_cfg: str | None = None
    """The observation group of the expert which may be different from student"""

    experts_observation_func: callable = None
    """The function that returns expert observation data, default is None, same as student observation."""

    experts_action_group_cfg: str | None = None
    """The action group of the expert which may be different from student"""

    learn_std: bool = False
    """Whether to learn the standard deviation of the expert policy."""

    cloning_loss_coeff: float = MISSING  # type: ignore
    """The coefficient for the cloning loss."""

    loss_decay: float = 1.0
    """The decay for the cloning loss coefficient. default to 1, no decay."""


@configclass
class OffPolicyAlgorithmCfg:
    """Configuration for the off-policy algorithm."""

    update_frequencies: float = 1
    """The frequency to update relative to online update."""

    batch_size: int | None = None
    """The batch size for the offline algorithm update, default to None, same of online size."""

    num_learning_epochs: int | None = None
    """The number of learning epochs for the offline algorithm update."""

    behavior_cloning_cfg: BehaviorCloningCfg | None = None
    """The configuration for the offline behavior cloning(dagger)."""


@configclass
class RslRlFancyActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the fancy actor-critic networks."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation."""

    noise_std_type: Literal["scalar", "log", "gsde"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""


@configclass
class RslRlFancyPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    behavior_cloning_cfg: BehaviorCloningCfg | None = None
    """The configuration for the online behavior cloning."""

    offline_algorithm_cfg: OffPolicyAlgorithmCfg | None = None
    """The configuration for the offline algorithms."""


@configclass
class RslRlVisionEncoderCfg:
    """Configuration for the visual encoders used by the vision actor."""

    name: Literal["resnet18"] = "resnet18"
    """Vision encoder backbone."""

    pretrained: bool = False
    """Whether to request pretrained torchvision weights."""

    share_camera_encoder: bool = False
    """Whether external and wrist cameras share one encoder."""

    feature_dim: int = 128
    """Feature dimension after each camera encoder projection."""

    imagenet_normalization: bool = False
    """Apply ImageNet normalization before the ResNet backbone."""


@configclass
class RslRlProprioEncoderCfg:
    """Configuration for the proprioceptive MLP used by the vision actor."""

    hidden_dims: list[int] = [128, 128]
    """Hidden dimensions for joint-position encoding."""

    feature_dim: int = 64
    """Output feature dimension for joint-position encoding."""


@configclass
class RslRlVisionActorCriticCfg:
    """Configuration for the ARX5 vision actor and privileged critic."""

    class_name: str = "VisionActorCritic"
    """Policy class name used by the custom vision distillation runner."""

    vision_encoder: RslRlVisionEncoderCfg = RslRlVisionEncoderCfg()
    """Camera encoder configuration."""

    proprio_mlp: RslRlProprioEncoderCfg = RslRlProprioEncoderCfg()
    """Joint-position encoder configuration."""

    actor_hidden_dims: list[int] = [512, 256, 128]
    """Actor trunk hidden dimensions after concatenating encoded inputs."""

    critic_hidden_dims: list[int] = [512, 256, 128, 64]
    """Critic hidden dimensions for privileged observations."""

    activation: str = "elu"
    """Activation used by proprio, actor, and critic MLPs."""

    init_noise_std: float = 1.0
    """Initial action noise standard deviation."""

    noise_std_type: Literal["scalar", "log", "gsde"] = "log"
    """Action noise parameterization."""

    state_dependent_std: bool = False
    """Unsupported for the first vision actor version."""

    actor_obs_normalization: bool = False
    """Kept for RSL-RL compatibility. The vision actor normalizes joint positions separately."""

    proprio_obs_normalization: bool = True
    """Whether to normalize joint positions before the proprio MLP."""

    critic_obs_normalization: bool = True
    """Whether to normalize privileged critic observations."""

    policy_group_name: str = "policy"
    """Observation group containing student actor terms."""

    joint_pos_term_name: str = "joint_pos"
    """Student joint-position observation term name."""

    external_rgb_term_name: str = "external_rgb"
    """Student external camera observation term name."""

    wrist_rgb_term_name: str = "wrist_rgb"
    """Student wrist camera observation term name."""


@configclass
class RslRlActionDistillationCfg:
    """Online action distillation from a frozen state-policy teacher."""

    enabled: bool = True
    """Whether to add the teacher action regularization loss."""

    loss_type: Literal["mse"] = "mse"
    """Distillation loss type."""

    lambda_initial: float = 1.0
    """Initial distillation coefficient."""

    lambda_final: float = 0.05
    """Final distillation coefficient after the decay window."""

    decay_iterations: int = 8000
    """Number of PPO updates over which to linearly decay the coefficient."""


@configclass
class RslRlVisionDistillPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO plus online action distillation from a frozen state-policy teacher."""

    class_name: str = "VisionDistillPPO"
    """Algorithm class name used by the custom vision distillation runner."""

    teacher_checkpoint: str = ""
    """Checkpoint path for the frozen state-policy teacher."""

    teacher_obs_group: str = "teacher_policy"
    """Observation group that matches the teacher actor input."""

    teacher_actor_hidden_dims: list[int] = [512, 256, 128, 64]
    """Teacher actor hidden dimensions, matching the state-policy checkpoint."""

    teacher_critic_hidden_dims: list[int] = [512, 256, 128, 64]
    """Teacher critic hidden dimensions, needed to load the full checkpoint."""

    teacher_activation: str = "elu"
    """Teacher activation function."""

    teacher_init_noise_std: float = 1.0
    """Teacher initial noise std used to instantiate checkpoint-compatible parameters."""

    teacher_noise_std_type: Literal["scalar", "log", "gsde"] = "gsde"
    """Teacher action noise parameterization."""

    teacher_state_dependent_std: bool = False
    """Teacher state-dependent standard deviation flag."""

    teacher_actor_obs_normalization: bool = True
    """Whether the teacher actor checkpoint has actor observation normalization."""

    teacher_critic_obs_normalization: bool = True
    """Whether the teacher critic checkpoint has critic observation normalization."""

    distillation: RslRlActionDistillationCfg = RslRlActionDistillationCfg()
    """Distillation loss schedule."""
