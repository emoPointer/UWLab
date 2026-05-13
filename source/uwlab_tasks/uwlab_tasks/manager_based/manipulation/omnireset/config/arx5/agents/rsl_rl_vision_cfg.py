# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from uwlab_rl.rsl_rl.rl_cfg import (
    RslRlActionDistillationCfg,
    RslRlProprioEncoderCfg,
    RslRlVisionActorCriticCfg,
    RslRlVisionDistillPpoAlgorithmCfg,
    RslRlVisionEncoderCfg,
)


@configclass
class VisionDistill_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Online PPO training for a vision student distilled from a state-policy teacher."""

    class_name = "VisionDistillOnPolicyRunner"
    num_steps_per_env = 16
    max_iterations = 40000
    save_interval = 100
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    resume = False
    experiment_name = "arx5_omnireset_vision_distill"

    policy = RslRlVisionActorCriticCfg(
        vision_encoder=RslRlVisionEncoderCfg(
            name="resnet18",
            pretrained=False,
            share_camera_encoder=False,
            feature_dim=128,
            imagenet_normalization=False,
        ),
        proprio_mlp=RslRlProprioEncoderCfg(hidden_dims=[128, 128], feature_dim=64),
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        proprio_obs_normalization=True,
        critic_obs_normalization=True,
    )

    algorithm = RslRlVisionDistillPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_checkpoint="",
        teacher_obs_group="teacher_policy",
        teacher_actor_hidden_dims=[512, 256, 128, 64],
        teacher_critic_hidden_dims=[512, 256, 128, 64],
        teacher_activation="elu",
        teacher_init_noise_std=1.0,
        teacher_noise_std_type="gsde",
        teacher_state_dependent_std=False,
        teacher_actor_obs_normalization=True,
        teacher_critic_obs_normalization=True,
        distillation=RslRlActionDistillationCfg(
            enabled=True,
            loss_type="mse",
            lambda_initial=1.0,
            lambda_final=0.05,
            decay_iterations=8000,
        ),
    )
