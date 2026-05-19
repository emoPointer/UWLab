# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniReset environments for ARX5 robot."""

import gymnasium as gym

from . import agents

# ============================================================================
# Partial assemblies (robot-independent, shared with UR5e)
# ============================================================================
# Reuse the UR5e partial assemblies env — it has no robot, only objects.
# Registered in ur5e_robotiq_2f85/__init__.py as "OmniReset-PartialAssemblies-v0"

# ============================================================================
# Grasp sampling
# ============================================================================
gym.register(
    id="OmniReset-Arx5-GraspSampling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.grasp_sampling_cfg:Arx5GraspSamplingCfg"},
    disable_env_checker=True,
)

# ============================================================================
# Reset states
# ============================================================================
gym.register(
    id="OmniReset-Arx5-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-Arx5-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-Arx5-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-Arx5-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectPartiallyAssembledEEAnywhereResetStatesCfg"
    },
)

gym.register(
    id="OmniReset-Arx5-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectPartiallyAssembledEEGraspedResetStatesCfg"
    },
)

# ============================================================================
# System identification (CMA-ES)
# ============================================================================
gym.register(
    id="OmniReset-Arx5-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sysid_cfg:SysidEnvCfg"},
)

# ============================================================================
# RL training (Stage 1: implicit actuator, no sysid DR)
# ============================================================================
gym.register(
    id="OmniReset-Arx5-OSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Arx5OSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Arx5-OSC-CubeStack-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Arx5OSCCubeStackTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# ============================================================================
# RL vision distillation (student vision actor, privileged critic, state teacher)
# ============================================================================
gym.register(
    id="OmniReset-Arx5-OSC-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_vision_cfg:Arx5OSCVisionTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_vision_cfg:VisionDistill_PPORunnerCfg",
    },
)

# ============================================================================
# RL finetune (Stage 2: explicit actuator, sysid DR curriculum)
# ============================================================================
gym.register(
    id="OmniReset-Arx5-OSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Arx5OSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Finetune_PPORunnerCfg",
    },
)

# ============================================================================
# Eval / Play
# ============================================================================
gym.register(
    id="OmniReset-Arx5-OSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Arx5OSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Arx5-OSC-Vision-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_vision_cfg:Arx5OSCVisionPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_vision_cfg:VisionDistill_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Arx5-OSC-Vision-Deploy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_vision_cfg:Arx5OSCVisionDeployPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_vision_cfg:VisionDistill_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Arx5-OSC-State-Deploy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Arx5OSCDeployEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Arx5-OSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Arx5OSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)
