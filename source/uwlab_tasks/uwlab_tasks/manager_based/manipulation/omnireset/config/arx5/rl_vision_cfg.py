# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ARX5 vision-policy RL configuration.

This file keeps the existing state-policy environment intact and adds a
vision-policy variant for online PPO distillation from the frozen state policy.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.config.arx5.actions import Arx5OSCTrainAction

from ... import mdp as task_mdp
from .rl_state_cfg import (
    ROBOSUITE_CAMERA_HEIGHT,
    ROBOSUITE_CAMERA_WIDTH,
    Arx5RlStateCfg,
    ObservationsCfg,
    RlStateSceneCfg,
    TrainEvalEventCfg,
    TrainEventCfg,
)


@configclass
class VisionSceneCfg(RlStateSceneCfg):
    """State scene plus the two cameras used by the vision student."""

    external_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Table/external_cam/Camera",
        update_period=0,
        height=ROBOSUITE_CAMERA_HEIGHT,
        width=ROBOSUITE_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=None,
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera/Camera",
        update_period=0,
        height=ROBOSUITE_CAMERA_HEIGHT,
        width=ROBOSUITE_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=None,
    )


@configclass
class VisionObservationsCfg:
    """Observation groups for vision-policy distillation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Student actor observations: joint positions and two RGB cameras."""

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        external_rgb = ObsTerm(
            func=task_mdp.process_image_crop_resize,
            params={
                "sensor_cfg": SceneEntityCfg("external_camera"),
                "data_type": "rgb",
                "crop_top": 0,
                "crop_left": None,
                "crop_right": 0,
                "crop_size": 400,
                "output_size": (128, 128),
                "normalize": True,
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image_crop_resize,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "crop_size": None,
                "output_size": (128, 128),
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False
            self.history_length = 1
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        """Privileged critic observations inherited from the state-policy task."""

        pass

    @configclass
    class TeacherPolicyCfg(ObservationsCfg.PolicyCfg):
        """Frozen state-policy teacher observations, matching the old actor exactly."""

        pass

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    teacher_policy: TeacherPolicyCfg = TeacherPolicyCfg()


@configclass
class Arx5OSCVisionTrainCfg(Arx5RlStateCfg):
    """Vision student training config for online PPO distillation."""

    scene: VisionSceneCfg = VisionSceneCfg(num_envs=128, env_spacing=1.5)
    observations: VisionObservationsCfg = VisionObservationsCfg()
    events: TrainEventCfg = TrainEventCfg()
    actions: Arx5OSCTrainAction = Arx5OSCTrainAction()


@configclass
class Arx5OSCVisionPlayCfg(Arx5OSCVisionTrainCfg):
    """Vision student play/evaluation config."""

    events: TrainEvalEventCfg = TrainEvalEventCfg()
