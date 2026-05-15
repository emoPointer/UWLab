# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ARX5 vision-policy RL configuration.

This file keeps the existing state-policy environment intact and adds a
vision-policy variant for online PPO distillation from the frozen state policy.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.config.arx5.actions import Arx5OSCTrainAction

from ... import mdp as task_mdp
from .rl_state_cfg import (
    Arx5RlStateCfg,
    DeployRlStateSceneCfg,
    ObservationsCfg,
    TrainEvalEventCfg,
    TrainEventCfg,
)


VISION_TABLE_POSE = (0.0, 0.0, 0.799375, 1.0, 0.0, 0.0, 0.0)
VISION_BACKDROP_TABLE_RELATIVE_POSES = (
    (-1.1, 0.0, -0.280375, 1.0, 0.0, 0.0, 0.0),
    (-0.05, 0.8, -0.280375, 0.707, 0.0, 0.0, -0.707),
    (-0.05, -0.8, -0.280375, 0.707, 0.0, 0.0, -0.707),
)
VISION_BACKDROP_ASSET_NAMES = (
    "curtain_back",
    "curtain_left",
    "curtain_right",
)
VISION_EXTERNAL_CAMERA_TABLE_RELATIVE_POSE = (0.517, 0.327, 0.589, 0.3604, 0.2030, 0.5000, 0.7609)
VISION_RECEPTIVE_OBJECT_POSE = (-0.30, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0)
VISION_WORKSPACE_X_RANGE = (-0.4, -0.2)
VISION_WORKSPACE_Y_RANGE = (-0.3, -0.1)


@configclass
class VisionSceneCfg(DeployRlStateSceneCfg):
    """Deploy-style scene plus cameras used by the vision student."""

    pass


@configclass
class VisionTrainEventCfg(TrainEventCfg):
    """Vision training events: reset manager stats plus visual domain randomization."""

    randomize_backdrop_visuals = EventTerm(
        func=task_mdp.randomize_backdrop_visuals,
        mode="reset",
        params={
            "table_cfg": SceneEntityCfg("table"),
            "table_pose": VISION_TABLE_POSE,
            "backdrop_asset_names": VISION_BACKDROP_ASSET_NAMES,
            "backdrop_table_relative_poses": VISION_BACKDROP_TABLE_RELATIVE_POSES,
            "backdrop_position_jitter_m": 0.02,
            "backdrop_color_range": ((0.2, 0.2, 0.2), (1.0, 1.0, 1.0)),
            "external_camera_table_relative_pose": VISION_EXTERNAL_CAMERA_TABLE_RELATIVE_POSE,
        },
    )

    sync_task_pair_visuals_to_sim = EventTerm(
        func=task_mdp.sync_task_pair_visuals_to_sim,
        mode="reset",
        params={
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    randomize_sky_light = EventTerm(
        func=task_mdp.randomize_dome_light,
        mode="reset",
        params={
            "light_path": "/World/skyLight",
            "intensity_range": (800.0, 3500.0),
            "rotation_range": (0.0, 360.0),
            "pitch_range": (-10.0, 10.0),
            "roll_range": (-5.0, 5.0),
        },
    )


@configclass
class VisionEvalEventCfg(TrainEvalEventCfg):
    """Vision eval events with corrected visual table/backdrop placement."""

    sync_visual_table_and_backdrop = EventTerm(
        func=task_mdp.randomize_backdrop_visuals,
        mode="reset",
        params={
            "table_cfg": SceneEntityCfg("table"),
            "table_pose": VISION_TABLE_POSE,
            "backdrop_asset_names": VISION_BACKDROP_ASSET_NAMES,
            "backdrop_table_relative_poses": VISION_BACKDROP_TABLE_RELATIVE_POSES,
            "backdrop_position_jitter_m": 0.0,
            "external_camera_table_relative_pose": VISION_EXTERNAL_CAMERA_TABLE_RELATIVE_POSE,
        },
    )

    sync_task_pair_visuals_to_sim = EventTerm(
        func=task_mdp.sync_task_pair_visuals_to_sim,
        mode="reset",
        params={
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
        },
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

    scene: VisionSceneCfg = VisionSceneCfg(num_envs=128, env_spacing=3.0)
    observations: VisionObservationsCfg = VisionObservationsCfg()
    events: VisionTrainEventCfg = VisionTrainEventCfg()
    actions: Arx5OSCTrainAction = Arx5OSCTrainAction()


@configclass
class Arx5OSCVisionPlayCfg(Arx5OSCVisionTrainCfg):
    """Vision student play/evaluation config."""

    events: VisionEvalEventCfg = VisionEvalEventCfg()
