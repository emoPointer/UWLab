# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniReset RL state-based training configuration for ARX5.

Adapted from ur5e_robotiq_2f85/rl_state_cfg.py with the following changes:
    - Robot: IMPLICIT_ARX5 / EXPLICIT_ARX5
    - EE body: link6 (was wrist_3_link)
    - Gripper base body: link6 (was robotiq_base_link)
    - Arm joints: joint[1-6] (was shoulder.*/elbow.*/wrist.*)
    - Gripper joints: joint[78] (was finger_joint)
    - OSC: IsaacLab built-in OperationalSpaceController (was custom analytical-Jacobian OSC)
    - Scene: no ur5_metal_support (ARX5 uses different mounting)
    - Finetune sysid: adapted joint names for ARX5
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_ASSETS_EXT_DIR, UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.arx5 import EXPLICIT_ARX5, IMPLICIT_ARX5

from uwlab_tasks.manager_based.manipulation.omnireset.config.arx5.actions import (
    Arx5OSCEvalAction,
    Arx5OSCTrainAction,
)

from ... import mdp as task_mdp


ROBOSUITE_CAMERA_WIDTH = 640
ROBOSUITE_CAMERA_HEIGHT = 480

ROBOSUITE_WRIST_CAMERA_POS = (0.0, 0.0, 0.0)
ROBOSUITE_WRIST_CAMERA_ROT = (1.0, 0.0, 0.0, 0.0)
ROBOSUITE_ROBOT_BASE_POSE = (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0)
ROBOSUITE_TABLE_POSE = (0.0, 0.0, 0.799375, 1.0, 0.0, 0.0, 0.0)
ROBOSUITE_RECEPTIVE_OBJECT_POSE = (-0.30, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0)
ROBOSUITE_WORKSPACE_X_RANGE = (-0.4, -0.2)
ROBOSUITE_WORKSPACE_Y_RANGE = (-0.3, -0.1)


# ============================================================================
# Scene
# ============================================================================


@configclass
class RlStateSceneCfg(InteractiveSceneCfg):
    """Scene configuration for ARX5 RL state environment."""

    robot = IMPLICIT_ARX5.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Environment — no ur5_metal_support needed for ARX5
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.799375), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_ASSETS_EXT_DIR}/uwlab_assets/props/robosuite_table/table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class DeployRlStateSceneCfg(RlStateSceneCfg):
    """Deploy scene with project-style visual-only backdrop curtains."""

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

    # Match the robosuite-style table-relative backdrop layout, but keep the backdrop as thin visual-only panels.
    curtain_back = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainBack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.1, 0.0, 0.519), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.6, 2.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.72, 0.70)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    curtain_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, 0.8, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 2.1, 2.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.72, 0.70)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    curtain_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, -0.8, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 2.1, 2.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.72, 0.70)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )


# ============================================================================
# Events (Domain Randomization)
# ============================================================================


@configclass
class BaseEventCfg:
    """Shared events: material/mass randomization, gripper gains, scene reset."""

    # mode: startup
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (1.0, 2.0),
            "dynamic_friction_range": (0.9, 1.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.2, 0.6),
            "dynamic_friction_range": (0.15, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    table_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.2, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "mass_distribution_params": (0.02, 0.2),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_receptive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_table_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[78]"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # mode: reset
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    set_cube_stack_colors = EventTerm(
        func=task_mdp.set_task_object_visual_colors,
        mode="reset",
        params={
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "insertive_object_color": (0.0, 1.0, 0.0),
            "receptive_object_color": (1.0, 0.0, 0.0),
            "required_insertive_usd_substring": "InsertiveCube",
            "required_receptive_usd_substring": "ReceptiveCube",
        },
    )


@configclass
class TrainEventCfg(BaseEventCfg):
    """Training events: material/mass randomization + 4-path resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class CubeStackTrainEventCfg(TrainEventCfg):
    """Cube-stack state training events with deploy/vision workspace alignment."""

    set_cube_stack_colors = None

    align_cube_stack_scene_to_robosuite_table = EventTerm(
        func=task_mdp.align_deploy_scene_to_robosuite_table,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "table_cfg": SceneEntityCfg("table"),
            "training_robot_base_pose": ROBOSUITE_ROBOT_BASE_POSE,
            "robosuite_robot_base_pose": ROBOSUITE_ROBOT_BASE_POSE,
            "table_pose": ROBOSUITE_TABLE_POSE,
            "receptive_object_pose": ROBOSUITE_RECEPTIVE_OBJECT_POSE,
            "workspace_x_range": ROBOSUITE_WORKSPACE_X_RANGE,
            "workspace_y_range": ROBOSUITE_WORKSPACE_Y_RANGE,
            "sync_visuals": False,
        },
    )


@configclass
class TrainEvalEventCfg(BaseEventCfg):
    """Eval after Stage 1: no sysid, 1-path resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class DeployEvalEventCfg(TrainEvalEventCfg):
    """Deployment play events: reset robot and peg from data, keep peghole in front of the robot."""

    align_deploy_scene_to_robosuite_table = EventTerm(
        func=task_mdp.align_deploy_scene_to_robosuite_table,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "table_cfg": SceneEntityCfg("table"),
            "training_robot_base_pose": (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0),
            "robosuite_robot_base_pose": (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0),
            "table_pose": (0.0, 0.0, 0.799375, 1.0, 0.0, 0.0, 0.0),
            "receptive_object_pose": (-0.30, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0),
            "robot_xy_jitter_m": 0.02,
            "workspace_x_range": (-0.4, -0.2),
            "workspace_y_range": (-0.3, -0.1),
            "task_object_color_range": ((0.2, 0.2, 0.2), (1.0, 1.0, 1.0)),
            "backdrop_asset_names": (
                "curtain_back",
                "curtain_left",
                "curtain_right",
            ),
            "backdrop_table_relative_poses": (
                (-1.1, 0.0, -0.280375, 1.0, 0.0, 0.0, 0.0),
                (-0.05, 0.8, -0.280375, 0.707, 0.0, 0.0, -0.707),
                (-0.05, -0.8, -0.280375, 0.707, 0.0, 0.0, -0.707),
            ),
            "backdrop_position_jitter_m": 0.02,
            "backdrop_color_range": ((0.2, 0.2, 0.2), (1.0, 1.0, 1.0)),
            "external_camera_table_relative_pose": (0.517, 0.327, 0.589, 0.3604, 0.2030, 0.5000, 0.7609),
            "log_once": True,
            "log_every_reset": True,
        },
    )

    reject_initial_successful_resets = EventTerm(
        func=task_mdp.RejectInitialAssemblySuccessReset,
        mode="reset",
        params={
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "reset_event_name": "reset_from_reset_states",
            "align_event_name": "align_deploy_scene_to_robosuite_table",
            "max_resample_attempts": 20,
            "log_rejections": True,
        },
    )


@configclass
class FinetuneEventCfg(BaseEventCfg):
    """Finetune events: fixed robot reset, workspace object reset, curriculum-ramped sysid."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.FixedRobotWorkspaceTaskPairReset,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "table_cfg": SceneEntityCfg("table"),
            "robot_pose": (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0),
            "table_pose": (0.0, 0.0, 0.799375, 1.0, 0.0, 0.0, 0.0),
            "insertive_object_pose": (-0.30, -0.20, 0.87, 1.0, 0.0, 0.0, 0.0),
            "receptive_object_pose": (-0.30, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0),
            "workspace_x_range": (-0.4, -0.2),
            "workspace_y_range": (-0.3, -0.1),
            "insertive_workspace_x_range": (-0.4, -0.2),
            "insertive_workspace_y_range": (-0.3, -0.1),
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "log_every_reset": False,
            "sync_visuals": False,
        },
    )

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
            "actuator_name": "arm",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 1),
            "initial_scale_progress": 0.0,
        },
    )


@configclass
class FinetuneEvalEventCfg(BaseEventCfg):
    """Eval after Stage 2: fixed sysid + 1-path resets."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
            "actuator_name": "arm",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 1),
        },
    )

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


# ============================================================================
# Commands
# ============================================================================


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="robot0_base"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )


# ============================================================================
# Observations
# ============================================================================


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations (includes privileged info)."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        # privileged observations
        time_left = ObsTerm(func=task_mdp.time_left)

        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        robot_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        insertive_object_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )

        receptive_object_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )

        table_material_properties = ObsTerm(
            func=task_mdp.get_material_properties_compat,
            params={"asset_cfg": SceneEntityCfg("table"), "output_dim": 21},
        )

        robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

        insertive_object_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )

        receptive_object_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )

        table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

        robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_stiffness = ObsTerm(
            func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================================
# Rewards
# ============================================================================


@configclass
class RewardsCfg:

    # safety rewards
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"])},
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-100.0)

    # task rewards
    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    ee_asset_distance = RewTerm(
        func=task_mdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 1.0,
        },
    )

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.1, params={"std": 1.0})

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)


# ============================================================================
# Terminations
# ============================================================================


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)


@configclass
class DeployTerminationsCfg(TerminationsCfg):
    """Deployment rollouts end shortly after stable assembly success."""

    success = DoneTerm(
        func=task_mdp.consecutive_success_state_with_min_length,
        params={"num_consecutive_successes": 10, "min_episode_length": 10},
    )


# ============================================================================
# Curriculum (Stage 2 finetune only)
# ============================================================================


@configclass
class FinetuneCurriculumsCfg:
    """Finetune curriculum: ADR sysid ramp."""

    adr_sysid = CurrTerm(
        func=task_mdp.adr_sysid_curriculum,
        params={
            "event_term_names": ["randomize_arm_sysid"],
            "reset_event_name": "reset_from_reset_states",
            "success_threshold_up": 0.95,
            "success_threshold_down": 0.9,
            "delta": 0.01,
            "update_every_n_steps": 200,
            "initial_scale_progress": 0.0,
            "warmup_success_threshold": 0.95,
        },
    )


@configclass
class NoCurriculumsCfg:
    """No curriculum (Stage 1 training / eval)."""

    pass


# ============================================================================
# Object variants (shared with UR5e)
# ============================================================================


def make_insertive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.insertive_object": {
        "fbleg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": make_insertive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
        "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cupcake": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
    },
    "scene.receptive_object": {
        "fbtabletop": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd"
        ),
        "fbdrawerbox": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd"
        ),
        "peghole": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"),
        "plate": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
        "cube": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/ReceptiveCube/receptive_cube.usd"),
        "wall": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"),
    },
}


# ============================================================================
# Top-level environment configs
# ============================================================================


@configclass
class Arx5RlStateCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Arx5OSCTrainAction = Arx5OSCTrainAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 16.0
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


# Stage 1: Train (implicit actuator, no sysid DR)
@configclass
class Arx5OSCTrainCfg(Arx5RlStateCfg):
    events: TrainEventCfg = TrainEventCfg()
    actions: Arx5OSCTrainAction = Arx5OSCTrainAction()


@configclass
class Arx5OSCCubeStackTrainCfg(Arx5OSCTrainCfg):
    """State-policy cube-stack training with the same workspace alignment as vision training."""

    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=128, env_spacing=3.0)
    events: CubeStackTrainEventCfg = CubeStackTrainEventCfg()


# Stage 2: Finetune (explicit actuator, curriculum ramps sysid)
@configclass
class Arx5OSCFinetuneCfg(Arx5RlStateCfg):
    events: FinetuneEventCfg = FinetuneEventCfg()
    actions: Arx5OSCTrainAction = Arx5OSCTrainAction()
    curriculum: FinetuneCurriculumsCfg = FinetuneCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_ARX5.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Eval after Stage 1
@configclass
class Arx5OSCEvalCfg(Arx5RlStateCfg):
    events: TrainEvalEventCfg = TrainEvalEventCfg()
    actions: Arx5OSCTrainAction = Arx5OSCTrainAction()


@configclass
class Arx5OSCDeployEvalCfg(Arx5OSCEvalCfg):
    scene: DeployRlStateSceneCfg = DeployRlStateSceneCfg(num_envs=32, env_spacing=1.5)
    events: DeployEvalEventCfg = DeployEvalEventCfg()
    terminations: DeployTerminationsCfg = DeployTerminationsCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.45, -1.15, 1.35),
        lookat=(-0.30, -0.20, 0.84),
        origin_type="world",
        env_index=0,
        asset_name="receptive_object",
    )


# Eval after Stage 2
@configclass
class Arx5OSCFinetuneEvalCfg(Arx5RlStateCfg):
    events: FinetuneEvalEventCfg = FinetuneEvalEventCfg()
    actions: Arx5OSCEvalAction = Arx5OSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_ARX5.replace(prim_path="{ENV_REGEX_NS}/Robot")
