# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configurations for the ARX5 robot.

ARX5 gripper uses prismatic slide joints (range: 0.002 ~ 0.044 m).
Binary open/close maps to the joint position limits.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)
from isaaclab.utils import configclass

# -- Gripper actions ---------------------------------------------------------

ARX5_GRIPPER_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["joint7", "joint8"],
    open_command_expr={
        "joint7": 0.044,
        "joint8": 0.044,
    },
    close_command_expr={
        "joint7": 0.002,
        "joint8": 0.002,
    },
)

ARX5_GRIPPER_POSITION_ACTIONS = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["joint[78]"],
    scale=0.021,   # (0.044 - 0.002) / 2
    offset=0.023,  # (0.044 + 0.002) / 2
    use_default_offset=False,
)


@configclass
class Arx5BinaryGripperAction:
    """Standalone binary gripper action (for grasp sampling)."""

    gripper = ARX5_GRIPPER_BINARY_ACTIONS
