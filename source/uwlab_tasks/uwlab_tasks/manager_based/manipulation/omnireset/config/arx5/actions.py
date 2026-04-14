# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniReset action configurations for ARX5.

Two OSC action families are provided and can be swapped at the task config
level:

1. ``ARX5_OSC_TRAIN`` / ``ARX5_OSC_EVAL`` — IsaacLab's built-in
   ``OperationalSpaceControllerActionCfg``. Uses the PhysX Jacobian, with
   inertial decoupling and gravity compensation. No analytical kinematics
   required. Good default for RL training that stays in-sim.
2. ``ARX5_REL_OSC_TRAIN`` / ``ARX5_REL_OSC_EVAL`` — custom
   ``RelCartesianOSCActionCfg`` sharing the same controller class as UR5e,
   but with an ARX5-specific ``jacobian_fn`` loaded from
   ``uwlab_assets.robots.arx5.kinematics``. Use this when sim2real
   alignment requires the same analytical Jacobian formulation on both
   sim and real sides.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.arx5.actions import ARX5_GRIPPER_BINARY_ACTIONS
from uwlab_assets.robots.arx5.kinematics import compute_jacobian_analytical as _arx5_jacobian_fn

from ...mdp.actions.actions_cfg import RelCartesianOSCActionCfg

# -- OSC action for training (soft gains, large action scale) ----------------

ARX5_OSC_TRAIN = mdp.OperationalSpaceControllerActionCfg(
    asset_name="robot",
    joint_names=["joint[1-6]"],
    body_name="link6",
    controller_cfg=OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        motion_control_axes_task=(1, 1, 1, 1, 1, 1),
        # Soft gains for training — curriculum can ramp to stiff later
        motion_stiffness_task=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
        motion_damping_ratio_task=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
        gravity_compensation=True,
        inertial_dynamics_decoupling=True,
        nullspace_control="none",
    ),
    position_scale=0.02,
    orientation_scale=0.2,
)

# -- OSC action for eval / sim2real (stiff gains, small action scale) --------

ARX5_OSC_EVAL = mdp.OperationalSpaceControllerActionCfg(
    asset_name="robot",
    joint_names=["joint[1-6]"],
    body_name="link6",
    controller_cfg=OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        motion_control_axes_task=(1, 1, 1, 1, 1, 1),
        motion_stiffness_task=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
        motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        gravity_compensation=True,
        inertial_dynamics_decoupling=True,
        nullspace_control="none",
    ),
    position_scale=0.01,
    orientation_scale=0.2,
)


@configclass
class Arx5OSCTrainAction:
    """Train-time action: soft OSC + binary gripper."""

    arm = ARX5_OSC_TRAIN
    gripper = ARX5_GRIPPER_BINARY_ACTIONS


@configclass
class Arx5OSCEvalAction:
    """Eval-time action: stiff OSC + binary gripper."""

    arm = ARX5_OSC_EVAL
    gripper = ARX5_GRIPPER_BINARY_ACTIONS


# ---------------------------------------------------------------------------
# Analytical-Jacobian OSC (shared class with UR5e, ARX5-specific jacobian_fn)
# ---------------------------------------------------------------------------
# Torque limits match ARX5_EFFORT_LIMITS (50 Nm across all arm joints).
# Stiffness / damping mirror the official-OSC soft/stiff presets above.

ARX5_REL_OSC_TRAIN = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["joint[1-6]"],
    body_name="link6",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
    torque_limit=(50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
    jacobian_fn=_arx5_jacobian_fn,
)

ARX5_REL_OSC_EVAL = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["joint[1-6]"],
    body_name="link6",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
    jacobian_fn=_arx5_jacobian_fn,
)


@configclass
class Arx5RelOSCTrainAction:
    """Train-time action: custom analytical-Jacobian OSC + binary gripper."""

    arm = ARX5_REL_OSC_TRAIN
    gripper = ARX5_GRIPPER_BINARY_ACTIONS


@configclass
class Arx5RelOSCEvalAction:
    """Eval-time action: custom analytical-Jacobian OSC (stiff gains) + binary gripper."""

    arm = ARX5_REL_OSC_EVAL
    gripper = ARX5_GRIPPER_BINARY_ACTIONS


# ---------------------------------------------------------------------------
# Unscaled analytical-Jacobian OSC (for system identification scripts)
# ---------------------------------------------------------------------------
# scale=1 → action = raw Cartesian delta (m, rad).  Stiff eval gains so the
# controller tracks aggressively and dynamics mismatch dominates the error.

ARX5_REL_OSC_UNSCALED = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["joint[1-6]"],
    body_name="link6",
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
    jacobian_fn=_arx5_jacobian_fn,
)


@configclass
class Arx5SysidOSCAction:
    """Unscaled arm action (Cartesian delta) + binary gripper. For sysid env / scripts."""

    arm = ARX5_REL_OSC_UNSCALED
    gripper = ARX5_GRIPPER_BINARY_ACTIONS
