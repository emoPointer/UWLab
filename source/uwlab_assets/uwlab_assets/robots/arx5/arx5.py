# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ARX5 robot arm with parallel gripper.

The following configurations are available:

* :obj:`ARX5_ARTICULATION`: Base articulation (USD, init state).
* :obj:`IMPLICIT_ARX5`: Full robot with ImplicitActuator arm (no motor delay, for RL training).
* :obj:`EXPLICIT_ARX5`: Full robot with DelayedPDActuator arm (PD delay, for sim2real finetuning).

ARX5 joint layout (from MuJoCo XML):
    Arm:     joint1..6 (revolute, 6-DOF)
    Gripper: joint7/joint8 (prismatic slide, parallel jaw)

Joint limits:
    joint1: [-pi, pi]
    joint2: [0, pi]
    joint3: [0, pi]
    joint4: [-pi/2, pi/2]
    joint5: [-1.67, 1.67]
    joint6: [-pi/2, pi/2]
    gripper: [0.002, 0.044] m (slide)

Effort limits: 50 Nm for all arm joints (from MuJoCo actuator ctrlrange).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# USD paths — relative to this file
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ARX5_USD_PATH = os.path.join(_ASSETS_DIR, "arx5.usd")
ARX5_GRIPPER_USD_PATH = os.path.join(_ASSETS_DIR, "arx5_gripper.usd")

# -- Default joint positions ------------------------------------------------

ARX5_GRIPPER_DEFAULT_JOINT_POS = {
    "joint7": 0.02,
    "joint8": 0.02,
}

ARX5_DEFAULT_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 1.0,     # [0, pi] — raised
    "joint3": 1.0,     # [0, pi]
    "joint4": 0.0,     # [-pi/2, pi/2]
    "joint5": 0.0,     # [-1.67, 1.67]
    "joint6": 0.0,     # [-pi/2, pi/2]
    **ARX5_GRIPPER_DEFAULT_JOINT_POS,
}

# -- Joint limits -----------------------------------------------------------

ARX5_VELOCITY_LIMITS = {
    "joint1": 3.14,
    "joint2": 3.14,
    "joint3": 3.14,
    "joint4": 3.14,
    "joint5": 3.14,
    "joint6": 3.14,
}

ARX5_EFFORT_LIMITS = {
    "joint1": 50.0,
    "joint2": 50.0,
    "joint3": 50.0,
    "joint4": 50.0,
    "joint5": 50.0,
    "joint6": 50.0,
}

# -- Base articulation -------------------------------------------------------

ARX5_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ARX5_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Aligned with UR5e's IMPLICIT config (disable_gravity=True).
            # ARX5 now trains in zero-G, so the RelCartesianOSCAction PD control
            # law (no gravity compensation term) can hold any pose. Cubes and
            # other props are unaffected — disable_gravity is per-body, not
            # global, and props still fall under the world gravity vector.
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.01,
            angular_damping=0.01,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos=ARX5_DEFAULT_JOINT_POS,
    ),
    soft_joint_pos_limit_factor=0.9,
)

# -- Standalone gripper (for grasp sampling) ---------------------------------

ARX5_GRIPPER = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Arx5Gripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ARX5_GRIPPER_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.1),
        rot=(1, 0, 0, 0),
        joint_pos=ARX5_GRIPPER_DEFAULT_JOINT_POS,
    ),
    actuators={
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[78]"],
            stiffness=1000.0,
            damping=50.0,
            effort_limit=50.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)

# -- Implicit actuator (for RL training, no motor delay) --------------------

IMPLICIT_ARX5 = ARX5_ARTICULATION.copy()  # type: ignore
IMPLICIT_ARX5.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["joint[1-6]"],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=ARX5_EFFORT_LIMITS,
        velocity_limit_sim=ARX5_VELOCITY_LIMITS,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["joint[78]"],
        stiffness=1000.0,
        damping=50.0,
        effort_limit=50.0,
        velocity_limit=0.5,
    ),
}

# -- Explicit actuator (for sim2real finetuning, with PD delay) -------------

EXPLICIT_ARX5 = ARX5_ARTICULATION.copy()  # type: ignore
EXPLICIT_ARX5.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["joint[1-6]"],
        stiffness=0.0,
        damping=0.0,
        effort_limit=ARX5_EFFORT_LIMITS,
        effort_limit_sim=ARX5_EFFORT_LIMITS,
        velocity_limit=ARX5_VELOCITY_LIMITS,
        velocity_limit_sim=ARX5_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["joint[78]"],
        stiffness=1000.0,
        damping=50.0,
        effort_limit=50.0,
        velocity_limit=0.5,
    ),
}
