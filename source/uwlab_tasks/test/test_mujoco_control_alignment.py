from __future__ import annotations

import tomllib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_arx5.control_alignment import (
    binary_gripper_targets,
    control_period_seconds,
    scale_arm_action,
)


CONTROL_TOML = REPO_ROOT / "mujoco_arx5" / "config" / "control_alignment.toml"


def _load_control_config() -> dict:
    assert CONTROL_TOML.exists()
    return tomllib.loads(CONTROL_TOML.read_text())


def test_control_timing_matches_current_isaac_decimation():
    cfg = _load_control_config()

    assert cfg["timing"]["physics_dt"] == pytest.approx(1 / 120.0)
    assert cfg["timing"]["decimation"] == 12
    assert cfg["timing"]["control_dt"] == pytest.approx(0.1)
    assert cfg["timing"]["control_frequency_hz"] == pytest.approx(10.0)
    assert control_period_seconds() == pytest.approx(0.1)


def test_arm_action_scaling_matches_current_arx5_osc_modes():
    cfg = _load_control_config()

    assert cfg["arm"]["body"] == "link6"
    assert cfg["arm"]["joints"] == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    assert cfg["arm"]["eval_position_scale"] == pytest.approx(0.01)
    assert cfg["arm"]["eval_orientation_scale"] == pytest.approx(0.2)
    assert cfg["arm"]["train_position_scale"] == pytest.approx(0.02)
    assert cfg["arm"]["train_orientation_scale"] == pytest.approx(0.2)
    assert cfg["arm"]["gravity_compensation"] is False
    assert cfg["arm"]["inertial_dynamics_decoupling"] is True
    assert cfg["arm"]["nullspace_control"] == "none"

    assert scale_arm_action([1, -1, 2, 0.5, -0.5, 1.0], mode="eval") == pytest.approx(
        [0.01, -0.01, 0.02, 0.1, -0.1, 0.2]
    )
    assert scale_arm_action([1, -1, 2, 0.5, -0.5, 1.0], mode="train") == pytest.approx(
        [0.02, -0.02, 0.04, 0.1, -0.1, 0.2]
    )


def test_binary_gripper_mapping_matches_isaac_float_sign_convention():
    cfg = _load_control_config()

    assert cfg["gripper"]["action_dim"] == 1
    assert cfg["gripper"]["float_sign_convention"] == "negative_close_nonnegative_open"
    assert cfg["gripper"]["open"]["joint7"] == pytest.approx(0.044)
    assert cfg["gripper"]["open"]["joint8"] == pytest.approx(0.044)
    assert cfg["gripper"]["close"]["joint7"] == pytest.approx(0.002)
    assert cfg["gripper"]["close"]["joint8"] == pytest.approx(0.002)

    assert binary_gripper_targets(-0.1) == {"joint7": 0.002, "joint8": 0.002}
    assert binary_gripper_targets(0.0) == {"joint7": 0.044, "joint8": 0.044}
    assert binary_gripper_targets(2.8) == {"joint7": 0.044, "joint8": 0.044}


def test_actuator_limits_match_isaac_arx5_articulation_config():
    cfg = _load_control_config()

    assert cfg["actuators"]["arm_effort_limit"] == pytest.approx(50.0)
    assert cfg["actuators"]["gripper_position_kp"] == pytest.approx(1000.0)
    assert cfg["actuators"]["gripper_position_kv"] == pytest.approx(50.0)
    assert cfg["actuators"]["gripper_velocity_limit"] == pytest.approx(0.5)
