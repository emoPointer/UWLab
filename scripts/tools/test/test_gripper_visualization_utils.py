import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts_v2" / "tools"))

from gripper_visualization_utils import resolve_gripper_joint_indices


def test_resolve_gripper_joint_indices_supports_robotiq_finger_joint():
    indices = resolve_gripper_joint_indices(["shoulder", "finger_joint"], {"finger_joint": 0.8})

    assert indices == [1]


def test_resolve_gripper_joint_indices_supports_arx5_parallel_jaws():
    indices = resolve_gripper_joint_indices(["joint1", "joint7", "joint8"], {"joint7": 0.002, "joint8": 0.002})

    assert indices == [1, 2]
