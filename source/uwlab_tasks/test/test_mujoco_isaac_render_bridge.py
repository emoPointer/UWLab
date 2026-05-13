from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DYNAMIC_XML = REPO_ROOT / "mujoco_arx5" / "models" / "arx5_robosuite_tabletop_dynamic.xml"


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    matches = [element for element in root.iter(tag) if element.attrib.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def test_mujoco_render_mirror_applies_isaac_state_without_stepping():
    mujoco = pytest.importorskip("mujoco")

    from mujoco_arx5.isaac_render_bridge import IsaacRenderState, MuJoCoRenderMirror, Pose

    model = mujoco.MjModel.from_xml_path(str(DYNAMIC_XML))
    data = mujoco.MjData(model)
    mirror = MuJoCoRenderMirror(model, data)

    state = IsaacRenderState(
        robot_root=Pose(np.array([-0.52, -0.19, 0.8]), np.array([1.0, 0.0, 0.0, 0.0])),
        joint_positions={f"joint{i}": 0.1 * i for i in range(1, 7)} | {"joint7": 0.03, "joint8": 0.031},
        object_poses={
            "insertive_object": Pose(np.array([-0.25, -0.18, 0.9]), np.array([1.0, 0.0, 0.0, 0.0])),
            "receptive_object": Pose(np.array([-0.28, -0.2, 0.84]), np.array([1.0, 0.0, 0.0, 0.0])),
        },
    )

    time_before = data.time
    mirror.apply(state)

    assert data.time == pytest.approx(time_before)
    assert data.qpos[:8] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.03, 0.031])
    assert data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arx5")] == pytest.approx(
        [-0.52, -0.19, 0.8]
    )
    assert data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "insertive_cube")] == pytest.approx(
        [-0.25, -0.18, 0.9]
    )


def test_dynamic_render_xml_uses_isaac_cube_size_and_hidden_collision_group():
    root = ET.parse(DYNAMIC_XML).getroot()

    for geom_name in ("insertive_cube_visual", "receptive_cube_visual"):
        geom = _named(root, "geom", geom_name)
        assert geom.attrib["size"] == "0.02 0.02 0.02"
        assert geom.attrib["group"] == "1"

    for geom_name in (
        "insertive_cube_collision",
        "receptive_cube_collision",
        "base_link_collision",
        "link1_collision",
        "link2_collision",
        "link3_collision",
        "link4_collision",
        "link5_collision",
        "link6_collision",
        "link7_collision",
        "link8_collision",
    ):
        assert _named(root, "geom", geom_name).attrib["group"] == "2"


def test_isaac_lab_state_extractor_uses_scene_assets_and_named_joints():
    from mujoco_arx5.isaac_render_bridge import IsaacLabStateExtractor

    class DummyRobot:
        joint_names = ["joint2", "joint1"]

        def __init__(self):
            self.data = SimpleNamespace(
                root_pose_w=np.array([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]]),
                joint_pos=np.array([[2.0, 1.0]]),
            )

        def find_joints(self, names):
            return [self.joint_names.index(name) for name in names], names

    scene = {
        "robot": DummyRobot(),
        "insertive_object": SimpleNamespace(
            data=SimpleNamespace(root_pos_w=np.array([[0.1, 0.2, 0.3]]), root_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]]))
        ),
    }
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))

    state = IsaacLabStateExtractor(
        object_names=("insertive_object", "missing_object"),
        joint_names=("joint1", "joint2"),
    ).extract(env, step=7)

    assert state.step == 7
    assert state.robot_root.position == pytest.approx([0.0, 0.0, 0.8])
    assert state.joint_positions == pytest.approx({"joint1": 1.0, "joint2": 2.0})
    assert state.object_poses["insertive_object"].position == pytest.approx([0.1, 0.2, 0.3])
