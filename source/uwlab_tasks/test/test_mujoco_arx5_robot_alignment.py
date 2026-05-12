from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
ROBOT_XML = REPO_ROOT / "mujoco_arx5" / "models" / "arx5_robot.xml"


def _parse_robot() -> ET.Element:
    assert ROBOT_XML.exists()
    return ET.parse(ROBOT_XML).getroot()


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    elem = root.find(f".//{tag}[@name='{name}']")
    assert elem is not None, f"Missing <{tag} name={name!r}>"
    return elem


def _floats(elem: ET.Element, attr: str) -> tuple[float, ...]:
    return tuple(float(value) for value in elem.attrib[attr].split())


def test_robot_model_records_isaac_base_pose_and_mesh_assets():
    root = _parse_robot()

    assert root.attrib["model"] == "arx5_robot_alignment"
    assert root.find("compiler").attrib["meshdir"] == "../../source/uwlab_assets/uwlab_assets/robots/arx5/assets"
    assert root.find("option").attrib["integrator"] == "implicitfast"

    arx5_root = _named(root, "body", "arx5")
    assert _floats(arx5_root, "pos") == (-0.535, -0.21, 0.8)
    assert _floats(arx5_root, "quat") == (1.0, 0.0, 0.0, 0.0)
    assert arx5_root.attrib["gravcomp"] == "1"

    expected_meshes = {
        "base_link_mesh": "meshes/base_link.obj",
        "link1_mesh": "meshes/link1.obj",
        "link2_mesh": "meshes/link2.obj",
        "link3_mesh": "meshes/link3.obj",
        "link4_mesh": "meshes/link4.obj",
        "link5_mesh": "meshes/link5.obj",
        "link6_mesh": "meshes/link6.obj",
        "link7_mesh": "meshes/link7.obj",
        "link8_mesh": "meshes/link8.obj",
    }
    for mesh_name, file_name in expected_meshes.items():
        assert _named(root, "mesh", mesh_name).attrib["file"] == file_name

    assert _named(root, "geom", "camera_base_visual").attrib["type"] == "box"
    assert _named(root, "geom", "camera_visual").attrib["type"] == "box"


def test_robot_joint_chain_matches_current_arx5_urdf_and_limits():
    root = _parse_robot()

    expected = {
        "joint1": {"body": "link1", "pos": (0.0, 0.0, 0.0605), "axis": (0.0, 0.0, 1.0), "range": (-10.0, 10.0)},
        "joint2": {"body": "link2", "pos": (0.02, 0.0, 0.04), "axis": (0.0, 1.0, 0.0), "range": (-10.0, 10.0)},
        "joint3": {"body": "link3", "pos": (-0.264, 0.0, 0.0), "axis": (0.0, 1.0, 0.0), "range": (-10.0, 10.0)},
        "joint4": {"body": "link4", "pos": (0.245, 0.0, -0.056), "axis": (0.0, 1.0, 0.0), "range": (-10.0, 10.0)},
        "joint5": {"body": "link5", "pos": (0.06775, 0.0005, -0.0865), "axis": (0.0, 0.0, 1.0), "range": (-10.0, 10.0)},
        "joint6": {"body": "link6", "pos": (0.02895, 0.0, 0.0865), "axis": (1.0, 0.0, 0.0), "range": (-1.5708, 1.5708)},
        "joint7": {
            "body": "link7",
            "pos": (0.08657, 0.024896, -0.0002436),
            "axis": (0.0, 1.0, 0.0),
            "range": (0.0, 0.044),
        },
        "joint8": {
            "body": "link8",
            "pos": (0.08657, -0.0249, -0.00024366),
            "axis": (0.0, -1.0, 0.0),
            "range": (0.0, 0.044),
        },
    }

    for joint_name, cfg in expected.items():
        body = _named(root, "body", cfg["body"])
        joint = _named(body, "joint", joint_name)
        assert _floats(body, "pos") == cfg["pos"]
        assert _floats(joint, "axis") == cfg["axis"]
        assert _floats(joint, "range") == cfg["range"]

    assert _named(root, "body", "link3").attrib["euler"] == "3.1416 0 0"
    assert _named(root, "body", "link6").attrib["euler"] == "-3.1416 0 0"


def test_robot_gripper_offset_camera_anchor_and_default_keyframe():
    root = _parse_robot()

    link6 = _named(root, "body", "link6")
    grasp_point = link6.find("./site[@name='grasp_point']")
    assert grasp_point is not None
    assert _floats(grasp_point, "pos") == (0.145, 0.0, 0.0)

    camera_base = _named(root, "body", "camera_base")
    assert _floats(camera_base, "pos") == (0.057, 0.0, 0.0)
    assert camera_base.attrib["euler"] == "0 0 -3.1416"

    camera = _named(root, "body", "camera")
    assert _floats(camera, "pos") == (-0.0275, 0.0, 0.05)
    assert camera.attrib["euler"] == "0 0.3491 3.141592653589793"

    wrist_camera = _named(root, "camera", "wrist_camera")
    assert _floats(wrist_camera, "pos") == (0.0, 0.0, 0.0)
    assert _floats(wrist_camera, "quat") == (1.0, 0.0, 0.0, 0.0)
    assert float(wrist_camera.attrib["fovy"]) == 42.47

    key = _named(root, "key", "isaac_default")
    assert _floats(key, "qpos") == (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.02, 0.02)


def test_robot_exposes_mujoco_actuators_matching_isaac_control_contract():
    root = _parse_robot()

    for joint_name in [f"joint{i}" for i in range(1, 7)]:
        motor = _named(root, "motor", f"{joint_name}_torque")
        assert motor.attrib["joint"] == joint_name
        assert _floats(motor, "ctrlrange") == (-50.0, 50.0)

    for joint_name in ("joint7", "joint8"):
        actuator = _named(root, "position", f"{joint_name}_position")
        assert actuator.attrib["joint"] == joint_name
        assert _floats(actuator, "ctrlrange") == (0.002, 0.044)
        assert float(actuator.attrib["kp"]) == 1000.0
        assert float(actuator.attrib["kv"]) == 50.0


def test_robot_uses_robosuite_box_collision_primitives_instead_of_visual_mesh_collisions():
    root = _parse_robot()

    for geom_name in [
        "base_link_visual",
        "link1_visual",
        "link2_visual",
        "link3_visual",
        "link4_visual",
        "link5_visual",
        "link6_visual",
        "link7_visual",
        "link8_visual",
    ]:
        geom = _named(root, "geom", geom_name)
        assert geom.attrib["type"] == "mesh"
        assert geom.attrib["contype"] == "0"
        assert geom.attrib["conaffinity"] == "0"

    expected_boxes = {
        "base_link_collision": {"pos": (0.0, 0.0, 0.03), "size": (0.03, 0.03, 0.03)},
        "link1_collision": {"pos": (0.0, 0.0, 0.02), "size": (0.03, 0.03, 0.03)},
        "link2_collision": {"pos": (-0.132, 0.0, 0.0), "size": (0.14, 0.02, 0.02)},
        "link3_collision": {"pos": (0.15, 0.0, -0.055), "size": (0.02, 0.02, 0.09)},
        "link4_collision": {"pos": (0.04, 0.004, -0.03), "size": (0.03, 0.03, 0.03)},
        "link5_collision": {"pos": (0.004, 0.0, 0.055), "size": (0.025, 0.025, 0.03)},
        "link6_collision": {"pos": (0.04, 0.0, 0.0), "size": (0.03, 0.02, 0.02)},
        "link7_collision": {"pos": (0.035, -0.02, 0.0), "size": (0.04, 0.005, 0.015)},
        "link8_collision": {"pos": (0.035, 0.02, 0.0), "size": (0.04, 0.005, 0.015)},
    }
    for geom_name, cfg in expected_boxes.items():
        geom = _named(root, "geom", geom_name)
        assert geom.attrib["type"] == "box"
        assert _floats(geom, "pos") == cfg["pos"]
        assert _floats(geom, "size") == cfg["size"]

    for geom_name in [
        "base_link_collision",
        "link1_collision",
        "link2_collision",
        "link3_collision",
        "link4_collision",
        "link5_collision",
        "link6_collision",
    ]:
        geom = _named(root, "geom", geom_name)
        assert geom.attrib["contype"] == "0"
        assert geom.attrib["conaffinity"] == "0"

    assert _named(root, "geom", "link7_collision").attrib["contype"] == "1"
    assert _named(root, "geom", "link7_collision").attrib["conaffinity"] == "2"
    assert _named(root, "geom", "link8_collision").attrib["contype"] == "1"
    assert _named(root, "geom", "link8_collision").attrib["conaffinity"] == "2"
