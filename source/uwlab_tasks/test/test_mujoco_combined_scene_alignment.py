from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).parents[3]
COMBINED_XML = REPO_ROOT / "mujoco_arx5" / "models" / "arx5_robosuite_tabletop.xml"


def _parse_scene() -> ET.Element:
    assert COMBINED_XML.exists()
    return ET.parse(COMBINED_XML).getroot()


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    elem = root.find(f".//{tag}[@name='{name}']")
    assert elem is not None, f"Missing <{tag} name={name!r}>"
    return elem


def _floats(elem: ET.Element, attr: str) -> tuple[float, ...]:
    return tuple(float(value) for value in elem.attrib[attr].split())


def test_combined_scene_uses_current_uwlab_world_poses():
    root = _parse_scene()

    assert root.attrib["model"] == "arx5_robosuite_tabletop"

    robot = _named(root, "body", "arx5")
    table = _named(root, "body", "table")
    receptive = _named(root, "body", "receptive_cube")
    insertive = _named(root, "body", "insertive_cube")
    external_cam = _named(root, "body", "external_cam")

    assert _floats(robot, "pos") == (-0.535, -0.21, 0.8)
    assert _floats(robot, "quat") == (1.0, 0.0, 0.0, 0.0)
    assert _floats(table, "pos") == (0.0, 0.0, 0.775)
    assert _floats(receptive, "pos") == (-0.3, -0.2, 0.84)
    assert _floats(insertive, "pos") == (-0.3, -0.2, 0.87)
    assert _floats(external_cam, "pos") == (0.517, 0.327, 1.364)
    assert _floats(external_cam, "quat") == (0.3604, 0.203, 0.5, 0.7609)


def test_combined_scene_contains_workspace_and_cameras():
    root = _parse_scene()

    assert _floats(_named(root, "numeric", "workspace_x_range"), "data") == (-0.4, -0.2)
    assert _floats(_named(root, "numeric", "workspace_y_range"), "data") == (-0.3, -0.1)

    assert _named(root, "camera", "external_camera").attrib["mode"] == "fixed"
    assert _named(root, "camera", "wrist_camera").attrib["mode"] == "fixed"

    assert _named(root, "material", "receptive_cube_red").attrib["rgba"] == "1.0 0.0 0.0 1.0"
    assert _named(root, "material", "insertive_cube_green").attrib["rgba"] == "0.0 1.0 0.0 1.0"

    assert root.find(".//actuator") is None


def test_combined_scene_defaults_to_current_isaac_joint_state():
    root = _parse_scene()

    expected_ref = {
        "joint1": "0.0",
        "joint2": "1.0",
        "joint3": "1.0",
        "joint4": "0.0",
        "joint5": "0.0",
        "joint6": "0.0",
        "joint7": "0.02",
        "joint8": "0.02",
    }
    for joint_name, ref in expected_ref.items():
        assert _named(root, "joint", joint_name).attrib["ref"] == ref


def test_combined_scene_visual_alignment_geoms_do_not_create_robot_self_contacts():
    root = _parse_scene()

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
        "receptive_cube_collision",
        "insertive_cube_collision",
    ]:
        geom = _named(root, "geom", geom_name)
        assert geom.attrib["contype"] == "0"
        assert geom.attrib["conaffinity"] == "0"

    assert _named(root, "body", "receptive_cube").find("./freejoint") is None
    assert _named(root, "body", "insertive_cube").find("./freejoint") is None


def test_combined_scene_has_robosuite_simplified_robot_collision_boxes():
    root = _parse_scene()

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

    assert _named(root, "geom", "link7_collision").attrib["contype"] == "1"
    assert _named(root, "geom", "link7_collision").attrib["conaffinity"] == "2"
    assert _named(root, "geom", "link8_collision").attrib["contype"] == "1"
    assert _named(root, "geom", "link8_collision").attrib["conaffinity"] == "2"


def test_mujoco_runtime_can_load_combined_scene():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(COMBINED_XML))
    assert model.ncam == 2
    assert model.nu == 0
    assert model.njnt == 8

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    assert data.qpos.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.02, 0.02]

    for _ in range(200):
        mujoco.mj_step(model, data)
    assert data.ncon == 0
    assert data.qvel.tolist() == [0.0] * model.nv
