from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
STATIC_SCENE_XML = REPO_ROOT / "mujoco_arx5" / "models" / "arx5_tabletop_static.xml"


def _parse_scene() -> ET.Element:
    assert STATIC_SCENE_XML.exists()
    return ET.parse(STATIC_SCENE_XML).getroot()


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    elem = root.find(f".//{tag}[@name='{name}']")
    assert elem is not None, f"Missing <{tag} name={name!r}>"
    return elem


def _floats(elem: ET.Element, attr: str) -> tuple[float, ...]:
    return tuple(float(value) for value in elem.attrib[attr].split())


def test_static_scene_records_aligned_workspace_and_reference_values():
    root = _parse_scene()

    assert root.attrib["model"] == "arx5_robosuite_tabletop_static"
    assert root.find("compiler").attrib["angle"] == "radian"

    workspace_x = _named(root, "numeric", "workspace_x_range")
    workspace_y = _named(root, "numeric", "workspace_y_range")
    assert _floats(workspace_x, "data") == (-0.4, -0.2)
    assert _floats(workspace_y, "data") == (-0.3, -0.1)

    robot_pose = _named(root, "numeric", "arx5_base_pose_reference")
    assert _floats(robot_pose, "data") == (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0)

    assert root.find(".//body[@name='arx5']") is None
    assert root.find(".//include") is None


def test_static_scene_matches_robosuite_table_and_external_camera():
    root = _parse_scene()

    table = _named(root, "body", "table")
    assert _floats(table, "pos") == (0.0, 0.0, 0.775)

    table_collision = _named(root, "geom", "table_collision")
    assert table_collision.attrib["type"] == "box"
    assert _floats(table_collision, "size") == (0.6, 0.45, 0.025)
    assert _floats(table_collision, "friction") == (1.0, 0.005, 0.0001)

    table_visual = _named(root, "geom", "table_visual")
    assert table_visual.attrib["material"] == "table_silver"
    assert table_visual.attrib["contype"] == "0"
    assert table_visual.attrib["conaffinity"] == "0"

    table_top = _named(root, "site", "table_top")
    assert _floats(table_top, "pos") == (0.0, 0.0, 0.025)

    external_cam = _named(root, "body", "external_cam")
    assert _floats(external_cam, "pos") == (0.517, 0.327, 1.364)
    assert _floats(external_cam, "quat") == (0.3604, 0.203, 0.5, 0.7609)

    camera = _named(root, "camera", "external_camera")
    assert camera.attrib["mode"] == "fixed"
    assert _floats(camera, "pos") == (0.0, 0.0, 0.0)
    assert float(camera.attrib["fovy"]) == 42.47


def test_static_scene_has_isaac_aligned_visual_backdrop_panels():
    root = _parse_scene()

    expected_panels = {
        "curtain_back": {
            "pos": (-1.1, 0.0, 0.519),
            "size": (0.005, 0.8, 1.0625),
            "quat": (1.0, 0.0, 0.0, 0.0),
        },
        "curtain_left": {
            "pos": (-0.05, 0.8, 0.519),
            "size": (0.005, 1.05, 1.0625),
            "quat": (0.707, 0.0, 0.0, -0.707),
        },
        "curtain_right": {
            "pos": (-0.05, -0.8, 0.519),
            "size": (0.005, 1.05, 1.0625),
            "quat": (0.707, 0.0, 0.0, -0.707),
        },
    }
    for name, expected in expected_panels.items():
        panel = _named(root, "geom", name)
        assert panel.attrib["type"] == "box"
        assert panel.attrib["contype"] == "0"
        assert panel.attrib["conaffinity"] == "0"
        assert panel.attrib["material"] == "curtain_mat"
        assert _floats(panel, "pos") == expected["pos"]
        assert _floats(panel, "size") == expected["size"]
        assert _floats(panel, "quat") == expected["quat"]


def test_static_scene_has_semantic_cube_colors_inside_workspace():
    root = _parse_scene()

    red = _named(root, "material", "receptive_cube_red")
    green = _named(root, "material", "insertive_cube_green")
    assert _floats(red, "rgba") == (1.0, 0.0, 0.0, 1.0)
    assert _floats(green, "rgba") == (0.0, 1.0, 0.0, 1.0)

    receptive = _named(root, "body", "receptive_cube")
    insertive = _named(root, "body", "insertive_cube")
    assert _floats(receptive, "pos") == (-0.3, -0.2, 0.84)
    assert _floats(insertive, "pos") == (-0.3, -0.2, 0.94)

    receptive_geom = _named(root, "geom", "receptive_cube_collision")
    insertive_geom = _named(root, "geom", "insertive_cube_collision")
    assert receptive_geom.attrib["material"] == "receptive_cube_red"
    assert insertive_geom.attrib["material"] == "insertive_cube_green"
    assert _floats(receptive_geom, "size") == (0.04, 0.04, 0.04)
    assert _floats(insertive_geom, "size") == (0.04, 0.04, 0.04)

    expected_corners = {
        "workspace_xmin_ymin": (-0.4, -0.3, 0.805),
        "workspace_xmin_ymax": (-0.4, -0.1, 0.805),
        "workspace_xmax_ymin": (-0.2, -0.3, 0.805),
        "workspace_xmax_ymax": (-0.2, -0.1, 0.805),
    }
    for name, pos in expected_corners.items():
        site = _named(root, "site", name)
        assert _floats(site, "pos") == pos
