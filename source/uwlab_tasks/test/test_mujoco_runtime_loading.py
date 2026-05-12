from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]


def test_mujoco_can_load_static_scene_and_robot_models():
    import mujoco

    expected = {
        "mujoco_arx5/models/arx5_tabletop_static.xml": {"ncam": 1, "nu": 0},
        "mujoco_arx5/models/arx5_robot.xml": {"ncam": 1, "nu": 8},
        "mujoco_arx5/models/arx5_robosuite_tabletop_dynamic.xml": {"ncam": 2, "nu": 8},
    }
    for relative_path, values in expected.items():
        model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / relative_path))
        assert model.ncam == values["ncam"]
        assert model.nu == values["nu"]
