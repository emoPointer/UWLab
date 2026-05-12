from pathlib import Path
from math import isclose, radians
from xml.etree import ElementTree

from pxr import Usd


ASSETS_DIR = Path(__file__).parents[1] / "uwlab_assets" / "robots" / "arx5" / "assets"


def _urdf_joint_limit(urdf_path: Path, joint_name: str) -> tuple[float, float]:
    root = ElementTree.parse(urdf_path).getroot()
    joint = root.find(f"./joint[@name='{joint_name}']")
    assert joint is not None
    limit = joint.find("limit")
    assert limit is not None
    return float(limit.attrib["lower"]), float(limit.attrib["upper"])


def test_arx5_joint6_position_limit_matches_real_joint_range():
    expected_urdf_radians = (-1.5708, 1.5708)
    expected_usd_degrees = (-90.0, 90.0)

    assert _urdf_joint_limit(ASSETS_DIR / "arx5.urdf", "joint6") == expected_urdf_radians
    assert _urdf_joint_limit(ASSETS_DIR / "arx5_colored.urdf", "joint6") == expected_urdf_radians

    stage = Usd.Stage.Open(str(ASSETS_DIR / "arx5.usd"))
    joint6 = stage.GetPrimAtPath("/X5A/joints/joint6")
    usd_lower = joint6.GetAttribute("physics:lowerLimit").Get()
    usd_upper = joint6.GetAttribute("physics:upperLimit").Get()

    assert isclose(usd_lower, expected_usd_degrees[0], abs_tol=1e-6)
    assert isclose(usd_upper, expected_usd_degrees[1], abs_tol=1e-6)
    assert isclose(radians(usd_lower), expected_urdf_radians[0], abs_tol=1e-4)
    assert isclose(radians(usd_upper), expected_urdf_radians[1], abs_tol=1e-4)
